"""
dirac_sim.core.evaluator
========================
Official scoring for Task B.

Scores a submitted DispatchPlan (or completed SimulationReport) against the
FCFS baseline according to the three-objective formula in the challenge spec:

    Score = 1 - (1/3)(ΔE/E_ref + ΔC/C_ref + ΔM/M_ref)

where ΔX = X_ref - X_submission (positive = improvement).

Also computes the declaration-aware rank which rewards the declared primary
objective.

Usage
-----
>>> from dirac_sim.core.evaluator import Evaluator
>>> score = Evaluator.score(submission_report, baseline_report,
...                         declared_objective="carbon")
>>> print(score)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Optional

from dirac_sim.core.wms import SimulationReport


@dataclass
class ObjectiveScores:
    energy_wh_total: float
    cfp_g_total: float
    makespan_total: float
    deadlines_met: int
    n_jobs: int
    jobs_dispatched: int

    @property
    def deadline_rate(self) -> float:
        return self.deadlines_met / self.n_jobs if self.n_jobs else 0.0


@dataclass
class EvaluationResult:
    submission: ObjectiveScores
    baseline: ObjectiveScores
    declared_objective: str

    # normalised deltas (positive = improvement vs baseline)
    delta_energy_norm: float = 0.0
    delta_carbon_norm: float = 0.0
    delta_makespan_norm: float = 0.0

    pareto_score: float = 0.0
    declaration_score: float = 0.0
    penalty_deadline: float = 0.0
    final_score: float = 0.0

    def __str__(self) -> str:
        return (
            "=== Task B Evaluation ===\n"
            f"Declared objective : {self.declared_objective}\n"
            f"ΔEnergy (norm)     : {self.delta_energy_norm:+.4f}\n"
            f"ΔCarbon (norm)     : {self.delta_carbon_norm:+.4f}\n"
            f"ΔMakespan (norm)   : {self.delta_makespan_norm:+.4f}\n"
            f"Pareto score       : {self.pareto_score:.4f}\n"
            f"Declaration score  : {self.declaration_score:.4f}\n"
            f"Deadline penalty   : {self.penalty_deadline:.4f}\n"
            f"──────────────────────────────\n"
            f"Final score        : {self.final_score:.4f}\n"
        )

    def to_dict(self) -> dict:
        baseline = asdict(self.baseline)
        submission = asdict(self.submission)
        return {
            "declared_objective": self.declared_objective,
            "baseline": baseline,
            "submission": submission,
            "deltas": {
                "energy_wh_total": (
                    baseline["energy_wh_total"] - submission["energy_wh_total"]
                ),
                "cfp_g_total": (
                    baseline["cfp_g_total"] - submission["cfp_g_total"]
                ),
                "makespan_total": (
                    baseline["makespan_total"] - submission["makespan_total"]
                ),
                "energy_norm": self.delta_energy_norm,
                "carbon_norm": self.delta_carbon_norm,
                "makespan_norm": self.delta_makespan_norm,
                "deadline_rate": (
                    submission["deadlines_met"] / submission["n_jobs"]
                    - baseline["deadlines_met"] / baseline["n_jobs"]
                    if submission["n_jobs"] and baseline["n_jobs"] else 0.0
                ),
            },
            "scores": {
                "pareto": self.pareto_score,
                "declaration": self.declaration_score,
                "deadline_penalty": self.penalty_deadline,
                "final": self.final_score,
            },
        }


class Evaluator:
    """
    Official Task B evaluator.

    All scoring is deterministic given two SimulationReports.
    """

    # Weights for the three objectives in the Pareto score
    W_ENERGY = 1 / 3
    W_CARBON = 1 / 3
    W_MAKESPAN = 1 / 3

    # Bonus multiplier for declared-objective improvement
    DECLARATION_BONUS = 0.15

    # Penalty per percentage point of missed deadlines
    DEADLINE_PENALTY_RATE = 0.005

    @classmethod
    def score(
        cls,
        submission: SimulationReport,
        baseline: SimulationReport,
        declared_objective: str = "carbon",
    ) -> EvaluationResult:
        sub_scores = cls._extract_scores(submission)
        base_scores = cls._extract_scores(baseline)

        # Normalised deltas: positive means submission is better
        de = cls._norm_delta(base_scores.energy_wh_total,
                             sub_scores.energy_wh_total)
        dc = cls._norm_delta(base_scores.cfp_g_total,
                             sub_scores.cfp_g_total)
        dm = cls._norm_delta(base_scores.makespan_total,
                             sub_scores.makespan_total)

        pareto = 1.0 - (cls.W_ENERGY * (1 - de)
                        + cls.W_CARBON * (1 - dc)
                        + cls.W_MAKESPAN * (1 - dm))
        pareto = max(0.0, min(1.0, pareto))

        # Declaration-aware bonus
        declared_delta = {"energy": de, "carbon": dc, "makespan": dm}.get(
            declared_objective.lower(), 0.0)
        declaration_score = cls.DECLARATION_BONUS * max(0.0, declared_delta)

        # Deadline penalty
        missed_rate = 1.0 - sub_scores.deadline_rate
        penalty = cls.DEADLINE_PENALTY_RATE * missed_rate * 100

        final = max(0.0, pareto + declaration_score - penalty)

        result = EvaluationResult(
            submission=sub_scores,
            baseline=base_scores,
            declared_objective=declared_objective,
            delta_energy_norm=de,
            delta_carbon_norm=dc,
            delta_makespan_norm=dm,
            pareto_score=pareto,
            declaration_score=declaration_score,
            penalty_deadline=penalty,
            final_score=final,
        )
        return result

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_scores(report: SimulationReport) -> ObjectiveScores:
        queued_jobs = report.queue.all_jobs()
        if queued_jobs and report.end_time is not None:
            queued_jobs = [j for j in queued_jobs if j.arrival_time <= report.end_time]
        n = len(queued_jobs) if queued_jobs else len(report.records)
        if n == 0:
            return ObjectiveScores(0, 0, 0, 0, 0, 0)
        return ObjectiveScores(
            energy_wh_total=sum(r.energy_wh for r in report.records),
            cfp_g_total=sum(r.cfp_g for r in report.records),
            makespan_total=sum(r.makespan_minutes for r in report.records),
            deadlines_met=sum(1 for r in report.records if r.deadline_met),
            n_jobs=n,
            jobs_dispatched=len(report.records),
        )

    @staticmethod
    def _norm_delta(baseline_val: float, submission_val: float) -> float:
        """
        Normalised improvement: (baseline - submission) / baseline.
        Positive = submission is better (lower resource use).
        Clamped to [-1, 1].
        """
        if baseline_val == 0:
            return 0.0
        delta = (baseline_val - submission_val) / baseline_val
        return max(-1.0, min(1.0, delta))

    @classmethod
    def compare_with_oracle(
        cls,
        submission: SimulationReport,
        oracle: SimulationReport,
        baseline: SimulationReport,
        declared_objective: str = "carbon",
    ) -> dict:
        """
        Compute the forecast-to-dispatch value gap:
        how much of the oracle improvement does the submission capture?
        """
        sub_result = cls.score(submission, baseline, declared_objective)
        oracle_result = cls.score(oracle, baseline, declared_objective)

        gap = {
            "submission_score": sub_result.final_score,
            "oracle_score": oracle_result.final_score,
            "value_gap": oracle_result.final_score - sub_result.final_score,
            "capture_ratio": (
                sub_result.final_score / oracle_result.final_score
                if oracle_result.final_score > 0 else 0.0
            ),
        }
        return gap
