"""
examples/forecast_api_demo.py
==============================
Demonstrates how to start the reference forecast server and call it
programmatically from a scheduler.

Run the server in one terminal:
    uvicorn dirac_sim.api.server:app --port 8000

Then run this script in another:
    python examples/forecast_api_demo.py

Or run the full demo against the offline CSV (no server needed):
    python examples/forecast_api_demo.py --offline
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dirac_sim.api.forecast_client import ForecastClient


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def site_ids():
    from dirac_sim.core.site_model import SiteRegistry
    registry = SiteRegistry.from_json(DATA_DIR / "site_config.json")
    return [site.site_id for site in registry.all_sites()]


def demo_offline():
    print("=== Offline forecast demo ===")
    client = ForecastClient(
        series_ids=site_ids(),
        offline_csv=str(DATA_DIR / "forecast_baseline.csv"),
    )
    now = datetime(2025, 11, 20, 0, 0, tzinfo=timezone.utc)
    bundle = client.fetch_bundle(now)
    print(f"Tick time  : {bundle.tick_time.isoformat()}")
    print(f"1h records : {len(bundle.horizon_1h)}")
    print(f"24h records: {len(bundle.horizon_24h)}")
    print("\nSample 1h-ahead records:")
    for rec in bundle.horizon_1h[:4]:
        print(f"  {rec['series_id']:6s}  "
              f"energy={rec['energy_wh_pred']:7.1f} Wh  "
              f"cfp={rec['cfp_g_pred']:6.1f} g")


def demo_live(api_url: str):
    print(f"=== Live forecast demo ({api_url}) ===")
    client = ForecastClient(
        base_url=api_url,
        series_ids=site_ids(),
    )
    now = datetime.now(timezone.utc)
    bundle = client.fetch_bundle(now)
    print(f"Tick time  : {bundle.tick_time.isoformat()}")
    print(f"1h records : {len(bundle.horizon_1h)}")
    print(f"24h records: {len(bundle.horizon_24h)}")

    if bundle.raw_response:
        print("\nRaw response (first 500 chars):")
        print(json.dumps(bundle.raw_response, indent=2, default=str)[:500])


def demo_green_window():
    """Show how a scheduler uses forecast to find green windows."""
    print("\n=== Green window selection demo ===")
    from dirac_sim.core.site_model import SiteRegistry

    registry = SiteRegistry.from_json(DATA_DIR / "site_config.json")
    client = ForecastClient(
        series_ids=site_ids(),
        offline_csv=str(DATA_DIR / "forecast_baseline.csv"),
    )
    now = datetime(2025, 11, 20, 8, 0, tzinfo=timezone.utc)
    bundle = client.fetch_bundle(now)

    # Ingest forecast into registry
    for site in registry.all_sites():
        site.bulk_update(bundle.horizon_24h)

    print(f"Reference time: {now.isoformat()}")
    print(f"{'Site':6s}  {'Carbon now':>12s}  {'Green windows (next 6h)':30s}")
    print("-" * 55)
    for site in registry.all_sites():
        carbon_now = site.get_carbon(now)
        windows = site.green_windows(now, horizon_minutes=360,
                                     carbon_threshold=100.0)
        win_str = ", ".join(
            f"{w[0].strftime('%H:%M')}-{w[1].strftime('%H:%M')}"
            for w in windows[:3]
        ) or "none"
        print(f"{site.site_id:6s}  {carbon_now:10.1f} g  {win_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    if args.offline:
        demo_offline()
    else:
        demo_live(args.api_url)

    demo_green_window()


if __name__ == "__main__":
    main()
