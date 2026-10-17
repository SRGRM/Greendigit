from task_a.cli import main

if __name__ == "__main__":
    import sys

    main(["prepare-data", *sys.argv[1:]])
