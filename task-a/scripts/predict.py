from task_a.cli import main

if __name__ == "__main__":
    import sys

    main(["predict", *sys.argv[1:]])
