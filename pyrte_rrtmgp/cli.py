import sys
import subprocess
import argparse


def run_tests():
    """Run tests using pytest."""
    try:
        print("Running tests...")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "pyrte_rrtmgp/tests"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(result.stdout)
        print("All tests passed successfully.")
    except subprocess.CalledProcessError as e:
        print("Tests failed!")
        print(e.stdout)
        print(e.stderr)
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="pyRTE-RRTMGP command line interface"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    run_tests_parser = subparsers.add_parser(
        "run_tests",
        help="Run the test suite",
    )
    run_tests_parser.set_defaults(func=run_tests)

    args = parser.parse_args()

    if args.command:
        args.func()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
