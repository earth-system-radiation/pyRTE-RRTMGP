"""Command line interface for pyRTE-RRTMGP."""

import argparse
import os
import subprocess
import sys


def run_tests() -> None:
    """Run tests using pytest."""
    package_root = os.path.dirname(os.path.abspath(__file__))
    tests_path = os.path.join(package_root, "tests")

    if not os.path.exists(tests_path):
        print(f"Error: Test directory '{tests_path}' does not exist.")
        sys.exit(1)

    try:
        print("Running tests...")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--cov=pyrte_rrtmgp",
                "--cov-report=term:skip-covered",
                tests_path,
            ],
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


def run_code_coverage() -> None:
    """Run code coverage using pytest."""
    package_root = os.path.dirname(os.path.abspath(__file__))
    tests_path = os.path.join(package_root, "tests")

    if not os.path.exists(tests_path):
        print(f"Error: Test directory '{tests_path}' does not exist.")
        sys.exit(1)

    try:
        print("Running code coverage...")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--cov=pyrte_rrtmgp",
                "--cov-report=term",
                tests_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(result.stdout)
        print("Code coverage completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Code coverage failed!")
        print(e.stdout)
        print(e.stderr)
        sys.exit(e.returncode)


def main() -> None:
    """Run the pyRTE-RRTMGP command line interface."""
    parser = argparse.ArgumentParser(description="pyRTE-RRTMGP command line interface")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    run_tests_parser = subparsers.add_parser(
        "run_tests",
        help="Run the test suite",
    )
    run_tests_parser.set_defaults(func=run_tests)

    run_code_coverage_parser = subparsers.add_parser(
        "run_code_coverage",
        help="Run code coverage using pytest",
    )
    run_code_coverage_parser.set_defaults(func=run_code_coverage)

    args = parser.parse_args()

    if args.command:
        args.func()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
