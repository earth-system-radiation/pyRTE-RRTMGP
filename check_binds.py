"""Compare C function declarations with Pybind11 bindings.

Example usage:

```
python check_binds.py \
    --c_headers /path/to/rrtmgp_kernels.h /path/to/rte_kernels.h \
    --pybind /path/to/pybind_interface.cpp
```

"""

import argparse
import re


def extract_c_functions(file_paths):  # type: ignore
    """Extract C function names from the given header files.

    Args:
        file_paths (list of str): List of paths to C header files.

    Returns:
        set: A set of C function names.
    """
    functionNames = set()

    for filePath in file_paths:
        try:
            with open(filePath, "r", encoding="utf-8") as file:
                content = file.read()

            functionPattern = re.findall(r'extern "C" \{([\s\S]*?)\}', content)

            if not functionPattern:
                print(f"Warning: No 'extern \"C\"' block found in {filePath}")
                # Skip to the next file
                continue

            functions = re.findall(r"\bvoid\s+(\w+)\s*\(", functionPattern[0])
            functionNames.update(functions)
        except FileNotFoundError:
            print(f"Error: File not found: {filePath}")

    return functionNames


def extract_pybind_functions(pybind_file):  # type: ignore
    """Extract function names bound using Pybind11 from the given file.

    Args:
        pybind_file (str): Path to the Pybind11 binding file.

    Returns:
        set: A set of function names bound using Pybind11.
    """
    try:
        with open(pybind_file, "r", encoding="utf-8") as file:
            content = file.read()

        boundFunctions = set(re.findall(r'm\.def\("(\w+)"', content))
        return boundFunctions
    except FileNotFoundError:
        print(f"Error: Pybind11 file not found: {pybind_file}")
        return set()


def main():  # type: ignore
    """Compare C function declarations with Pybind11 bindings."""
    parser = argparse.ArgumentParser(
        description="Compare C function declarations with Pybind11 bindings."
    )
    parser.add_argument(
        "--c_headers",
        nargs="+",
        required=True,
        help="List of C header files to check.",
    )
    parser.add_argument(
        "--pybind",
        required=True,
        help="Path to the Pybind11 binding file.",
    )

    args = parser.parse_args()

    cFunctions = extract_c_functions(args.c_headers)
    pybindFunctions = extract_pybind_functions(args.pybind)

    missingPyBindings = cFunctions - pybindFunctions
    missingCBindings = pybindFunctions - cFunctions

    print(f"Total C functions found: {len(cFunctions)}")
    print(f"Total Py functions bound: {len(pybindFunctions)}")

    if missingCBindings:
        print("\nWarning: Missing C function bindings:")
        for func in sorted(missingCBindings):
            print(f"- {func}")

    if missingPyBindings:
        print("\nWarning: Missing Py function bindings:")
        for func in sorted(missingPyBindings):
            print(f"- {func}")
    else:
        print("\nAll C functions are properly bound.")


if __name__ == "__main__":
    main()
