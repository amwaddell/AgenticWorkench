#!/usr/bin/env python
"""
Simple test runner for when pytest is not available.
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_test_file(test_file: Path):
    """Run all test functions in a file."""
    print(f"\n{'=' * 70}")
    print(f"Running {test_file.name}")
    print("=" * 70)

    # Import the test module
    module_name = test_file.stem
    spec = __import__("importlib.util").util.spec_from_file_location(
        module_name, test_file
    )
    module = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find all test functions
    test_functions = [
        (name, getattr(module, name))
        for name in dir(module)
        if name.startswith("test_") and callable(getattr(module, name))
    ]

    passed = 0
    failed = 0
    errors = []

    for test_name, test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name} - FAILED")
            failed += 1
            errors.append((test_name, str(e), traceback.format_exc()))
        except Exception as e:
            print(f"✗ {test_name} - ERROR")
            failed += 1
            errors.append((test_name, str(e), traceback.format_exc()))

    return passed, failed, errors


def main():
    """Run all tests."""
    tests_dir = Path(__file__).parent.parent / "tests"
    test_files = sorted(tests_dir.glob("test_*.py"))

    if not test_files:
        print("No test files found!")
        return 1

    total_passed = 0
    total_failed = 0
    all_errors = []

    for test_file in test_files:
        passed, failed, errors = run_test_file(test_file)
        total_passed += passed
        total_failed += failed
        all_errors.extend(errors)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Test Summary: {total_passed} passed, {total_failed} failed")
    print("=" * 70)

    if all_errors:
        print("\nFailures:")
        for test_name, error_msg, tb in all_errors:
            print(f"\n{test_name}:")
            print(f"  {error_msg}")
            if "--verbose" in sys.argv or "-v" in sys.argv:
                print(f"\n{tb}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
