#!/usr/bin/env python
"""
Day 2 Setup Verification Script

Run this to verify your Day 2 setup is complete and working.
"""

import sys
from pathlib import Path

print("=" * 70)
print("Day 2 Setup Verification")
print("=" * 70)

# Check we're in the right directory
if not (Path.cwd() / "src" / "workbench").exists():
    print("\n❌ ERROR: Must run from project root directory")
    print("   Current directory:", Path.cwd())
    print("   Expected: agentic_workbench/")
    sys.exit(1)

print("\n✓ Running from correct directory")

# Check all required files exist
required_files = [
    "src/workbench/__init__.py",
    "src/workbench/core/__init__.py",
    "src/workbench/core/types.py",
    "src/workbench/core/interfaces.py",
    "src/workbench/core/config.py",
    "src/workbench/core/registry.py",
    "src/workbench/core/run_context.py",
    "src/workbench/observability/__init__.py",
    "src/workbench/observability/tracing.py",
    "src/workbench/observability/phoenix.py",
    "src/workbench/observability/logging.py",
    "src/workbench/observability/metrics.py",
    "src/workbench/observability/otel_stub.py",
    "configs/defaults.yaml",
    "scripts/run_tests.py",
    "scripts/trace_smoke_test.py",
    "tests/test_config_loads.py",
    "tests/test_registry_builds.py",
    "tests/test_jsonl_logging.py",
    "tests/test_metrics_writes.py",
    "tests/test_tracing_spans.py",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-phoenix.txt",
    "pyproject.toml",
]

print("\nChecking required files...")
missing_files = []
for filepath in required_files:
    path = Path(filepath)
    if path.exists():
        print(f"  ✓ {filepath}")
    else:
        print(f"  ❌ {filepath}")
        missing_files.append(filepath)

if missing_files:
    print(f"\n❌ ERROR: {len(missing_files)} files missing!")
    print("\nMissing files:")
    for f in missing_files:
        print(f"  - {f}")
    sys.exit(1)

print(f"\n✓ All {len(required_files)} required files present")

# Check Python can import modules
print("\nChecking Python imports...")

imports_to_test = [
    ("workbench", "workbench package"),
    ("workbench.core", "core module"),
    ("workbench.core.types", "types"),
    ("workbench.core.interfaces", "interfaces"),
    ("workbench.core.config", "config"),
    ("workbench.core.registry", "registry"),
    ("workbench.core.run_context", "run_context"),
    ("workbench.observability", "observability package"),
    ("workbench.observability.tracing", "tracing"),
    ("workbench.observability.phoenix", "phoenix"),
    ("workbench.observability.logging", "logging"),
    ("workbench.observability.metrics", "metrics"),
]

import_errors = []
for module_name, description in imports_to_test:
    try:
        __import__(module_name)
        print(f"  ✓ {description}")
    except ImportError as e:
        print(f"  ❌ {description}: {e}")
        import_errors.append((description, str(e)))

if import_errors:
    print(f"\n❌ ERROR: {len(import_errors)} import errors!")
    print("\nTo fix:")
    print("  1. Make sure you're in the project root")
    print("  2. Run: pip install -e .")
    print("  3. Run: pip install -r requirements.txt")
    print("\nErrors:")
    for desc, err in import_errors:
        print(f"  - {desc}: {err}")
    sys.exit(1)

print(f"\n✓ All {len(imports_to_test)} imports successful")

# Check specific functions are available
print("\nChecking key functions...")

checks = []
try:
    from workbench.core.config import load_config

    checks.append(("load_config", True, None))
except Exception as e:
    checks.append(("load_config", False, str(e)))

try:
    from workbench.core.run_context import run_context

    checks.append(("run_context", True, None))
except Exception as e:
    checks.append(("run_context", False, str(e)))

try:
    from workbench.observability import (
        get_logger,
        setup_tracing,
        start_span,
    )

    checks.append(("observability functions", True, None))
except Exception as e:
    checks.append(("observability functions", False, str(e)))

for name, success, error in checks:
    if success:
        print(f"  ✓ {name}")
    else:
        print(f"  ❌ {name}: {error}")

if not all(c[1] for c in checks):
    print("\n❌ ERROR: Some functions not available")
    sys.exit(1)

print("\n✓ All key functions available")

# Try to run a quick functional test
print("\nRunning quick functional test...")

try:
    from workbench.core.config import create_config_snapshot, load_config
    from workbench.core.run_context import run_context
    from workbench.observability import get_logger, setup_tracing, start_span

    # Load config
    config = load_config()
    snapshot = create_config_snapshot(config)

    # Setup tracing
    setup_tracing()

    # Create run context
    with run_context(snapshot, run_type="verification") as ctx:
        # Create a span
        with start_span("test_span"):
            # Get logger
            logger = get_logger()
            logger.log_event("verification", {"status": "success"})

    print("  ✓ Functional test passed")
    print(f"  ✓ Created run context: {ctx.run_id}")

except Exception as e:
    print(f"  ❌ Functional test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print("\n✅ All checks passed!")
print("\nNext steps:")
print("  1. Run tests: python scripts/run_tests.py")
print("  2. Run smoke test: python scripts/trace_smoke_test.py")
print("  3. (Optional) Install Phoenix: pip install -r requirements-phoenix.txt")
print("  4. (Optional) Start Phoenix: phoenix serve")
print("\nYour Day 2 setup is complete and working! 🎉")
