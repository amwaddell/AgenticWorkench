#!/usr/bin/env python
"""
Day 3 Setup Verification Script

Run this to verify your Day 3 setup is complete and working.
"""

import sys
from pathlib import Path

print("=" * 70)
print("Day 3 Setup Verification")
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
    "src/workbench/models/__init__.py",
    "src/workbench/models/language_model.py",
    "src/workbench/models/llamacpp_server.py",
    "src/workbench/models/token_counting.py",
    "src/workbench/models/errors.py",
    "scripts/start_model_server.sh",
    "notebooks/00_local_model_smoke.ipynb",
    "tests/test_language_model_unit.py",
    "tests/test_language_model_integration.py",
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
    ("workbench.models", "models package"),
    ("workbench.models.language_model", "language_model"),
    ("workbench.models.llamacpp_server", "llamacpp_server"),
    ("workbench.models.token_counting", "token_counting"),
    ("workbench.models.errors", "errors"),
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
    print("\nErrors:")
    for desc, err in import_errors:
        print(f"  - {desc}: {err}")
    sys.exit(1)

print(f"\n✓ All {len(imports_to_test)} imports successful")

# Check specific classes and functions are available
print("\nChecking key classes and functions...")

checks = []
try:
    from workbench.models import (
        LlamaCppServerModel,
    )

    checks.append(("ChatMessage", True, None))
    checks.append(("GenerationSettings", True, None))
    checks.append(("LlamaCppServerModel", True, None))
except Exception as e:
    checks.append(("Model classes", False, str(e)))

try:
    checks.append(("language_model utilities", True, None))
except Exception as e:
    checks.append(("language_model utilities", False, str(e)))

try:
    checks.append(("token_counting utilities", True, None))
except Exception as e:
    checks.append(("token_counting utilities", False, str(e)))

for name, success, error in checks:
    if success:
        print(f"  ✓ {name}")
    else:
        print(f"  ❌ {name}: {error}")

if not all(c[1] for c in checks):
    print("\n❌ ERROR: Some classes/functions not available")
    sys.exit(1)

print("\n✓ All key classes and functions available")

# Check dependencies
print("\nChecking dependencies...")

deps_to_check = [
    ("httpx", "httpx (HTTP client)"),
    ("pydantic", "pydantic (validation)"),
]

missing_deps = []
for dep, description in deps_to_check:
    try:
        __import__(dep)
        print(f"  ✓ {description}")
    except ImportError:
        print(f"  ❌ {description}")
        missing_deps.append(dep)

if missing_deps:
    print(f"\n⚠️  WARNING: {len(missing_deps)} dependencies missing")
    print("\nTo install:")
    print("  pip install -r requirements.txt")
    print("\nMissing:")
    for dep in missing_deps:
        print(f"  - {dep}")
    # Don't exit, this is just a warning
else:
    print("\n✓ All dependencies installed")

# Check model file exists (optional)
print("\nChecking for model file...")

model_dir = Path("data/models")
if model_dir.exists():
    model_files = list(model_dir.glob("*.gguf"))
    if model_files:
        print(f"  ✓ Found {len(model_files)} model file(s)")
        for mf in model_files:
            size_mb = mf.stat().st_size / (1024 * 1024)
            print(f"    - {mf.name} ({size_mb:.1f} MB)")
    else:
        print("  ⚠️  No .gguf model files found in data/models/")
        print("    Download a model to proceed with integration tests")
else:
    print("  ⚠️  data/models/ directory not found")
    print("    Create it and download a model file")

# Try to create a model instance
print("\nTrying to create model instance...")

try:
    from workbench.models import LlamaCppServerModel

    model = LlamaCppServerModel(
        base_url="http://localhost:8080", model_name="test", timeout_seconds=5.0
    )
    print("  ✓ Model instance created successfully")
    print(f"    Base URL: {model.base_url}")
    print(f"    Model name: {model.model_name}")

    # Close the client
    model.close()
except Exception as e:
    print(f"  ❌ Failed to create model instance: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Check if server is running (non-blocking)
print("\nChecking if model server is running...")

try:
    import httpx

    with httpx.Client(timeout=2.0) as client:
        response = client.get("http://localhost:8080/health")
        if response.status_code == 200:
            print("  ✓ Model server is running at http://localhost:8080")
        else:
            print(f"  ⚠️  Server responded but with status {response.status_code}")
except Exception:
    print("  ⚠️  Model server not running")
    print("    This is OK for now - start it when ready:")
    print("    bash scripts/start_model_server.sh")

# Summary
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

all_critical_passed = (
    not missing_files and not import_errors and all(c[1] for c in checks)
)

if all_critical_passed:
    print("\n✅ All critical checks passed!")
else:
    print("\n❌ Some critical checks failed - see errors above")
    sys.exit(1)

print("\nNext steps:")
print("\n1. If you haven't yet:")
print("   - Install llama-cpp-python: See DAY3_GUIDE.md Step 2")
print("   - Download a model: See DAY3_GUIDE.md Step 3")
print("   - Update start_model_server.sh with your model filename")

print("\n2. Start the model server:")
print("   bash scripts/start_model_server.sh")

print("\n3. Run tests:")
print("   # Unit tests (no server needed)")
print("   python -m pytest tests/test_language_model_unit.py -v")
print("")
print("   # Integration tests (server needed)")
print("   python -m pytest tests/test_language_model_integration.py -v -m integration")

print("\n4. Try the notebook:")
print("   jupyter notebook notebooks/00_local_model_smoke.ipynb")

print("\n5. (Optional) Start Phoenix for trace visualization:")
print("   pip install arize-phoenix arize-phoenix-otel")
print("   phoenix serve")
print("   # Then visit http://localhost:6006")

print("\n" + "=" * 70)
print("Your Day 3 setup is ready! 🚀")
print("=" * 70)
