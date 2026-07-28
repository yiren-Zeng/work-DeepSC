"""Dependency-free runner for the project's pytest-style CPU tests.

The test functions intentionally use plain assertions and no fixtures, so this
runner provides the same coverage on machines where pytest is not installed.
"""

import importlib
import inspect
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_MODULES = (
    "tests.test_residual_simvq_quantizer",
    "tests.test_residual_simvq_training_contract",
    "tests.test_residual_simvq_depth4_training",
    "tests.test_residual_simvq_depth_extension",
    "tests.test_adaptive_topk",
    "tests.test_adaptive_transport",
    "tests.test_rq_ema_quantizer",
    "tests.test_rq_ema_adaptive",
    "tests.test_rq_ema_depth_extension",
    "tests.test_rq_ema_integration",
    "tests.test_rq_ema_project_contract",
)


def main():
    failures = []
    passed = 0
    for module_name in TEST_MODULES:
        module = importlib.import_module(module_name)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_") or inspect.signature(function).parameters:
                continue
            test_id = f"{module_name}::{name}"
            try:
                function()
            except Exception as error:  # noqa: BLE001 - report every failed assertion.
                failures.append((test_id, error))
                print(f"FAIL {test_id}: {error}")
            else:
                passed += 1
                print(f"PASS {test_id}")

    print(f"\n{passed} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
