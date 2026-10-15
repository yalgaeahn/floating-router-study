import os
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if any(item.get_closest_marker(name) for name in ("converged", "expensive")):
            if os.environ.get("CIRCUIT_STUDY_RUN_EXPENSIVE") != "1":
                item.add_marker(pytest.mark.skip(reason="set CIRCUIT_STUDY_RUN_EXPENSIVE=1"))
