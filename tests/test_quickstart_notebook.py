from pathlib import Path

import nbformat
from nbclient import NotebookClient


def test_quickstart_executes_from_study_root():
    root = Path(__file__).parents[1]
    notebook = nbformat.read(root / "notebooks/quickstart.ipynb", as_version=4)
    executed = NotebookClient(
        notebook, timeout=120, kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    ).execute()
    assert any(cell.get("execution_count") for cell in executed.cells)
