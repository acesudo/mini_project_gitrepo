"""
Module 7 — Server entry point

Start the API from the project root:

    uvicorn module7.main:app --reload

or simply:

    python module7/main.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from module7.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("module7.main:app", host="127.0.0.1", port=8000, reload=True)
