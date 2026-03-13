"""Run the PureArc FastAPI scaffold locally.

Usage:
    cd backend
    python run_api.py
"""

from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
