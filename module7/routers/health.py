"""GET /health — quick check that the server, database and gallery are alive."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text

from module6.database import get_db_session
from module7.dependencies import AppState, get_state
from module7.schemas import HealthOut

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthOut)
def health_check(state: AppState = Depends(get_state)):
    try:
        with get_db_session(state.manager.engine) as sess:
            sess.execute(text("SELECT 1"))
    except Exception as err:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {err}")

    return HealthOut(
        status="ok",
        database="connected",
        registered_faces=len(state.matcher.registered_embeddings),
        match_threshold=state.matcher.cosine_threshold,
    )
