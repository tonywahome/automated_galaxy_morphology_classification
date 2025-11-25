# api/routes/health.py
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

# Will be set by main.py
START_TIME = None
MODEL = None
MODEL_LOAD_TIME = None

@router.get("/health")
async def health_check():
    """Health check and uptime information."""
    uptime = (datetime.now() - START_TIME).total_seconds() if START_TIME else 0
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "uptime_formatted": f"{uptime // 3600:.0f}h {(uptime % 3600) // 60:.0f}m",
        "model_loaded": MODEL is not None,
        "model_load_time": MODEL_LOAD_TIME.isoformat() if MODEL_LOAD_TIME else None
    }
