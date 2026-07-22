from fastapi import APIRouter, Request, Response, status

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    # Always 200: inference must stay reachable even if the DB/breaker is degraded.
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict:
    model_ready = getattr(request.app.state, "model_ready", False)
    if not model_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"model_ready": model_ready}
