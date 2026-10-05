from fastapi import APIRouter, HTTPException

from backend.config import get_settings
from backend.mcp_client.gateway import MCPToolGateway
from backend.models.registry import get_model_registry
from backend.models.schemas import ProcessThreadRequest, ProcessThreadResult
from backend.orchestrator.pipeline import ProcessingPipeline
from backend.threads.progress import (
    extract_post_id,
    get_analysis_progress,
    set_analysis_progress,
    start_analysis_progress,
)


router = APIRouter(prefix="/api")


def _tool_gateway() -> MCPToolGateway:
    settings = get_settings()
    if settings.mcp_in_process:
        from mcp_servers.internal_tools import mcp

        return MCPToolGateway(mcp)
    return MCPToolGateway(settings.mcp_server_url)


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    models = get_model_registry()
    return {
        "status": "ok" if not settings.validate_model_paths() else "configuration_error",
        "missing_paths": settings.validate_model_paths(),
        "device_requested": settings.model_device,
        "bert_loaded": models.bert.loaded,
        "summary_loaded": models.summarizer.loaded,
        "mcp_server_url": settings.mcp_server_url,
        "mcp_in_process": settings.mcp_in_process,
        "threads_tree_data_dir": str(settings.threads_tree_data_dir),
    }


@router.post("/process-thread", response_model=ProcessThreadResult)
async def process_thread(request: ProcessThreadRequest) -> dict:
    settings = get_settings()
    post_id = extract_post_id(request.url)
    if post_id:
        start_analysis_progress(post_id, request.max_nodes)
    try:
        pipeline = ProcessingPipeline(
            get_model_registry(),
            _tool_gateway(),
            settings.threads_tree_data_dir,
            rag_enabled=settings.rag_enabled,
        )
        return await pipeline.run_thread(
            url=request.url,
            max_nodes=request.max_nodes,
            force_refresh=request.force_refresh,
            max_new_tokens=request.max_new_tokens,
            classifier_batch_size=request.classifier_batch_size,
            judgment_limit=request.judgment_limit,
        )
    except ValueError as exc:
        if post_id:
            set_analysis_progress(post_id, state="failed", message=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        if post_id:
            set_analysis_progress(post_id, state="failed", message=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        if post_id:
            set_analysis_progress(post_id, state="failed", message=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/collection-progress/{post_id}")
async def collection_progress(post_id: str) -> dict:
    progress = get_analysis_progress(post_id)
    if progress is None:
        return {
            "post_id": post_id,
            "state": "idle",
            "message": "尚未開始處理",
            "current": 0,
            "total": 0,
            "elapsed_seconds": 0,
            "idle_seconds": 0,
        }
    return progress


@router.get("/tools")
async def list_tools() -> list[dict]:
    try:
        return await _tool_gateway().list_tools()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
