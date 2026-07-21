from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..budget import Budget
from ..graph import run_enrichment
from ..repository import (
    configure_engine,
    get_record,
    get_trace,
    list_records,
    save_run,
)
from ..tools.llm import DEFAULT_GROQ_CONFIG, GroqLLMClient, LLMClient
from ..validation import MissingNameError, validate_request
from .schemas import (
    CreateRecordRequest,
    CreateRecordResponse,
    HealthOut,
    RecordDetailOut,
    RecordSummaryOut,
    ResolvedFieldOut,
    TraceEventOut,
)

_cached_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Lazy-init the Groq LLM client so the env-var check only runs once.

    Without caching, every request would construct a new OpenAI client
    handle, which is wasteful and would re-enforce the env-var guard on
    every call. The cached client lives for the process lifetime.
    """
    global _cached_llm_client
    if _cached_llm_client is None:
        _cached_llm_client = GroqLLMClient(DEFAULT_GROQ_CONFIG)
    return _cached_llm_client


def get_wiki_fetcher() -> Optional[Callable]:
    return None


def get_budget_from_request(req: CreateRecordRequest) -> Budget:
    return Budget(max_cost_usd=req.max_cost_usd, max_latency_ms=req.max_latency_ms)


# Deliberately NOT adding CORSMiddleware: the frontend is served
# same-origin by this same FastAPI app, so cross-origin config is
# unnecessary. Permissive CORS would be a security regression.
app = FastAPI(title="Enrichment Router", version="0.1.0")

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.on_event("startup")
def _startup_init_db() -> None:
    configure_engine()


@app.get("/api/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(status="ok")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the frontend at the root."""
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.post("/api/records", response_model=CreateRecordResponse)
def create_record(
    req: CreateRecordRequest,
    llm_client: LLMClient = Depends(get_llm_client),
    wiki_fetcher: Optional[Callable] = Depends(get_wiki_fetcher),
) -> CreateRecordResponse:
    raw = req.model_dump()
    try:
        enrichment_request = validate_request(raw)
    except MissingNameError:
        raise HTTPException(
            status_code=422,
            detail="name is required and must be a non-empty string",
        )

    budget = get_budget_from_request(req)
    result, trace_events = run_enrichment(
        request=enrichment_request,
        budget=budget,
        llm_client=llm_client,
        wiki_fetcher=wiki_fetcher,
    )

    run_id, record_id = save_run(enrichment_request, result, trace_events)

    return CreateRecordResponse(
        run_id=run_id,
        record_id=record_id,
        name=enrichment_request.name,
        domain=enrichment_request.domain,
        status=result.status,
        total_cost_usd=result.total_cost_usd,
        total_latency_ms=result.total_latency_ms,
        resolved=[
            ResolvedFieldOut(
                name=rf.name,
                value=rf.value,
                tier=rf.tier,
                confidence=rf.confidence,
                caller_supplied=rf.caller_supplied,
            )
            for rf in result.resolved.values()
        ],
        unresolved_fields=list(result.unresolved_fields),
    )


@app.get("/api/records", response_model=list[RecordSummaryOut])
def list_records_route() -> list[RecordSummaryOut]:
    return [RecordSummaryOut(**rs.__dict__) for rs in list_records()]


@app.get("/api/records/{record_id}", response_model=RecordDetailOut)
def get_record_route(record_id: int) -> RecordDetailOut:
    detail = get_record(record_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="record not found")
    return RecordDetailOut(
        record_id=detail.record_id,
        name=detail.name,
        domain=detail.domain,
        run_id=detail.run_id,
        status=detail.status,
        total_cost_usd=detail.total_cost_usd,
        total_latency_ms=detail.total_latency_ms,
        resolved=detail.resolved,
        created_at=detail.created_at,
    )


@app.get("/api/records/{record_id}/trace", response_model=list[TraceEventOut])
def get_trace_route(record_id: int) -> list[TraceEventOut]:
    detail = get_record(record_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="record not found")
    if detail.run_id == 0:
        return []
    return [
        TraceEventOut(
            node=ev.node,
            detail=ev.detail,
            timestamp=ev.timestamp,
            id=ev.id,
        )
        for ev in get_trace(detail.run_id)
    ]
