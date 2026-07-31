from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.graph.workflow import build_workflow
from app.schemas.api import QueryRequest, QueryResponse

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.workflow = build_workflow()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment}


@app.post("/v1/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = app.state.workflow.invoke(request.model_dump())
    return QueryResponse(
        answer=result["answer"],
        route=result["route"],
        data=result.get("data", {}),
        screen_action=result.get("screen_action"),
        validation_errors=result.get("validation_errors", []),
        trace=result.get("trace", []),
    )
