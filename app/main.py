from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from app.config import get_settings
from app.graph.workflow import build_workflow
from app.schemas.api import QueryRequest, QueryResponse

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.workflow = build_workflow(checkpointer=InMemorySaver())
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
async def query(request: QueryRequest) -> QueryResponse:
    graph_input = request.model_dump(exclude={"session_id"})
    graph_input["trace"] = []
    result = await app.state.workflow.ainvoke(
        graph_input,
        config={"configurable": {"thread_id": request.session_id}},
    )
    return QueryResponse(
        session_id=request.session_id,
        answer=result["answer"],
        intent=result.get("intent"),
        route=result["route"],
        faq_id=result.get("faq_id"),
        classification_source=result.get("classification_source", "rules"),
        compose_source=result.get("compose_source", "fixed"),
        data=result.get("data", {}),
        screen_action=result.get("screen_action"),
        validation_errors=result.get("validation_errors", []),
        trace=result.get("trace", []),
    )
