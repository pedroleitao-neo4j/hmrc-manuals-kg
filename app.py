"""FastAPI service for the HMRC manuals tax agent.

Run (from anywhere — .env is loaded from the project root):
    .venv/bin/uvicorn app:app --port 8000
Then open http://localhost:8000
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent import config
from agent.graph import run_agent

STATIC = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    question: str


app = FastAPI(title="HMRC Manuals Tax Agent")


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict:
    """Non-streaming: run the agent, return the final structured answer."""
    events = list(run_agent(req.question))
    answer = next((e for e in events if e["type"] == "answer"), None)
    if answer is None:
        return {"error": "agent produced no answer"}
    return answer


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE stream: tool_call events as the agent works, then the answer."""

    def gen():
        try:
            for event in run_agent(req.question):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:                      # noqa: BLE001
            yield (f"data: {json.dumps({'type': 'error', 'message': str(e)})}"
                   "\n\n")
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/viz/{viz_id}")
async def viz(viz_id: str) -> FileResponse:
    """Self-contained neo4j_viz HTML of the subgraph that answered a question."""
    # viz_id is generated server-side (uuid hex); reject anything else.
    if not viz_id.isalnum() or len(viz_id) != 12:
        raise HTTPException(status_code=404, detail="not found")
    path = STATIC / "viz" / f"{viz_id}.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="text/html")
