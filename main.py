import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from src.agent import answer_question
from src.logger import log_event
from src.config import settings

app = FastAPI()

chat_history: dict[int, list[str]] = {}


@app.post("/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()
    log_event({"stage": "telegram_update", "update": update})

    message = update.get("message")
    if not message or "text" not in message:
        return {"ok": True}

    chat_id = message["chat"]["id"]
    text = message["text"]

    history = chat_history.setdefault(chat_id, [])
    history.append(text)
    if len(history) > settings.MAX_HISTORY_PER_CHAT:
        history[:] = history[-settings.MAX_HISTORY_PER_CHAT:]

    reply_json = await answer_question(history)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{settings.TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": reply_json},
        )
        log_event({"stage": "telegram_send_response", "status_code": resp.status_code})

    return {"ok": True}


@app.get("/run.jsonl")
def get_log():
    """
    Serves the JSONL log publicly so `log_url` in your replies is wget-able.
    Only use this if your host's filesystem persists between requests -
    otherwise prefer committing logs to GitHub or a cloud bucket instead.
    """
    if not os.path.exists(settings.LOG_PATH):
        return JSONResponse({"error": "no logs yet"}, status_code=404)
    return FileResponse(settings.LOG_PATH, media_type="application/x-ndjson")


@app.get("/")
def health_check():
    return {"status": "alive"}
