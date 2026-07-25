import json
import time
from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from .config import settings
from .logger import log_event
from .tools import TOOLS

SYSTEM_PROMPT = """You are a meticulous data analyst.

The user's message is a data-analysis question. It will tell you EXACTLY
what JSON shape to reply with, e.g.:
{{"answer": {{"state": "<state name>"}}, "log_url": "<url>"}}

Rules:
- Only answer the LAST user message; earlier messages (if any) are just context.
- If the question references a public dataset (MOSPI or similar) and you
  need exact figures, use the fetch_csv_as_dataframe and run_pandas_query
  tools rather than guessing.
- When you are done, reply with ONLY the JSON object requested by the
  question - no markdown fences, no extra commentary, nothing before or
  after it. Leave the log_url value as the placeholder "LOG_URL_PLACEHOLDER";
  it will be replaced automatically before the reply is sent.
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

_SHARED_LLM: Optional[ChatOpenAI] = None
_SHARED_AGENT: Optional[AgentExecutor] = None


def _build_llm() -> ChatOpenAI:
    kwargs = dict(
        model=settings.MODEL_NAME,
        temperature=settings.TEMPERATURE,
        api_key=settings.LLM_API_KEY,
    )
    if settings.LLM_BASE_URL:
        kwargs["base_url"] = settings.LLM_BASE_URL
    return ChatOpenAI(**kwargs)


def get_shared_llm() -> ChatOpenAI:
    global _SHARED_LLM
    if _SHARED_LLM is None:
        _SHARED_LLM = _build_llm()
    return _SHARED_LLM


def reset_shared_llm() -> None:
    global _SHARED_LLM
    _SHARED_LLM = None


def _build_agent_executor() -> AgentExecutor:
    llm = get_shared_llm()
    _agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=_agent,
        tools=TOOLS,
        return_intermediate_steps=True, 
        max_iterations=6,
    )


def get_shared_agent() -> AgentExecutor:
    global _SHARED_AGENT
    if _SHARED_AGENT is None:
        _SHARED_AGENT = _build_agent_executor() 
    return _SHARED_AGENT


def reset_shared_agent() -> None:
    global _SHARED_AGENT
    _SHARED_AGENT = None


def _clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[-1]
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text.strip()


async def answer_question(history: list[str]) -> str:
    question = history[-1]
    chat_history = [("human", turn) for turn in history[:-1]]

    log_event({"stage": "received", "question": question, "history": history, "ts": time.time()})

    agent = get_shared_agent()
    result = await agent.ainvoke({
        "input": question,
        "chat_history": chat_history,
    })

    for action, observation in result.get("intermediate_steps", []):
        log_event({
            "stage": "tool_call",
            "tool": action.tool,
            "tool_input": action.tool_input,
            "observation": str(observation)[:2000],
            "ts": time.time(),
        })

    raw_output = _clean_json_text(result.get("output", ""))
    log_event({"stage": "llm_raw_reply", "raw": raw_output, "ts": time.time()})

    try:
        parsed = json.loads(raw_output)
        if not isinstance(parsed, dict):
            parsed = {"answer": parsed}
    except json.JSONDecodeError:
        parsed = {"answer": raw_output}

    parsed["log_url"] = settings.LOG_URL

    final_reply = json.dumps(parsed, ensure_ascii=False)
    log_event({"stage": "final_reply", "reply": final_reply, "ts": time.time()})
    return final_reply