"""The agent behind an HTTP API (FastAPI): customers talk to it, a reviewer decides refunds.

    POST /conversations                   customer  start a conversation
    POST /conversations/{id}/messages     customer  send a message, get the agent's answer
    GET  /health                          anyone    is the server up
    GET  /approvals                       reviewer  refunds waiting for a decision
    POST /approvals/{id}                  reviewer  decide them; the agent finishes its turn

Who is calling comes from the bearer token in the Authorization header, never from the
request body: a customer id written in the body could be anyone's. The token says the
role (customer or reviewer) and, for a customer, which one. The MCP server of a request
is opened for that customer only, as in the chat.

Run locally (after llama-server):

    BRINDLE_TOKENS="demo-c21=customer:C0021,demo-rev=reviewer" \\
        uv run uvicorn reliable_tool_agent.api:app_from_env --factory

Then open http://127.0.0.1:8000/docs for the interactive documentation.

Known limits, kept on purpose for a demo: a request waits for the whole agent turn (minutes
on a CPU; a job queue or streaming would fix it), and conversations live in memory (lost
on restart; a database checkpointer would fix it).
"""

from __future__ import annotations

import asyncio
import hmac
import os
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langgraph.checkpoint.memory import InMemorySaver
from mcp import Client
from pydantic import BaseModel, Field

from reliable_tool_agent.agent_graph import Decision, GraphAgent
from reliable_tool_agent.agent_loop import AgentResult, ChatModel, McpToolbox
from reliable_tool_agent.db import DEFAULT_DB_PATH
from reliable_tool_agent.llama_server import ModelUnavailableError
from reliable_tool_agent.mcp_server import build_server
from reliable_tool_agent.prompts import system_prompt
from reliable_tool_agent.tools import Session

# --- Who is calling --------------------------------------------------------------------


@dataclass(frozen=True)
class Caller:
    role: Literal["customer", "reviewer"]
    customer_id: str | None = None  # set for a customer only


def parse_tokens(spec: str) -> dict[str, Caller]:
    """Read "token=customer:C0021,token2=reviewer" (the BRINDLE_TOKENS variable)."""
    tokens = {}
    for entry in filter(None, (e.strip() for e in spec.split(","))):
        token, _, who = entry.partition("=")
        role, _, customer = who.partition(":")
        if (
            not token
            or role not in ("customer", "reviewer")
            or (role == "customer") != bool(customer)
        ):
            raise ValueError(
                f"bad token entry {entry!r}: use token=customer:<id> or token=reviewer"
            )
        tokens[token] = Caller(role, customer or None)
    return tokens


bearer = HTTPBearer(auto_error=False)


def caller(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Caller:
    """The caller named by the bearer token, or 401."""
    if credentials is not None:
        for token, who in request.app.state.service.tokens.items():
            # Compare every token in constant time: a plain == stops at the first wrong
            # character, and the time it takes tells an attacker how much was right.
            if hmac.compare_digest(credentials.credentials.encode(), token.encode()):
                return who
    raise HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "missing or unknown token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def customer(who: Annotated[Caller, Depends(caller)]) -> str:
    """The id of the calling customer; 403 for anyone else."""
    if who.role != "customer":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "customers only")
    return who.customer_id


def reviewer(who: Annotated[Caller, Depends(caller)]) -> Caller:
    """The calling reviewer; 403 for anyone else."""
    if who.role != "reviewer":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "reviewers only")
    return who


# --- The conversations -----------------------------------------------------------------


@dataclass
class Service:
    """What the API keeps between requests. HTTP itself remembers nothing.

    The conversation itself is in the checkpointer, under its id; here are only who owns
    each conversation, the refunds waiting for a reviewer, and one lock per conversation
    (two requests on the same thread at once would interleave their writes to it).
    """

    model: ChatModel
    tokens: dict[str, Caller]
    db_path: Path
    max_steps: int = 8
    checkpointer: InMemorySaver = field(default_factory=InMemorySaver)
    owners: dict[str, str] = field(default_factory=dict)  # conversation id -> customer id
    pending: dict[str, list[dict]] = field(default_factory=dict)  # conversation id -> requests
    locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    @asynccontextmanager
    async def agent(self, customer_id: str) -> AsyncIterator[GraphAgent]:
        """An agent whose MCP server serves this customer only, for one request.

        Opened and closed within the request: the conversation survives in the shared
        checkpointer, and nothing outlives the request that opened it.
        """
        async with Client(build_server(Session(customer_id, self.db_path))) as client:
            yield GraphAgent(
                self.model, McpToolbox(client), system_prompt(), self.checkpointer, self.max_steps
            )

    def owned(self, conversation_id: str, customer_id: str) -> None:
        """404 unless this conversation is this customer's (another's looks missing)."""
        if self.owners.get(conversation_id) != customer_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such conversation")

    def record(self, conversation_id: str, result: AgentResult) -> TurnOut:
        """Remember what the turn left waiting, and say how it ended."""
        if result.stop_reason == "awaiting_approval":
            self.pending[conversation_id] = result.pending
            return TurnOut(conversation_id=conversation_id, status="awaiting_approval")
        self.pending.pop(conversation_id, None)
        if result.stop_reason == "max_steps":
            return TurnOut(conversation_id=conversation_id, status="no_answer")
        return TurnOut(conversation_id=conversation_id, status="answered", answer=result.answer)


def service(request: Request) -> Service:
    return request.app.state.service


# What a route asks for, by naming one of these in its parameters (FastAPI "dependencies").
Customer = Annotated[str, Depends(customer)]
Reviewer = Annotated[Caller, Depends(reviewer)]
Svc = Annotated[Service, Depends(service)]


# --- What goes in and out --------------------------------------------------------------


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class ConversationOut(BaseModel):
    conversation_id: str


class TurnOut(BaseModel):
    conversation_id: str
    status: Literal["answered", "awaiting_approval", "no_answer"]
    answer: str | None = None


class ApprovalRequestOut(BaseModel):
    tool_call_id: str
    tool: str
    arguments: dict
    preview: dict  # the quote computed by the code: what an approval approves
    customer_said: list[str]  # the customer's own messages, copied by the code


class PendingOut(BaseModel):
    conversation_id: str
    customer_id: str
    requests: list[ApprovalRequestOut]


class DecisionsIn(BaseModel):
    decisions: dict[str, Decision]  # tool_call_id -> decision; a missing one is a rejection


@contextmanager
def model_unavailable_is_503() -> Iterator[None]:
    """Model out of quota or down: 503 with the wait the model server asked for.

    Not a 500 (nothing is broken here) and not a hang. The error arrives wrapped in an
    exception group by the MCP client's task group, hence the search inside it."""
    try:
        yield
    except* ModelUnavailableError as group:
        error = group.exceptions[0]
        while isinstance(error, BaseExceptionGroup):
            error = error.exceptions[0]
        headers = {"Retry-After": str(int(error.retry_after))} if error.retry_after else None
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "the model is over its quota or unavailable, try again later",
            headers=headers,
        ) from None


# --- The app ---------------------------------------------------------------------------


def create_app(
    model: ChatModel, tokens: dict[str, Caller], db_path: Path, max_steps: int = 8
) -> FastAPI:
    app = FastAPI(title="Brindle Home support agent")
    app.state.service = Service(model, tokens, db_path, max_steps)

    @app.get("/", include_in_schema=False)
    async def home() -> RedirectResponse:
        """A visitor opening the bare URL lands on the interactive API page."""
        return RedirectResponse("/docs")

    @app.get("/health")
    async def health() -> dict:
        """Is the server up? No token: a container platform asks it to know when to restart.
        It does not call the model: a slow model is not a dead server."""
        return {"status": "ok"}

    @app.post("/conversations", status_code=status.HTTP_201_CREATED)
    async def start_conversation(customer_id: Customer, svc: Svc) -> ConversationOut:
        conversation_id = f"c-{uuid.uuid4().hex[:12]}"  # chosen by the server, not guessable
        svc.owners[conversation_id] = customer_id
        svc.locks[conversation_id] = asyncio.Lock()
        return ConversationOut(conversation_id=conversation_id)

    @app.post("/conversations/{conversation_id}/messages")
    async def send_message(
        conversation_id: str, message: MessageIn, customer_id: Customer, svc: Svc
    ) -> TurnOut:
        svc.owned(conversation_id, customer_id)
        async with svc.locks[conversation_id]:
            if conversation_id in svc.pending:
                raise HTTPException(
                    status.HTTP_409_CONFLICT, "a refund in this conversation awaits a reviewer"
                )
            with model_unavailable_is_503():
                async with svc.agent(customer_id) as agent:
                    result = await agent.ask(conversation_id, message.text)
            return svc.record(conversation_id, result)

    @app.get("/approvals")
    async def list_approvals(_: Reviewer, svc: Svc) -> list[PendingOut]:
        return [
            PendingOut(
                conversation_id=conversation_id,
                customer_id=svc.owners[conversation_id],
                requests=[ApprovalRequestOut.model_validate(r) for r in requests],
            )
            for conversation_id, requests in svc.pending.items()
        ]

    @app.post("/approvals/{conversation_id}")
    async def decide(conversation_id: str, body: DecisionsIn, _: Reviewer, svc: Svc) -> TurnOut:
        """The reviewer's decisions on a conversation's pending refunds; the agent then
        finishes the paused turn, with the tools of the customer who owns the conversation."""
        if conversation_id not in svc.owners:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such conversation")
        async with svc.locks[conversation_id]:
            # Checked under the lock: otherwise two reviewers could both see the refund
            # pending, and both resume the conversation.
            if conversation_id not in svc.pending:
                raise HTTPException(status.HTTP_409_CONFLICT, "nothing awaits a decision here")
            with model_unavailable_is_503():
                async with svc.agent(svc.owners[conversation_id]) as agent:
                    result = await agent.resume(conversation_id, body.decisions)
            return svc.record(conversation_id, result)

    return app


def app_from_env() -> FastAPI:
    """The app configured by environment variables (used by uvicorn and Docker).

    BRINDLE_TOKENS   who may call: "token=customer:C0021,token2=reviewer" (required)
    MODEL_BASE_URL   OpenAI-compatible server (default: local llama-server)
    MODEL_NAME       model name sent to that server (default: "local")
    MODEL_API_KEY    key for a hosted server; unset for llama-server
    MAX_STEPS        model calls per turn (default 8)

    Refunds are written to a private copy of the shop database, made at start-up: the
    committed data/shop.db never changes.
    """
    from reliable_tool_agent.llama_server import LlamaServerModel

    api_key = os.environ.get("MODEL_API_KEY")
    model = LlamaServerModel(
        base_url=os.environ.get("MODEL_BASE_URL", "http://127.0.0.1:8080/v1"),
        model=os.environ.get("MODEL_NAME", "local"),
        thinking=None if api_key else False,  # the thinking switch is llama-server's only
        api_key=api_key,
        max_retries=5 if api_key else 0,
        max_wait_s=60,  # a customer waits a minute at most for a quota, then gets a 503
    )
    workdir = Path(tempfile.mkdtemp(prefix="brindle-api-"))
    db_path = Path(shutil.copyfile(DEFAULT_DB_PATH, workdir / "shop.db"))
    return create_app(
        model,
        parse_tokens(os.environ["BRINDLE_TOKENS"]),
        db_path,
        int(os.environ.get("MAX_STEPS", "8")),
    )
