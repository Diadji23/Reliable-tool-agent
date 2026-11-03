"""Draw the system architecture (docs/architecture.png).

Built components are drawn in colour with solid borders, planned ones in grey with dashed
borders. Update BUILT as the project progresses, then re-run:
    uv run python scripts/make_architecture_diagram.py
(needs matplotlib, from the `notebook` dependency group)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"

BUILT = {"mcp", "get_order", "list_my_orders", "search_policies", "shop.db", "policies"}

COLORS = {
    "app": "#4C72B0",
    "agent": "#DD8452",
    "guard": "#C44E52",
    "mcp": "#55A868",
    "data": "#8172B3",
    "ops": "#937860",
    "planned": "#9A9A9A",
}


def box(ax, key, x, y, w, h, text, color, fontsize=9.5, bold=False, fill=0.18):
    built = key in BUILT
    c = color if built else COLORS["planned"]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=to_rgba(c, fill if built else 0.07),
            ec=c,
            lw=1.8 if built else 1.4,
            ls="-" if built else (0, (4, 3)),
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color="black" if built else "#555555",
    )


def frame(ax, key, x, y, w, h, title, color):
    """A container: border + title at the top."""
    built = key in BUILT
    c = color if built else COLORS["planned"]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            fc=to_rgba(c, 0.06),
            ec=c,
            lw=2.0 if built else 1.5,
            ls="-" if built else (0, (5, 3)),
        )
    )
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=11,
            fontweight="bold", color="black" if built else "#555555")  # fmt: skip


def arrow(ax, x1, y1, x2, y2, color="black", dashed=False, both=False, lw=1.6):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="<|-|>" if both else "-|>",
            color=color,
            lw=lw,
            ls=(0, (4, 3)) if dashed else "-",
            shrinkA=2,
            shrinkB=2,
        ),
    )


def label(ax, x, y, text, color="black", fontsize=8.5, **kw):
    ax.text(x, y, text, color=color, fontsize=fontsize, ha="center", va="center", **kw)


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(8, 8.65, "reliable-tool-agent — architecture", ha="center", fontsize=15,
            fontweight="bold")  # fmt: skip

    # --- Left: customer and API ---------------------------------------------------------
    box(ax, "customer", 0.2, 4.6, 1.6, 1.0, "Customer", COLORS["app"], bold=True)
    box(ax, "api", 0.2, 2.8, 1.6, 1.2, "FastAPI\n+ auth\n(session)", COLORS["app"])
    arrow(ax, 1.0, 4.6, 1.0, 4.0, COLORS["planned"], dashed=True, both=True)

    # --- Agent ------------------------------------------------------------------------
    frame(ax, "agent", 2.5, 1.95, 6.3, 5.35, "Agent (LangGraph)", COLORS["agent"])
    box(ax, "guard_in", 2.8, 5.85, 5.7, 0.75,
        "Input guardrails — prompt injection, out-of-scope refusal", COLORS["guard"])  # fmt: skip
    box(ax, "loop", 2.8, 4.1, 3.5, 1.5,
        "LLM  ⇄  tool calls\nmemory + context\nmanagement", COLORS["agent"])  # fmt: skip
    box(ax, "llm", 6.55, 4.1, 1.95, 1.5, "Local LLM\n(llama.cpp)\n/ free API", COLORS["agent"],
        fontsize=8.5)  # fmt: skip
    arrow(ax, 6.3, 4.85, 6.55, 4.85, COLORS["planned"], dashed=True, both=True)
    box(ax, "guard_out", 2.8, 3.15, 5.7, 0.7,
        "Output validation — structured answer (Pydantic)", COLORS["guard"])  # fmt: skip
    hitl = "Human approval before refund / cancel  (interrupt → operator)"
    box(ax, "hitl", 2.8, 2.2, 5.7, 0.7, hitl, COLORS["guard"])
    arrow(ax, 1.8, 3.4, 2.5, 3.9, COLORS["planned"], dashed=True, both=True)

    # --- MCP server ---------------------------------------------------------------------
    frame(ax, "mcp", 9.6, 1.95, 3.4, 5.35, "MCP server", COLORS["mcp"])
    tools = [
        ("get_order", "get_order"),
        ("list_my_orders", "list_my_orders"),
        ("search_policies", "search_policies"),
        ("refund", "refund_order  ⚠"),
        ("cancel", "cancel_order  ⚠"),
    ]
    for i, (key, text) in enumerate(tools):
        box(ax, key, 9.9, 6.05 - i * 0.82, 2.8, 0.6, text, COLORS["mcp"], fontsize=9.5)
    label(ax, 11.3, 2.25, "⚠ = sensitive, needs approval", "#555555", fontsize=8)
    arrow(ax, 8.8, 4.85, 9.6, 4.85, COLORS["planned"], dashed=True, both=True)
    label(ax, 9.2, 5.15, "MCP\n(stdio)", "#555555", fontsize=8)

    # Session: the customer id goes from the application to the server, around the model.
    ax.plot([1.0, 1.0, 11.3, 11.3], [5.6, 7.75, 7.75, 7.3], color=COLORS["app"], lw=1.6,
            ls=(0, (2, 2)))  # fmt: skip
    arrow(ax, 11.3, 7.5, 11.3, 7.3, COLORS["app"])
    label(ax, 6.2, 7.95,
          "customer id from the authenticated session — never from the model",
          COLORS["app"], fontsize=9, fontweight="bold")  # fmt: skip

    # --- Data ---------------------------------------------------------------------------
    box(ax, "shop.db", 13.6, 4.6, 2.2, 1.6,
        "shop.db\n(SQLite, synthetic)\nread-only: 2 locks", COLORS["data"])  # fmt: skip
    box(ax, "policies", 13.6, 2.6, 1.7, 1.3, "Shop policies\n(5 Markdown\nfiles)",
        COLORS["data"])  # fmt: skip
    arrow(ax, 13.0, 5.4, 13.6, 5.4, COLORS["data"])
    arrow(ax, 13.0, 3.25, 13.6, 3.25, COLORS["data"])

    # --- Bottom: observability and evaluation --------------------------------------------
    box(ax, "obs", 0.2, 0.2, 4.6, 1.3,
        "Observability (Langfuse)\ntraces · latency · tokens · cost per request",
        COLORS["ops"])  # fmt: skip
    box(ax, "eval", 5.1, 0.2, 10.7, 1.3,
        "Evaluation — reference tasks · success = expected final DB state · attack set\n"
        "LLM-as-judge (calibrated) · cost & latency · runs tracked in MLflow · CI (pytest)",
        COLORS["ops"])  # fmt: skip
    # The agent emits traces; the evaluation reads the final state of the database.
    arrow(ax, 3.4, 1.95, 3.4, 1.5, COLORS["planned"], dashed=True)
    label(ax, 2.75, 1.73, "traces", "#555555", fontsize=8)
    arrow(ax, 15.55, 4.6, 15.55, 1.5, COLORS["planned"], dashed=True)
    label(ax, 15.55, 4.25, "final\nstate", "#555555", fontsize=8, rotation=0,
          bbox=dict(fc="white", ec="none", pad=1))  # fmt: skip

    legend = [
        Patch(fc=to_rgba(COLORS["mcp"], 0.18), ec=COLORS["mcp"], lw=1.8, label="built"),
        Patch(
            fc=to_rgba(COLORS["planned"], 0.07),
            ec=COLORS["planned"],
            lw=1.4,
            ls="--",
            label="planned",
        ),  # fmt: skip
        Line2D([], [], color=COLORS["app"], ls=(0, (2, 2)), label="trusted identity path"),
    ]
    ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False,
              fontsize=9)  # fmt: skip

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
