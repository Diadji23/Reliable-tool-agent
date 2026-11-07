"""MCP server exposing the shop tools.

The server is built for one authenticated customer: the application starts it with
`--customer-id`, out of the model's reach. Tool schemas therefore never contain a
customer id, and the model can only choose the arguments listed in them.

Run over stdio (the client launches this as a subprocess):
    uv run python -m reliable_tool_agent.mcp_server --customer-id C0001
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from reliable_tool_agent import tools
from reliable_tool_agent.db import DEFAULT_DB_PATH
from reliable_tool_agent.tools import (
    NotFoundError,
    Order,
    OrderStatus,
    OrderSummary,
    PolicyRule,
    ReturnEligibility,
    Session,
)


def build_server(session: Session) -> MCPServer:
    """Create an MCP server whose tools all act for `session.customer_id`."""
    server = MCPServer(
        name="brindle-support",
        instructions="Tools for Brindle Home customer support, scoped to one customer.",
    )

    @server.tool()
    def get_order(order_id: str) -> Order:
        """Get one of the current customer's orders: status, dates, items and amounts.

        Amounts are integer cents (4900 = 49.00 EUR). Fails with 'No order ... found'
        if the order does not exist for this customer.
        """
        try:
            return tools.get_order(session, order_id)
        except NotFoundError as exc:
            # An expected failure: ToolError sends its message to the model, so it can
            # tell the customer or fix the order number. Any other exception is a crash
            # and the SDK keeps its text on the server.
            raise ToolError(str(exc)) from exc

    @server.tool()
    def list_my_orders(status: OrderStatus | None = None) -> list[OrderSummary]:
        """List the current customer's orders, newest first.

        Use it when the customer does not give an order number. Optionally filter by
        status: processing, shipped, delivered or cancelled. Amounts are integer cents.
        """
        return tools.list_my_orders(session, status)

    @server.tool()
    def search_policies(query: str) -> list[PolicyRule]:
        """Search the shop policies (returns, refunds, cancellations, privacy, scope).

        Returns up to 3 numbered rules by keyword match, or an empty list if nothing
        matches. Use plain words from the topic, e.g. "return final sale damaged".
        Always check the policy before telling a customer what is allowed.
        """
        return tools.search_policies(session, query)

    @server.tool()
    def check_return_eligibility(order_id: str) -> ReturnEligibility:
        """Check whether the items of one of the customer's orders can be returned today.

        Applies the returns policy (30-day window, delivered orders only, final sale) and
        gives, per item, `returnable`, `returnable_if_defective_or_wrong` and the reason,
        with the last day a return is accepted. Use it for any question about returning
        an item instead of computing dates yourself.
        """
        try:
            return tools.check_return_eligibility(session, order_id)
        except NotFoundError as exc:
            raise ToolError(str(exc)) from exc

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Brindle Home support tools over MCP (stdio).")
    parser.add_argument("--customer-id", required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    build_server(Session(args.customer_id, args.db)).run()


if __name__ == "__main__":
    main()
