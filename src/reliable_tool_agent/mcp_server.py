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
from mcp_types import ToolAnnotations

from reliable_tool_agent import tools
from reliable_tool_agent.db import DEFAULT_DB_PATH
from reliable_tool_agent.tools import (
    NotFoundError,
    Order,
    OrderStatus,
    OrderSummary,
    PolicyRule,
    Refund,
    RefundCondition,
    RefundQuote,
    RefundRefusedError,
    ReturnEligibility,
    Session,
)

# Hints for MCP clients. They describe the tool; they enforce nothing. The approval step
# in the agent decides on its own which tools need a human, it does not trust these.
READ_ONLY = ToolAnnotations(read_only_hint=True)
ISSUES_MONEY = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False)


def build_server(session: Session) -> MCPServer:
    """Create an MCP server whose tools all act for `session.customer_id`."""
    server = MCPServer(
        name="brindle-support",
        instructions="Tools for Brindle Home customer support, scoped to one customer.",
    )

    @server.tool(annotations=READ_ONLY)
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

    @server.tool(annotations=READ_ONLY)
    def list_my_orders(status: OrderStatus | None = None) -> list[OrderSummary]:
        """List the current customer's orders, newest first.

        Use it when the customer does not give an order number. Optionally filter by
        status: processing, shipped, delivered or cancelled. Amounts are integer cents.
        """
        return tools.list_my_orders(session, status)

    @server.tool(annotations=READ_ONLY)
    def search_policies(query: str) -> list[PolicyRule]:
        """Search the shop policies (returns, refunds, cancellations, privacy, scope).

        Returns up to 3 numbered rules by keyword match, or an empty list if nothing
        matches. Use plain words from the topic, e.g. "return final sale damaged".
        Always check the policy before telling a customer what is allowed.
        """
        return tools.search_policies(session, query)

    @server.tool(annotations=READ_ONLY)
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

    @server.tool(annotations=READ_ONLY)
    def quote_refund(
        order_id: str, product_id: str, quantity: int, condition: RefundCondition
    ) -> RefundQuote:
        """Compute the refund for returned units of one product, without issuing it.

        `condition` is changed_mind or defective_or_wrong (defective, damaged or wrong item).
        Gives the item amount, any shipping refunded and the total, in integer cents.
        Fails with the reason if a policy rule forbids the refund.
        """
        try:
            return tools.quote_refund(session, order_id, product_id, quantity, condition)
        except (NotFoundError, RefundRefusedError) as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(annotations=ISSUES_MONEY)
    def refund_order(
        order_id: str,
        product_id: str,
        quantity: int,
        condition: RefundCondition,
        approved_amount_cents: int,
    ) -> Refund:
        """Issue a refund for returned units of one product of an order.

        `condition` is changed_mind or defective_or_wrong (defective, damaged or wrong item).
        The amount is computed from the policy, never chosen by you. Only call it once the
        customer has asked for a refund of these units. Fails with the reason if a policy
        rule forbids the refund; nothing is written then.

        `approved_amount_cents` is set by the application from the human approval: the
        refund is issued only if the policy still gives exactly that amount.
        """
        try:
            return tools.refund_order(
                session,
                order_id,
                product_id,
                quantity,
                condition,
                approved_amount_cents=approved_amount_cents,
            )
        except (NotFoundError, RefundRefusedError) as exc:
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
