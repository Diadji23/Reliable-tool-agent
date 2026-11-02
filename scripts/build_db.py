"""Generate the fictional shop database (data/shop.db) deterministically."""

from __future__ import annotations

import argparse
from pathlib import Path

from reliable_tool_agent.db import DEFAULT_DB_PATH, connect_readonly
from reliable_tool_agent.world import DEFAULT_SEED, build_database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    path = build_database(args.out, seed=args.seed)
    with connect_readonly(path) as conn:
        for table in ("customers", "products", "orders", "order_items", "refunds"):
            (n,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            print(f"{table:12s} {n:5d} rows")
        rows = conn.execute("SELECT status, COUNT(*) FROM orders GROUP BY status").fetchall()
        print("order status:", ", ".join(f"{s}={n}" for s, n in rows))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
