"""Deterministic P&L math for exit levels (do not rely on the LLM for arithmetic)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PnlAtPrice:
    exit_price: float
    pnl_per_unit: float
    pnl_pct: float
    pnl_total: float | None


def pnl_at_price(entry_price: float, exit_price: float, quantity: float | None = 1.0) -> PnlAtPrice:
    """Return gain/loss vs entry at a given exit price."""
    if entry_price <= 0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    diff = exit_price - entry_price
    pct = 100.0 * diff / entry_price
    total = diff * quantity if quantity is not None else None
    return PnlAtPrice(
        exit_price=exit_price,
        pnl_per_unit=diff,
        pnl_pct=pct,
        pnl_total=total,
    )


def format_pnl_line(entry_price: float, exit_price: float, quantity: float) -> str:
    row = pnl_at_price(entry_price, exit_price, quantity)
    sign = "+" if row.pnl_per_unit >= 0 else ""
    per = f"{sign}{row.pnl_per_unit:,.4f} ({sign}{row.pnl_pct:.2f}%) per unit"
    if quantity != 1.0:
        total_sign = "+" if (row.pnl_total or 0) >= 0 else ""
        per += f"; {total_sign}{row.pnl_total:,.2f} on {quantity:,.4g} units"
    return per


def cost_basis_header(
    ticker: str,
    entry_price: float,
    quantity: float,
    trade_date: str,
    current_price: float | None = None,
) -> str:
    """Markdown block prepended to the agent report."""
    lines = [
        "# Technical exit analysis",
        "",
        f"- **Ticker:** {ticker}",
        f"- **Analysis date:** {trade_date}",
        f"- **Your average entry:** {entry_price:,.4f}",
        f"- **Position size:** {quantity:,.4g} units",
    ]
    if current_price is not None and current_price > 0:
        unreal = pnl_at_price(entry_price, current_price, quantity)
        sign = "+" if unreal.pnl_per_unit >= 0 else ""
        lines.append(f"- **Mark vs entry (if cited in report):** {current_price:,.4f} "
                     f"({sign}{unreal.pnl_pct:.2f}% per unit)")
    lines.append("")
    lines.append(
        "_Not financial advice. Exit levels below are model-generated from technical data; "
        "verify prices with your broker._"
    )
    lines.append("")
    return "\n".join(lines)
