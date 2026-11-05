"""Save exit advisor reports under results_dir."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ta_exit.pnl import cost_basis_header
from tradingagents.dataflows.symbols import safe_ticker_component


def build_report_document(
    ticker: str,
    entry_price: float,
    quantity: float,
    trade_date: str,
    body: str,
) -> str:
    header = cost_basis_header(ticker, entry_price, quantity, trade_date)
    return f"{header}\n## Exit plan (technical)\n\n{body.strip()}\n"


def save_report(
    content: str,
    ticker: str,
    trade_date: str,
    results_dir: str | Path,
) -> Path:
    root = Path(results_dir) / "ta_exit"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = safe_ticker_component(ticker)
    path = root / f"{safe}_{trade_date}_{stamp}.md"
    path.write_text(content, encoding="utf-8")
    return path
