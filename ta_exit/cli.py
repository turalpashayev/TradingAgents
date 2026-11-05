"""CLI for the technical exit advisor (ta-exit)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ta_exit.graph.run import ExitAdvisorGraph
from ta_exit.reporting import build_report_document, save_report
from tradingagents.default_config import DEFAULT_CONFIG

app = typer.Typer(
    name="ta-exit",
    help="Technical exit advisor: when to exit a position based on TA and your entry price.",
    add_completion=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def analyze(
    ctx: typer.Context,
    ticker: str = typer.Option(..., "--ticker", "-t", help="Symbol, e.g. AAPL or NVDA"),
    entry: float = typer.Option(..., "--entry", "-e", help="Your average purchase price per unit"),
    qty: float = typer.Option(1.0, "--qty", "-q", help="Units held (for dollar P&L in the report)"),
    date: str | None = typer.Option(None, "--date", "-d", help="Analysis date YYYY-MM-DD (default: today)"),
    save: bool = typer.Option(False, "--save", "-s", help="Write report markdown under results_dir/ta_exit"),
    no_show: bool = typer.Option(False, "--no-show", help="Do not print the report to the terminal"),
):
    """Run the technical exit advisor for one ticker and cost basis."""
    if ctx.invoked_subcommand is not None:
        return
    config = DEFAULT_CONFIG.copy()
    graph = ExitAdvisorGraph(config=config)

    console.print(Panel.fit(f"[bold]Technical exit analysis[/bold] — {ticker.upper()}", border_style="cyan"))
    with console.status("[bold green]Running exit advisor (tools + LLM)..."):
        try:
            _state, body = graph.propagate(ticker, entry, quantity=qty, trade_date=date)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1) from None
        except Exception as exc:
            console.print(f"[red]Run failed:[/red] {exc}")
            raise typer.Exit(code=1) from None

    trade_date = date or _state["trade_date"]
    document = build_report_document(ticker.upper(), entry, qty, trade_date, body)

    if save:
        path = save_report(document, ticker, trade_date, config["results_dir"])
        console.print(f"[green]Saved[/green] {path}")

    if not no_show:
        console.print()
        console.print(Markdown(document))


if __name__ == "__main__":
    app()
