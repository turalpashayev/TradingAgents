import logging
from collections import Counter

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from ta_exit.agents.exit_advisor import create_exit_advisor
from ta_exit.state import ExitAgentState
from tradingagents.agents.analysts.turn import WRAP_UP

logger = logging.getLogger(__name__)


def _tools_or_done(state) -> str:
    return "tools" if state["messages"][-1].tool_calls else END


def _build_exit_graph(llm, max_tool_rounds: int):
    """One tool-using agent: model ↔ tools until report or round limit."""
    from tradingagents.agents.tools import get_indicators, get_stock_data, get_verified_market_snapshot

    tools = [get_stock_data, get_indicators, get_verified_market_snapshot]
    agent = create_exit_advisor(llm)

    graph = StateGraph(ExitAgentState)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")

    def calls(messages):
        return [call["name"] for m in messages for call in (getattr(m, "tool_calls", None) or [])]

    def rounds(messages) -> int:
        return sum(1 for m in messages if getattr(m, "tool_calls", None))

    def more_or_wrap_up(state) -> str:
        return "wrap_up" if rounds(state["messages"]) >= max_tool_rounds else "agent"

    def wrap_up(state):
        repeated = ", ".join(f"{name} x{n}" for name, n in Counter(calls(state["messages"])).most_common())
        logger.warning("Exit advisor used %d tool rounds (%s); asking for final report", max_tool_rounds, repeated)
        return agent({**state, "messages": [*state["messages"], HumanMessage(WRAP_UP)]})

    graph.add_node("tools", ToolNode(tools))
    graph.add_node("wrap_up", wrap_up)
    graph.add_conditional_edges("agent", _tools_or_done, ["tools", END])
    graph.add_conditional_edges("tools", more_or_wrap_up, ["agent", "wrap_up"])
    graph.add_edge("wrap_up", END)
    return graph.compile()


class ExitAdvisorGraph:
    """Minimal LangGraph: one technical exit advisor, no debates."""

    def __init__(self, config: dict | None = None, callbacks: list | None = None):
        import os

        from tradingagents.dataflows.config import set_config
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.llm_clients import create_tier_client

        self.config = dict(config or DEFAULT_CONFIG)
        set_config(self.config)
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)
        extra = {"callbacks": callbacks} if callbacks else {}
        self.llm = create_tier_client(self.config, "quick", **extra).get_llm()
        max_tool_rounds = self.config["max_tool_rounds"]
        max_recur = self.config["max_recur_limit"]
        if 2 * max_tool_rounds + 2 >= max_recur:
            raise ValueError(
                f"max_tool_rounds={max_tool_rounds} needs max_recur_limit above {2 * max_tool_rounds + 2}"
            )
        self.graph = _build_exit_graph(self.llm, max_tool_rounds)

    def resolve_instrument_context(self, ticker: str, trade_date: str) -> str:
        from tradingagents.agents.context import build_instrument_context, resolve_instrument_identity

        identity = resolve_instrument_identity(ticker)
        return build_instrument_context(ticker, "stock", identity, trade_date)

    def propagate(
        self,
        ticker: str,
        entry_price: float,
        *,
        quantity: float = 1.0,
        trade_date: str | None = None,
    ) -> tuple[dict, str]:
        from datetime import datetime

        from tradingagents.dataflows.config import run_config
        from tradingagents.dataflows.date_window import get_current_date

        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")

        trade_date = trade_date or get_current_date()
        value = str(trade_date)
        try:
            canonical = datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") == value
        except ValueError:
            canonical = False
        if not canonical:
            raise ValueError(f"trade_date must be YYYY-MM-DD, got {trade_date!r}")
        if value > get_current_date():
            raise ValueError(f"trade_date cannot be in the future: {value}")

        symbol = ticker.strip().upper()
        instrument_context = self.resolve_instrument_context(symbol, value)
        position_context = (
            f"Average entry price: {entry_price:,.4f} per unit. "
            f"Position size: {quantity:,.4g} units. "
            f"Total cost basis: {entry_price * quantity:,.2f}."
        )
        task = (
            f"Analyze technical exit options for {symbol} as of {value}. "
            f"I bought at {entry_price:,.4f} and hold {quantity:,.4g} units."
        )

        initial = {
            "messages": [("human", task)],
            "company_of_interest": symbol,
            "trade_date": value,
            "instrument_context": instrument_context,
            "entry_price": float(entry_price),
            "quantity": float(quantity),
            "position_context": position_context,
            "exit_report": "",
        }

        with run_config(self.config):
            final = self.graph.invoke(
                initial,
                config={"recursion_limit": self.config["max_recur_limit"]},
            )

        report = final.get("exit_report") or ""
        if not report.strip():
            last = final["messages"][-1]
            content = getattr(last, "content", last)
            report = content if isinstance(content, str) else str(content)

        return final, report
