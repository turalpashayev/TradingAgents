from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import get_indicators, get_stock_data, get_verified_market_snapshot

TOOLS = (
    get_stock_data,
    get_indicators,
    get_verified_market_snapshot,
)


def create_exit_advisor(llm):
    """Single technical analyst focused on exit timing from a known entry price."""

    def exit_advisor_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        entry = state["entry_price"]
        qty = state["quantity"]
        position_context = state["position_context"]

        system_message = (
            """You are a **technical exit advisor**. The user already owns this instrument and wants to know
**when and why to exit**, based on technical analysis only (no fundamentals, news, or sentiment).

You know their cost basis:
"""
            + position_context
            + """

Your job:
1. Call **get_stock_data** first, then **get_indicators** for a focused set (e.g. close_50_sma, close_200_sma,
   close_10_ema, rsi, macd, macds, macdh, atr, boll, boll_ub, boll_lb — up to 8, non-redundant).
2. Before your final report, call **get_verified_market_snapshot** and treat it as the source of truth for
   last price and recent OHLCV. Do not invent price levels.
3. Explain **current technical regime** (trend, momentum, volatility) relative to their **entry**.
4. Propose **2–4 concrete exit scenarios**, each with:
   - A **price level** or clear rule (e.g. daily close below 50 SMA)
   - **Why** (which indicators / structure support it)
   - **Unrealized P&L vs their entry** at that level: show percent and, because they hold """
            + f"{qty:,.4g}"
            + """ units, the **total dollar P&L** at that level (use simple arithmetic: (exit − entry) × quantity).
5. Recommend a **primary stop / max-loss level** (often ATR- or structure-based) and a **take-profit or de-risk**
   path if the trend is still favorable.
6. End with a Markdown table: Scenario | Exit level | Trigger | Technical reason | P&L vs entry (% and $).

If data is missing, say what you could not fetch; do not guess numbers.

This is research output, not a order instruction."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant with tools for market data and indicators."
                    " Use tools to ground every price claim."
                    " You have access to: {tool_names}."
                    " Today's date is {current_date}. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join(t.name for t in TOOLS))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, TOOLS, state["messages"])

        return {
            "messages": [result],
            "exit_report": report,
        }

    return exit_advisor_node
