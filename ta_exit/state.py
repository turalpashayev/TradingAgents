from typing import Annotated

from langgraph.graph import MessagesState


class ExitAgentState(MessagesState):
    company_of_interest: Annotated[str, "Ticker under analysis"]
    trade_date: Annotated[str, "Analysis date YYYY-MM-DD"]
    instrument_context: Annotated[str, "Resolved company identity"]
    entry_price: Annotated[float, "User average purchase price"]
    quantity: Annotated[float, "Units held"]
    position_context: Annotated[str, "Human-readable cost basis for the prompt"]
    exit_report: Annotated[str, "Final technical exit report"]
