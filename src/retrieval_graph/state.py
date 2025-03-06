from typing import Annotated, List, Sequence
from dataclasses import dataclass, field
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


def add_queries(existing: Sequence[str], new: Sequence[str]) -> Sequence[str]:
    """Combine existing queries with new queries.

    Args:
        existing (Sequence[str]): The current list of queries in the state.
        new (Sequence[str]): The new queries to be added.

    Returns:
        Sequence[str]: A new list containing all queries from both input sequences.
    """
    return list(existing) + list(new)

@dataclass
class State:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    queries: Annotated[list[str], add_queries] = field(default_factory=list)
    context: str = field(default_factory=str)
    sources: List[str] = field(default_factory=list)
    language: str = field(default_factory=str)
    provider: str = field(default_factory=str)
    summary: str = field(default_factory=str)