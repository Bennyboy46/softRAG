import re
from typing import Tuple


def should_use_graph(question: str) -> bool:
    if not question or not question.strip():
        return False
    question_text = question.lower()
    signals = [
        "called by",
        "what calls",
        "who calls",
        "what does",
        "execution flow",
        "dependency",
        "dependencies",
        "imports",
        "imported by",
        "caller",
        "callee",
        "route",
        "endpoint",
        "flow",
        "calls",
        "used by",
        "where is",
    ]
    for signal in signals:
        if signal in question_text:
            return True
    return False


def route_query(question: str) -> Tuple[bool, str]:
    return should_use_graph(question), "graph" if should_use_graph(question) else "rag"
