from langchain_core.tools import tool


@tool
def ping() -> str:
    """Pong! (Test tool)"""
    return "Pong!"
