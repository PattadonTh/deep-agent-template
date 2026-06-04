from langchain_core.tools import tool


@tool
def example_tool(query: str) -> str:
    """
    Describe what this tool does — the agent reads this docstring
    to decide when and how to use it.
    """
    # replace with your logic
    return f"result for {query}"
