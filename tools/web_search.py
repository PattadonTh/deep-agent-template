from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """
    Search the internet for current information.
    Use for: recent events, documentation, technical specs, anything time-sensitive.
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        search = DuckDuckGoSearchRun()
        return search.invoke(query)
    except Exception as e:
        return f"Search failed: {e}"
