import logging
from time import sleep as _sleep

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool(
    description="""Blocks execution for the given amount of time before returning.
Good for waiting for concurrent processes to finish.

# Example
You have a compilation process you are running on background.
```
[Start compilation]
sleep(10)
[Check if it's done]
# continue sleeping or doing something else if not done.
sleep(10)
...
```"""
)
def sleep(seconds: float) -> str:
    logger.info("Sleeping for %s second(s)", seconds)
    _sleep(seconds)
    logger.info("Done sleeping")
    return f"Slept for {seconds} second(s)."


@tool(
    description="""Preforms a simple web search and returns a summary of the top search results.
Use this for retrieving simple real-time information."""
)
def web_search(query: str) -> str:
    from ddgs.ddgs import DDGS

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region="us-en", max_results=5)
            if not results:
                results = ddgs.news(query, region="us-en", max_results=5)
                if not results:
                    return f"No results found for '{query}'."
            output = [f"Web search results for '{query}':"]
            for i, res in enumerate(results, 1):
                title = res.get("title") or res.get("href")
                url = res.get("href")
                snippet = res.get("body") or res.get("description")
                output.append(f"# {i}. {title}\n## URL\n{url}\n\n## Snippet\n{snippet}")

            return "\n\n\n".join(output)
    except Exception as e:
        logger.error("Error while doing web search: %s", e)
        return f"Error during web search: {e}"


...  # and more basic stuff
