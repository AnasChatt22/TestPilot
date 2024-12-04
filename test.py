import nest_asyncio
import asyncio
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser

# Apply nest_asyncio to allow nested event loops in Jupyter notebooks
nest_asyncio.apply()

async def main():
    # Create an asynchronous Playwright browser instance
    async_browser = create_async_playwright_browser()

    # Initialize the PlayWrightBrowserToolkit with the browser instance
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)

    # Retrieve the available tools from the toolkit
    tools = toolkit.get_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    # Access specific tools by their names
    navigate_tool = tools_by_name["navigate_browser"]
    get_elements_tool = tools_by_name["get_elements"]

    # Navigate to the desired URL
    url = "https://web.archive.org/web/20230428133211"
    navigation_result = await navigate_tool.arun({"url": url})
    print(f"Navigation Result: {navigation_result}")

    # Extract elements using a CSS selector
    selector = ".container__headline"
    attributes = ["innerText"]
    elements = await get_elements_tool.arun({"selector": selector, "attributes": attributes})
    print(f"Extracted Elements: {elements}")

    # Close the browser after operations are complete
    await async_browser.close()

# Run the asynchronous main function
asyncio.run(main())
