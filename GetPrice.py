from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_groq import ChatGroq
from langchain.tools import tool

import os
import dotenv
import asyncio
import nest_asyncio

# Charger les variables d'environnement
dotenv.load_dotenv()

# Configuration de Groq API
api_key = os.getenv("GROQ_API_KEY")

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = api_key

# Initialiser le navigateur asynchrone
async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)

# Extraire les outils nécessaires
tools_list = toolkit.get_tools()
tools_by_name = {tool.name: tool for tool in tools_list}
navigate_tool = tools_by_name.get("navigate_browser")
get_elements_tool = tools_by_name.get("get_elements")


async def navigate_browser(url):
    """Navigate to the given URL and return the page's HTML content."""
    try:
        response = await navigate_tool.arun({"url": url})
        print(response)
        return response
    except Exception as e:
        print(f"Error while navigating to {url}: {e}")
        return None


async def extract_price(selector):
    """Extract the price of the product using the specified CSS selector."""
    try:
        response = await get_elements_tool.arun(
            {"selector": selector, "attributes": ["innerText"]}
        )
        return response
    except Exception as e:
        print(f"Error while extracting price with selector {selector}: {e}")
        return None

nest_asyncio.apply()




async def main():


    # Initialiser un modèle Groq avec LangChain
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=1024,
    )

    # Définir les outils pour l'agent
    tools = [
        Tool(
            name="navigate_browser",
            func=navigate_browser,
            description="Navigates on browser to search for a site with url."
        ),
        Tool(
            name="extract_price",
            func=extract_price,
            description="Extracts the price of a given product from the HTML content using a CSS selector."
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    # URL, Nom du produit, et sélecteur CSS pour extraire le prix
    url = "https://www.amazon.fr/b?ie=UTF8&node=100878139031"  # Remplacez par l'URL réelle
    product_name = "Swarovski Amazon Décoration de Noël"  # Remplacez par le nom réel du produit
    product_selector = "a-price-whole"  # Assurez-vous que le sélecteur CSS est correct

    # Utiliser l'agent pour extraire le prix
    prompt = (
        f"Navigate to {url} and extract the price of {product_name} using a CSS selector {product_selector}"
    )

    try:
        result = await agent.arun(prompt)
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")

# Lancer le programme
if __name__ == "__main__":
    asyncio.run(main())
