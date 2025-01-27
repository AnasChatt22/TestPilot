from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser, create_sync_playwright_browser
from langchain_groq import ChatGroq

import os
import dotenv


# Charger les variables d'environnement
dotenv.load_dotenv()

# Configuration de Groq API
api_key = os.getenv("GROQ_API_KEY")

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = api_key

# Initialiser le navigateur asynchrone
sync_browser = create_sync_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)


# Extraire les outils nécessaires
tools = toolkit.get_tools()
print(tools)

# ajouter une fonction qui interagit avec un llm (outil)


def main():


    # Initialiser un modèle Groq avec LangChain
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
    )
    # initialize conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=10,
        return_messages=True
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        memory=conversational_memory
    )

    # URL, Nom du produit
    url = "https://www.opencart.com/index.php?route=common/home"
    product_name = "iStore Theme"
    product_name_1 = "iStore Theme"
    product_name_2 = "PDF Invoice Pro"

    prompt_getPrice = (
        f"Trouve le prix du produit '{product_name}' à partir de la plateforme eCommerce : {url}. "
        "Récupère uniquement le texte contenant le prix et renvoie uniquement le montant."
    )

    prompt_checkPrice = (
        f"Vérifie si le prix du produit '{product_name}' est égal à 34.99 dans la plateforme eCommerce : {url}. "
        "Si le produit coûte 30 euros, confirme que le prix est correct. Sinon, indique le prix actuel du produit."
    )

    prompt_comparePrices = (
        f"Compare le prix du produit '{product_name_1}' avec celui du produit '{product_name_2}' dans la plateforme eCommerce : {url}."
        "Indique lequel des deux est le moins cher et donne les prix de chacun."
    )

    try:
        result = agent.run(prompt_getPrice)
        print(f"Résultat : {result}")
    except ValueError as e:
        print(f"Erreur de valeur : {e}")
    except Exception as e:
        print(f"Une erreur imprévue est survenue : {e}")


# Lancer le programme
if __name__ == "__main__":
    main()
