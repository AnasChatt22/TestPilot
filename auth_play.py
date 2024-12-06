from langchain import hub
from langchain.agents import initialize_agent, Tool, AgentType, create_openai_tools_agent
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
sync_browser = create_sync_playwright_browser() # changement de sync
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)#

# Extraire les outils nécessaires
tools = toolkit.get_tools()
prompt = hub.pull("hwchase17/openai-tools-agent")


# ajouter une fonction qui interagit avec un llm (outil)


def main():


    # Initialiser un modèle Groq avec LangChain7777
    llm = ChatGroq(
        model="llama3-70b-8192",
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

    # URL, Nom du produit, et sélecteur CSS pour extraire le prix
    url = "https://boutique.univ-angers.fr/index.php?controller=authentication&back=my-account"  # Remplacez par l'URL réelle
    email = "csvhjsc@gmail.com"  # Remplacez par le nom réel du produit
    password = "extensionname23425"  # Assurez-vous que le sélecteur CSS est correct

    # Utiliser l'agent pour extraire le prix
    prompt = (
        f"Tu doit essayer de s'authentifier sur le site : {url} avec cet email {email} et ce mot de passe {password}."
    )

    try:
        result = agent.invoke(prompt)
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")

# Lancer le programme
if __name__ == "__main__":
    main()
