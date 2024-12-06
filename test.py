import dotenv
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.tools.playwright.utils import create_async_playwright_browser, create_sync_playwright_browser
from langchain_groq import ChatGroq
import os
# Charger les variables d'environnement
dotenv.load_dotenv()

# Configuration de Groq API
api_key = os.getenv("GROQ_API_KEY")

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = api_key
async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)

tools = toolkit.get_tools()

tools_by_name = {tool.name: tool for tool in tools}
navigate_tool = tools_by_name["navigate_browser"]
get_elements_tool = tools_by_name["get_elements"]

print(tools)

# conversational agent memory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)

from langchain.agents import initialize_agent
# Set up the turbo LLM
turbo_llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=1024,
    )
from langchain.agents import AgentType

# create our agent
agent = initialize_agent(
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=turbo_llm,
    verbose=True,
)

out = agent.run("Is there an article about Clubhouse on https://techcrunch.com/? today")

print(out)