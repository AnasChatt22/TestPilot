from langchain.agents import initialize_agent, Tool, AgentType
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import os
import dotenv


## Charger les variables d'environnement
dotenv.load_dotenv()

# Configuration de Groq API
api_key = os.getenv("GROQ_API_KEY")

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = api_key


# Initialiser un modèle Groq avec LangChain
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=1024,
)


# Définir une fonction pour interagir avec Groq
def generate_test_cases(prompt):
    """
    Utilise Groq pour générer des cas de test basés sur des exigences.
    """
    try:
        # Appel direct à l'instance LLM de Groq
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Erreur lors de l'appel de l'agent : {e}"

# Définir les outils pour l'agent
tools = [
    Tool(
        name="GenerateTestCases",
        func=generate_test_cases,
        description="Génère des cas de test basés sur une exigence."
    )
]


# initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

memory = MemorySaver()

AgentType = 'conversational-react-description'
AgentType2 = 'zero-shot-react-description'
AgentType3 = 'structured-chat-zero-shot-react-description'

# Initialiser l'agent
agent = initialize_agent(
    agent=AgentType,
    tools=tools,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True,
    #checkpointer=memory
    memory = conversational_memory

)

# Exécuter l'agent avec un prompt utilisateur
exigences = "l'utilisateur doit pouvoir s'authentifier."

prompt = f"""Tu es un ingénieur de test logiciel expérimenté. 
À partir de l'exigence suivante :
{exigences}
Génère une liste complète de cas de test fonctionnels en suivant ces directives :
- N'inclut pas des scripts, des exemples de code ou des détails techniques liés au développement.
- Concentre-toi uniquement sur les descriptions fonctionnelles des tests.
- Rédige les cas de test, sous forme de liste numérotée avec ses descriptions.

Format de l'output :
1. **Titre du cas**  
   Description : ...  
   Résultat attendu : ...

2. **Titre du cas**  
   Description : ...  
   Résultat attendu : ...
...
"""

# Exécuter l'agent
try:
    result = agent(prompt)
    print(result)
except Exception as e:
    print(f"Erreur lors de l'exécution de l'agent : {e}")
