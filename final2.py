from langchain.agents import initialize_agent, Tool
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
    model="llama3-8b-8192",
    temperature=0.5,
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
        return response  # Nettoyer le texte de la réponse
    except Exception as e:
        return f"Erreur lors de l'appel de l'agent : {e}"

# Définir les outils pour l'agent
tools = [
    Tool(
        name="GenerateTestCases",
        func=generate_test_cases,
        description="Génère des cas de test basés sur des exigences."
    )
]


# Initialiser l'agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Exécuter l'agent avec un prompt utilisateur
exigences = "L'utilisateur doit pouvoir s'authentifier"

prompt = f"""
Tu es un ingénieur de test logiciel expérimenté. À partir des exigences suivantes :
{exigences}

Génère une liste complète de cas de test fonctionnels en suivant ces directives :
- Fournis une description concise mais précise de chaque cas de test.
- Spécifie les préconditions nécessaires pour chaque test.
- Identifie les actions principales que l'utilisateur ou le système doit effectuer.
- Décris le résultat attendu pour valider le bon fonctionnement.
- Évite d'inclure des scripts ou des détails techniques liés au code.

Retourne les cas de test sous la forme d'une liste numérotée, rédigée en français.
"""

# Exécuter l'agent
try:
    result = agent.invoke(prompt)
    print(result)
except Exception as e:
    print(f"Erreur lors de l'exécution de l'agent : {e}")
