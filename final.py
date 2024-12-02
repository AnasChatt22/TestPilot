from langchain.agents import initialize_agent, Tool
from langchain_groq import ChatGroq
from groq import Groq
import os
import dotenv

# Charger les variables d'environnement
dotenv.load_dotenv()

# Configuration de Groq API
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Définir une fonction pour appeler l'API Groq
def generate_test_cases(prompt):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_tokens=1024,
            top_p=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur lors de l'appel API : {e}"

# Définir les outils utilisés par l'agent
tools = [
    Tool(
        name="Generation_Cas_de_Test",
        func=generate_test_cases,
        description="Générer des cas de test à partir d'exigences"
    )
]

# Initialiser un modèle de chat avec LangChain
llm = ChatGroq(model="llama3-8b-8192")


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

result = agent.invoke(prompt)

print(result)
