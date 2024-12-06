from langchain.agents import initialize_agent, Tool
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults  # Import de l'outil Tavily
from groq import Groq
import os
import dotenv

# Charger les variables d'environnement
dotenv.load_dotenv()

# Configuration de Groq API
api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")  # Clé API pour Tavily
client = Groq(api_key=api_key)

# Définir l'outil pour l'agent (ajout de TavilySearchResults)
def generate_test_cases_with_agent(prompt):
    try:
        # Création de l'agent avec les outils (Groq + Tavily)
        tools = [
            Tool(
                name="Test Case Generator",
                func=lambda x: client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "system", "content": x}],
                    temperature=0.5,
                    max_tokens=2000,  # Augmentation du nombre de tokens
                    top_p=0.7
                ).choices[0].message.content,
                description="Génère des cas de test et des scripts Python"
            ),
            Tool(
                name="Tavily Search",
                func=TavilySearchResults(max_results=1, tavily_api_key=tavily_api_key).run,  # Appel à l'outil Tavily pour effectuer une recherche
                description="Effectue une recherche dans les résultats de Tavily"
            )
        ]

        agent = initialize_agent(
            tools,
            llm=ChatGroq(api_key=api_key),
            agent_type="zero-shot-react-description",  # Type d'agent que vous utilisez
            verbose=False,  # Désactive l'affichage dans le terminal
            handle_parsing_errors=True  # Gérer les erreurs de parsing
        )

        # Gestion de la pagination des réponses
        response = ""
        stop_condition = False
        continuation_prompt = "Continue à partir de là où tu t'es arrêté."
        while not stop_condition:
            temp_result = agent.run(prompt)
            response += temp_result
            if "Fin des cas de test" in temp_result or len(temp_result) < 2000:
                stop_condition = True
            else:
                prompt = continuation_prompt

        return response

    except Exception as e:
        return f"Erreur dans l'exécution de l'agent : {e}"

# Définir le prompt pour les exigences en français, en demandant plusieurs cas de test
exigences = """
Le but est de répondre à des questions concernant le site web de démonstration OpenCart. Voici un extrait de la conversation :

1. L'utilisateur a partagé un lien vers le site https://www.opencart.com/index.php?route=cms/demo et souhaite tester les pages de ce site.
2. L'utilisateur demande le prix d'un iPhone sur ce site de démonstration, en précisant que ce n'est pas un prix réel mais juste pour le test.

Les tâches de l'agent LLM sont les suivantes :
- Analyser la question pour identifier l'information demandée (par exemple, le prix de l'iPhone).
- Utiliser le contexte de la page d'OpenCart pour répondre de manière précise aux questions posées.
- Générer les réponses appropriées, comme le prix de l'iPhone, basé sur les informations visibles sur la page de démonstration d'OpenCart.

Le modèle doit répondre aux questions comme suit :
- Quel est le prix d'un iPhone sur le site de démonstration OpenCart ?
"""

# Création du prompt pour l'agent
prompt = f"""
Tu es un agent virtuel pour OpenCart. À partir des exigences suivantes, réponds aux questions avec précision en utilisant les informations du site de démonstration OpenCart :

Exigences :
{exigences}

Questions :
1. Quel est le prix de l'iPhone sur le site de démonstration d'OpenCart ?
2. Quels autres produits sont visibles sur le site de démonstration ?
3. Peut-on tester le panier d'achat sur le site ?
"""

# Appeler l'agent pour générer la réponse aux questions
result = generate_test_cases_with_agent(prompt)

# Sauvegarder le résultat dans un fichier
output_file = "test_results_opencart.txt"
if result and result.strip():
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(result)
else:
    # Pas de résultats générés
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("Aucun résultat généré. Veuillez vérifier l'agent ou le prompt.")