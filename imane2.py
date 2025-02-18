from langchain.agents import initialize_agent, Tool
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from langchain_community.agent_toolkits import load_tools
from groq import Groq
import os
import dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from webdriver_manager.chrome import ChromeDriverManager  # Utilisation de webdriver-manager

# Charger les variables d'environnement
dotenv.load_dotenv()

# Configurer les API keys
api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Initialiser Groq client
client = Groq(api_key=api_key)

# Fonction pour générer des scripts de test avec LangChain + Groq
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
            verbose=True,  # Affichage détaillé dans le terminal pour débogage
            handle_parsing_errors=True  # Gérer les erreurs de parsing
        )

        # Exécution de l'agent avec le prompt
        response = agent.run(prompt)

        # Afficher le script généré pour débogage
        print("Script généré :")
        print(response)

        return response

    except Exception as e:
        return f"Erreur dans l'exécution de l'agent : {e}"

# Fonction pour exécuter les scripts générés avec Selenium
def execute_test_script(script):
    results = {}
    driver = None  # Définir driver ici pour éviter l'erreur UnboundLocalError
    try:
        # Créer un WebDriver avec le gestionnaire de ChromeDriver
        options = webdriver.ChromeOptions()  # Options pour Chrome
        options.add_argument('--headless')  # Exécuter en mode headless pour ne pas ouvrir le navigateur
        options.add_argument("--disable-gpu")  # Désactiver GPU en mode headless (recommandé)
        options.add_argument("--no-sandbox")  # Utiliser un environnement sandbox pour plus de stabilité

        # Initialiser le WebDriver en utilisant le Service
        service = Service(ChromeDriverManager().install())  # Définir le service
        driver = webdriver.Chrome(service=service, options=options)  # Utiliser le service ici
        driver.maximize_window()

        # Vérifier si le script est une chaîne valide et exécuter
        if isinstance(script, str):
            exec(script)  # Exécute dynamiquement le script généré par LangChain
        else:
            results["Erreur"] = "Le script généré n'est pas valide."

        # Résultats d'exécution
        results["Statut"] = "Test exécuté avec succès"

    except Exception as e:
        results["Erreur"] = str(e)

    finally:
        if driver:
            # Fermer le navigateur
            driver.quit()

    return results

# Prompt pour l'agent LangChain
prompt = """
Tu es un agent virtuel spécialisé dans l'automatisation des tests avec Selenium. Génère un script Python complet qui suit ces étapes pour tester le site de démonstration OpenCart : 

1. Ouvre le site de démonstration OpenCart (https://demo.opencart.com/).
2. Navigue sur les pages suivantes : 
    - Page d'accueil (si nécessaire).
    - Section "Phones & PDAs" dans le menu.
    - Section "Laptops" dans le menu.
3. Récupère le prix de l'article 'iPhone' (ou un produit similaire) dans la section 'Phones & PDAs'.
4. Récupère le prix de l'article 'MacBook' (ou un produit similaire) dans la section 'Laptops'.
5. Calcule la somme des prix des produits 'iPhone' et 'MacBook'.
6. Ajoute l'article 'iPhone' au panier.
7. Vérifie et affiche le nombre d'articles dans le panier.
8. Affiche le contenu du panier.
9. Ferme le navigateur à la fin du test.

Le script doit respecter ces bonnes pratiques :

- Utilise WebDriverWait pour garantir que les éléments sont chargés avant toute interaction (par exemple, attendre que le bouton "Ajouter au panier" soit cliquable).
- Assure-toi que toutes les interactions avec les éléments se fassent avec click(), find_element(), ou find_elements(), en utilisant des identifiants ou des sélecteurs CSS appropriés.
- Utilise try-except pour gérer les exceptions qui peuvent se produire durant le test (par exemple, si un élément n'est pas trouvé).
- Sois précis dans la récupération des prix, en t'assurant que les valeurs extraites ne contiennent pas de caractères indésirables comme des symboles ou des espaces inutiles.
- Utilise time.sleep() uniquement si nécessaire pour la synchronisation.

Assure-toi que le script généré soit fonctionnel et puisse être exécuté sans erreur sur un environnement Selenium classique avec Chrome je veux que le code m'affiche pas cette erreur:Erreur :" invalid syntax (<string>, line 1)".

Voici un exemple de code pour te guider :

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from webdriver_manager.chrome import ChromeDriverManager

# Initialisation du WebDriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Exécution sans interface graphique
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# Démarrage du navigateur
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get('https://demo.opencart.com/')

# Attendre que le menu soit visible
WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, 'top-menu')))

# Naviguer vers les pages 'Phones & PDAs' et 'Laptops'
phones_link = driver.find_element(By.LINK_TEXT, 'Phones & PDAs')
phones_link.click()
WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.product-layout')))

macbook_link = driver.find_element(By.LINK_TEXT, 'Laptops & Notebooks')
macbook_link.click()
WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.product-layout')))

# Récupérer les prix des produits
iphone_price = driver.find_element(By.CSS_SELECTOR, '.product-thumb .price').text
macbook_price = driver.find_element(By.CSS_SELECTOR, '.product-thumb .price').text

# Comparer les prix
iphone_price_value = float(iphone_price.replace('$', '').strip())
macbook_price_value = float(macbook_price.replace('$', '').strip())
total_price = iphone_price_value + macbook_price_value
print(f"Total price of iPhone and MacBook: ${total_price}")

# Ajouter l'iPhone au panier
add_to_cart_button = driver.find_element(By.CSS_SELECTOR, '.button-group .button-cart')
add_to_cart_button.click()

# Vérifier le contenu du panier
cart_icon = driver.find_element(By.ID, 'cart')
cart_icon.click()
WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.cart-quantity')))
items_in_cart = driver.find_element(By.CSS_SELECTOR, '.cart-quantity').text
cart_contents = driver.find_element(By.CSS_SELECTOR, '.table tbody').text

print(f"Number of items in cart: {items_in_cart}")
print(f"Cart contents: {cart_contents}")

# Fermer le navigateur
driver.quit()
"""

# Étape 1 : Générer le script de test
generated_script = generate_test_cases_with_agent(prompt)

# Vérifier que le script généré est valide
if generated_script and isinstance(generated_script, str):
    # Étape 2 : Exécuter le script et obtenir les résultats
    test_results = execute_test_script(generated_script)

    # Étape 3 : Afficher uniquement les résultats dans le terminal
    if "Erreur" in test_results:
        print(f"Erreur : {test_results['Erreur']}")
    else:
        print(f"Statut : {test_results['Statut']}")
else:
    print("Erreur : Le script généré est invalide ou vide.")