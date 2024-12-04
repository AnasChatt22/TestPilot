from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_groq import ChatGroq
import dotenv

# Charger les variables d'environnement
dotenv.load_dotenv()

# Configuration de ChatGroq pour utiliser LLaMA 3
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.5,
    max_tokens=1024,
)

# Charger le contenu du site
url = "http://opencart.kereval.com/opencart/"
loader = AsyncChromiumLoader([url])
docs = loader.load()

# Transformer le contenu en texte structuré
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(
    docs, tags_to_extract=["p", "li", "div", "a"]
)


# Définition du schéma pour l'extraction
schema = {
    "type": "object",
    "properties": {
        "product_name": {
            "type": "string",
            "description": "The name of the product"
        },
        "product_price": {
            "type": "string",
            "description": "The price of the product in USD"
        }
    },
    "required": ["product_name", "product_price"]
}


# Traitement des documents
for doc in docs_transformed:
    print("Processing document...")
    print(doc.page_content)
    try:
        # Créer une invite personnalisée pour le modèle
        prompt = f"""
        Extract the following information from the provided text based on this schema:
        Schema: {schema}

        Text:
        {doc.page_content}
        """

        # Passer l'invite au modèle ChatGroq
        response = llm.invoke(prompt)
        print(response)
    except Exception as e:
        print(f"Erreur lors de l'extraction : {e}")
    print("finished processing document.....")