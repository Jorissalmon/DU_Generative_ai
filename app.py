import streamlit as st
import openai
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
# 1. Initialiser l'API OpenAI
load_dotenv()  # Charger les variables depuis le fichier .env
#openai_api_key = os.getenv("OPENAI_API_KEY")  # Récupérer la clé API d'OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 2. Fonction pour charger et splitter les documents
def load_and_split_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Splitter les documents en petits morceaux
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs

# 3. Fonction pour générer des embeddings avec OpenAI et les stocker dans Chroma
def store_embeddings(docs, persist_directory):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # Passer la clé API ici
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()  # Sauvegarder les embeddings dans Chroma
    return vectordb

# 4. Fonction pour interroger Chroma et récupérer les documents pertinents
def get_relevant_docs(vectordb, query):
    docs = vectordb.max_marginal_relevance_search(query, k=4, fetch_k=10, lambda_mult=0.5)  # Récupérer les 4 documents les plus pertinents
    return docs

# 5. Fonction pour générer une réponse à partir des documents récupérés
def generate_response(query, qa, relevant_docs):

    # On construit une chaîne avec les documents pertinents pour donner un contexte au modèle
    docs_content = "\n\n".join([doc.page_content for doc in relevant_docs])

    template = """Use the following pieces of context to answer the question in a detailed manner. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Provide as much relevant information as possible based on the context. 
    {docs_content}

    Question: {query}
    Detailed Answer:"""

    # Formater le template avec les documents et la question
    prompt = template.format(docs_content=docs_content, query=query)


    # Utiliser la chaîne QA pour générer une réponse
    result = qa({"question": prompt})
    return result['answer']

# 6. Interface Streamlit
st.title("Chatbot pour tes documents")

# Uploader un document
uploaded_file = st.file_uploader("Uploader un document", type=["pdf"])

# Stocker les documents et les embeddings
persist_directory = "chroma_db"
vectordb = None

if uploaded_file is not None:
    # Charger et splitter les documents
    with open("temp_uploaded_file.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    docs = load_and_split_documents("temp_uploaded_file.txt")
    
    # Stocker les embeddings dans Chroma
    vectordb = store_embeddings(docs, persist_directory)
    st.write("Document chargé")

# Prendre une question de l'utilisateur
query = st.text_input("Posez une question :")

if st.button("Poser la question") and vectordb and query:
    # 1. Chercher les documents pertinents
    relevant_docs = get_relevant_docs(vectordb, query)

    # 2. Créer le retriever
    retriever = vectordb.as_retriever()

    # 3. Créer la chaîne QA avec mémoire
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    
    # 4. Générer la réponse à partir des documents récupérés
    response = generate_response(query, qa, relevant_docs)
    
    st.write("Réponse : ", response)
