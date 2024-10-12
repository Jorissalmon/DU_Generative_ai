import streamlit as st 
import random
import os
import markdown2
import markdown
import faiss
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from fpdf import FPDF
from openai import OpenAI
from langchain.vectorstores import FAISS

# 1. Chargement de l'API OpenAI
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]
eleven_labs_api_key=st.secrets["ELEVENLABS_API_KEY"]
client = OpenAI(
    api_key=openai_api_key,
)


##################################
###########Fonctions##############
##################################

# Fonction pour charger et splitter les documents
def load_and_split_documents(file_path):
    loader = PyPDFLoader(file_path)# Chargement du fichier
    documents = loader.load()
    
    # Splitter les documents en petits morceaux
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)#Splits de 500 caractères au maximum, avec un retour de 50 caractères à chaque fois
    docs = text_splitter.split_documents(documents)
    return docs

# Fonction pour générer des embeddings avec OpenAI et les stocker dans Chroma
def store_embeddings(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Stockage des embeddings dans FAISS
    vectordb = FAISS.from_documents(docs, embeddings)
    
    # (Optionnel) Sauvegarder le vectordb FAISS sur disque
    faiss.write_index(vectordb.index, "faiss_index")
    return vectordb

# Fonction pour interroger Chroma et récupérer les documents pertinents pour le cours
def get_relevant_docs_cours(vectordb, query, len_k):
    docs = vectordb.max_marginal_relevance_search(query, k=10, fetch_k=len_k, lambda_mult=0.7)
    docs_content = "\n\n".join([doc.page_content for doc in docs])
    print(docs_content)
    return docs_content

# Fonction pour interroger Chroma et récupérer les documents pertinents pour le chat
# @st.cache_data(show_spinner=False)
def get_relevant_docs_chat(_vectordb, query, len_k):
    docs = _vectordb.max_marginal_relevance_search(query, k=10, fetch_k=len_k, lambda_mult=0.3)
    docs_content = "\n\n".join([doc.page_content for doc in docs])
    print(docs_content)
    return docs_content

# Fonction pour sélectionner des documents au hasard
def select_random_documents(docs, num_docs=50):
    selected_docs = [docs[0]] if len(docs) > 0 else []
    
    # Sélectionner des documents supplémentaires de manière aléatoire, en excluant le premier
    if len(docs) > 1:
        selected_docs += random.sample(docs[1:], min(num_docs - 1, len(docs) - 1))
    
    return selected_docs

# Fonction pour générer une réponse à partir des documents récupérés et de la question
def generate_response(qa,prompt_template):
    # Passer le prompt formaté à la chaîne de question-réponse (qa)
    response = qa({"question": prompt_template})

    # Renvoyer directement la réponse
    return response["answer"]

# Fonction pour générer un plan structuré sur un sujet donné
def generate_structure_plan(docs):
    prompt_template = """
    À partir des documents suivants, définir le sujet et élaborez un plan détaillé. 
    Le plan doit inclure des sections principales et des sous-sections. 
    Ceci sera un plan de cours. Tu dois commencer par écrire le plan. 
    Les sections principales doivent être commencées par des ##. Met le "#Sujet :" en premier.
    Ton message de retour sera un plan détaillé avec des sections principales commençant par des ## et le sujet écrit après "#Sujet :" en premier.

    Les documents : {docs_content}
    """

    prompt = PromptTemplate(
        input_variables=["docs_content"],
        template=prompt_template
    )

    # Formater le prompt avec les documents fournis
    formatted_prompt = prompt.format(docs_content=docs)
    sujet = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": formatted_prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )

    # Récupérer le contenu du message (le plan)
    plan = sujet.choices[0].message.content
    print("Plan proposé :\n", plan)

    return plan

# Fonction pour générer l'introduction, le développement et la conclusion du cours
def generate_course_sections(subject_parts, relevant_docs, sujet):
    course_content = ""

    # # Ajouter une introduction avant toutes les sections
    # introduction_prompt = f"""
    # Veuillez rédiger une introduction résumant les points clés des sections {sujet}.
    # Vous avez à votre disposition les éléments de développement suivant :{relevant_docs}.
    # Restez conçit pour introduire en généralité le sujet, afin que le développement se fasse bien par la suite.
    # L'introduction doit être concise et directe. Limitez votre réponse à 5 phrases et à un maximum de 100 mots.
    # """
    # introduction_result = client.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": introduction_prompt,
    #         }
    #     ],
    #     model="gpt-3.5-turbo",
    # )
    # course_content += f"### Introduction\n{introduction_result.choices[0].message.content}\n"

    # Itérer sur chaque partie du sujet
    for part in subject_parts:
        # Créer le prompt pour chaque partie
        formatted_prompt = f"""
        À l'aide des éléments suivants, développez la section sur '{part}' à l'aide de ces documents. 
        Assurez-vous d'inclure des exemples, des détails pertinents et des concepts clés. Ce qui a déjà été rédigé : {course_content}.
        Le développement doit être concit et clair. Limitez votre réponse à 10 phrases et à un maximum de 200 mots.
        Essaies de mettre le mieux possible  en forme le document, avec du gras de l'italique, des titres et sous titres.
        """
        
        # Générer le contenu pour cette partie
        result = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": formatted_prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )
        
        # Ajouter le contenu généré au cours
        course_content += f"###{result.choices[0].message.content}\n\n"
    
    # # Ajouter une conclusion après toutes les sections
    # conclusion_prompt = f"""
    # Veuillez rédiger une conclusion résumant les points clés des sections qui ont été développer par ce plan {sujet}
    # Voici le developpement {course_content}.
    # La conclusion doit être concise et directe. Limitez votre réponse à 5 phrases et à un maximum de 100 mots.
    # """
    # conclusion_result = client.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": conclusion_prompt,
    #         }
    #     ],
    #     model="gpt-3.5-turbo",
    # )
    
    # course_content += f"### Conclusion\n{conclusion_result.choices[0].message.content}\n"
    
    return course_content

# Fonction pour générer un fichier MP3 à partir du texte
def text_to_mp3_Openai(text, output_path):
    # Utilisation de l'API de synthèse vocale
    response = client.audio.speech.create(
    model="tts-1",
    voice="onyx",
    input=text,
    )
    response.stream_to_file(output_path)
    return output_path

##################################
###########Streamlit##############
##################################

st.title("Ton assitant de cours")

#Injection de CSS
st.markdown("""
<style>
div.stDownloadButton > button {
    background-color: #FF6347;
    color: white;
}
div.stDownloadButton > button:hover {
    background-color: #FF4500;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Uploader de plusieurs documents
uploaded_files = st.file_uploader("Uploader plusieurs documents", type=["pdf"], accept_multiple_files=True)


#################### Initialisation ############################
# Initialisation de valeurs dans le cache
if 'response' not in st.session_state:
    st.session_state.response = None 
if 'sujet' not in st.session_state:
    st.session_state.sujet = None 
if 'relevant_docs' not in st.session_state:
    st.session_state.relevant_docs = None 
    
# Stocker les documents et les embeddings
persist_directory = "chroma_db"
vectordb = None

generer_cours=False

if uploaded_files and isinstance(uploaded_files, list) and len(uploaded_files) > 0 and ('relevant_docs' not in st.session_state or st.session_state.relevant_docs is None):
    #Réinitialisation de la variable 
    generer_cours=False


    # Liste pour stocker tous les documents chargés
    all_docs = []

    for uploaded_file in uploaded_files:
        # Charger et splitter chaque document PDF
        file_path = os.path.join("temp_uploaded_file.pdf")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Splitter les documents et ajouter les morceaux à la liste
        docs = load_and_split_documents(file_path)
        all_docs.extend(docs)  # Ajouter les morceaux à la liste principale

    st.write(f"Nombre total de morceaux de documents chargés : {len(all_docs)}")
    #Enregistrement de la variable dans la session
    st.session_state.all_docs = all_docs
    # Stocker les embeddings dans Chroma
    if all_docs:  # Vérifie que nous avons des documents à stocker
        vectordb = store_embeddings(all_docs)
        st.write("Documents chargés et stockés avec succès.")
    else:
        st.error("Aucun document valide trouvé pour le stockage.")

################################ barre latérale pour les boutons de téléchargement #################################################

st.sidebar.header("Téléchargement")

##################### Générer le cours
if st.sidebar.button("Générer le cours"):
    if uploaded_files and isinstance(uploaded_files, list) and len(uploaded_files) > 0 :
        #Pour dire que le cours à déjà été générer
        generer_cours=True
        #Selection de 10 documents parmis les documents dont le 1er
        docs_randoms = select_random_documents(all_docs, 10)
        #Génère le sujet et la structure du plan
        sujet = generate_structure_plan(docs_randoms)
        #Prend les différentes parties du sujet
        subject_parts = sujet.split("##")
        subjet=subject_parts[1]
        #Récupère les principaux documents concernant le sujet
        relevant_docs = get_relevant_docs_cours(vectordb, subjet, len(all_docs))
        #Template pour créer le cours
        course_prompt_template = """
        A laide des éléments suivants et du plan, générez un cours structuré. Assurez-vous d'inclure une introduction, des concepts clés, des exemples, et une conclusion bien détaillée.
        N'oubliez pas d'organiser les sections de manière logique. Vous devez rédiger un cours complet. Détailles bien chaque partie.
        """
        #Génération du cours
        response = generate_course_sections(subject_parts[1:len(subject_parts)], relevant_docs,sujet)

        html_output = "./rendu/cours.html"
        html=markdown.markdown(response)

        # Sauvegarder les variables dans le session state
        st.session_state.response = response
        st.session_state.sujet = sujet
        st.session_state.relevant_docs = relevant_docs

        #Téléchargement
        st.sidebar.download_button(
            label="Télécharger le cours en HTML",
            data=html,
            file_name=html_output,
            mime="text/html"
        )
        st.write("### 📝 Cours")
        st.write(response)
    else :
        st.toast("Veuillez charger des documents", icon="ℹ️")


################################## 
################################## Générer des fiches de révision
if st.sidebar.button("Générer une fiche de révision"):
    if 'response' in st.session_state and st.session_state.response:
        revision_prompt_template = f"""
        Utilisez les documents suivants pour résumer les points clés du cours sous forme de fiches de révision. Les informations doivent être concises, organisées en sections claires, et inclure des définitions, des concepts importants, et des questions pour l'auto-évaluation.
        Voici les informations du cours, nhesite pas à en rajouter : {st.session_state.relevant_docs}
        Voici le plan du cours : {st.session_state.sujet}
        """
        #Interrogation LLM
        fiches = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": revision_prompt_template,
                }
            ],
            model="gpt-3.5-turbo",
        )
        #Récupération du message
        fiches_content = fiches.choices[0].message.content

        html_output_fiches = "./rendu/fiches.html"
        html_fiches=markdown.markdown(fiches_content)

        #Téléchargement
        st.sidebar.download_button(
            label="Télécharger la fiche de révision en HTML",
            data=html_fiches,
            file_name=html_output_fiches,
            mime="text/html"
        )

        st.write("### 📋 Fiches de révision")
        st.write(fiches_content)
    else :
        st.toast("Veuillez générer le cours pour générer une fiche de révision.", icon="ℹ️")

################################## 
################################## Générer le podcast
if st.sidebar.button("Générer le podcast"):
    if 'response' in st.session_state and st.session_state.response:
        podcast_prompt_template = f"""
            Vous allez rédiger le script d’un podcast captivant et accessible basé sur les documents suivants. L'objectif est de transmettre les points essentiels du cours de manière agréable et dynamique. N'hésitez pas à utiliser des exemples pertinents pour illustrer les concepts abordés.

            Voici les informations clés du cours : {st.session_state.relevant_docs}
            Voici le plan du cours à suivre : {st.session_state.sujet}

            Le podcast doit être structuré de façon à capter l'attention de l'auditeur dès le début, puis à expliquer chaque point de manière claire et détaillée. Utilisez un ton amical et engageant, et ponctuez la narration avec des anecdotes ou des exemples concrets pour aider à mieux comprendre les idées. Concluez par un résumé efficace des idées principales, en laissant l'auditeur avec des points mémorables à retenir.
            
            Structure suggérée :
            1. Introduction : Présentation du sujet et de l’objectif du podcast.
            2. Développement : Décrivez chaque point clé du cours en détail, avec des exemples concrets et des anecdotes.
            3. Conclusion : Résumez les idées principales et terminez sur une note encourageante pour l'auditeur.
            """
        #Interrogation LLM
        podcast = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": podcast_prompt_template,
                }
            ],
            model="gpt-3.5-turbo",
        )
        #Récupération du message
        podcast_content = podcast.choices[0].message.content
        print(podcast_content)
        # Génération du fichier MP3
        mp3_output_path = "./rendu/podcast.mp3"  # Chemin de sortie pour le fichier MP3
        audio_file = text_to_mp3_Openai(podcast_content, mp3_output_path)

        # Proposer le téléchargement du fichier MP3
        if audio_file:
            st.sidebar.download_button(
                label="Télécharger le podcast en MP3",
                data=open(audio_file, "rb").read(),
                file_name="podcast.mp3",
                mime="audio/mpeg"
        )
        st.write("### 🎙 Podcast")
        st.write(podcast_content)
    else :
        st.toast("Veuillez générer le cours pour générer un podcast.", icon="ℹ️")

################################## 
################################## Interface utilisateur pour poser des questions

st.write("## Poses des questions sur le cours")
if generer_cours==True:
    ### Initilisatin du modèle pour les questions/réponses
    #Utilisation de langchain, pour un model avec historique
    llm_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=llm_name, temperature=0.2, max_tokens=1000)

    #Initialisation de la mémoire
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    # Initialisation de l'historique pour le llm de question réponse.
    retriever = vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        chain_type="refine",  # Changez ceci selon vos besoins: 'stuff', 'map_reduce', 'refine'
        # return_source_documents=True
    )

####################### Champ de texte pour entrer une question
user_question = st.text_input("Poses ta question")

if st.button("Envoyer") and (user_question and st.session_state.get('last_user_question') != user_question):
    if user_question and vectordb:
        relevant_docs = get_relevant_docs_chat(vectordb, user_question, len(all_docs))
        response = generate_response(qa, f"""
        Répondre à la question posé à l'aide des documents. Vous serez pédagogue en expliquerez de manière claire la réponse à la question.
        Soyez le plus précis possible.
        Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas.    
        {relevant_docs}

        Question : {user_question}
        """)

        st.write("### 🤖 Réponse")
        st.write(response)

