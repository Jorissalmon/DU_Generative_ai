import streamlit as st 
import random
import os
import markdown2
import markdown
import faiss
import whisper
import requests
import openai
from io import BytesIO
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from fpdf import FPDF
from openai import OpenAI
from langchain.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi
from PIL import Image
import pytesseract  # Pour faire l'OCR
import assemblyai as aai # Speech to text

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #Erreur de librairie

# 1. Chargement des API
load_dotenv()
# openai_api_key = st.secrets["OPENAI_API_KEY"]
# eleven_labs_api_key=st.secrets["ELEVENLABS_API_KEY"]
assembly_api_key=st.secrets["ASSEMBLY_API_KEY"]


##################################
###########Fonctions##############
##################################
# Fonction pour tester la clé API
def test_openai_api_key(api_key, client):
    openai.api_key = api_key
    try:
        # Essaie de faire un appel simple pour vérifier la clé
        response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello !",
            }
        ],
        model="gpt-3.5-turbo",
    )
        return True, response.choices[0].message.content
    except Exception as e:
        return False, str(e)
###########################################Fonctions de loading de fichiers et split
# Fonction pour traiter les fichiers texte
def load_txt_file(file):
    return file.read().decode("utf-8")

# Fonction pour récupérer la transcription d'un fichier MP3
def transcribe_audio(file_path):
    aai.settings.api_key = assembly_api_key
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    # st.write(transcript.text) Regarder la transcription
    return transcript.text

# Fonction pour récupérer la transcription d'une vidéo YouTube
def get_youtube_transcription(youtube_url):
    video_id = youtube_url.split("v=")[-1]  # Extraire l'ID de la vidéo à partir de l'URL
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['fr'])
    return " ".join([t['text'] for t in transcript])

# Fonction pour charger les pdfs
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)  # Chargement du fichier
    documents = loader.load()
    return documents

# Fonction pour extraire le texte des images (OCR)
def extract_text_from_image(image,file_type):
    url = 'https://api.ocr.space/parse/image'
    api_key_OCR = st.secrets['OCR_SPACE_API_KEY']

    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format=file_type)
    img_byte_arr = img_byte_arr.getvalue()
    
    # Envoyer la requête POST à l'API OCR
    response = requests.post(
        url,
        files={'filename': img_byte_arr},
        data={
            'apikey': api_key_OCR,
            'filetype': file_type.lower()
        }
    )
    
    result = response.json()
    # st.json(result) Voir les résulats en JSON
    return result['ParsedResults'][0]['ParsedText']

# Fonction pour splitter les documents
def load_and_split_documents(documents, chunk_size=100, chunk_overlap=10):
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

###########################################Fonctions de traitement des splits
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
    if len_k*0.2>1 :
        k=len_k*0.2 
    else : k=1
    docs = vectordb.max_marginal_relevance_search(query, k=k, fetch_k=len_k, lambda_mult=0.7)
    docs_content = "\n\n".join([doc.page_content for doc in docs])
    print(docs_content)
    return docs_content

# Fonction pour interroger Chroma et récupérer les documents pertinents pour le chat
# @st.cache_data(show_spinner=False)
def get_relevant_docs_chat(_vectordb, query, len_k):
    if len_k*0.2>1 :
        k=len_k*0.2 
    else : k=1
    docs = _vectordb.max_marginal_relevance_search(query, k=k, fetch_k=len_k, lambda_mult=0.3)
    docs_content = "\n\n".join([doc.page_content for doc in docs])
    print(docs_content)
    return docs_content

###########################################Fonctions de réponses
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
###########Style STREAMLIT########
##################################
# Configurer la page pour un layout large
st.set_page_config(layout="wide")

# Ajouter une description du projet en utilisant un style markdown
st.markdown("""
# Projet d'un Assistant de Synthèse

Ce projet est une application interactive qui permet aux utilisateurs de télécharger plusieurs types de fichiers (PDF, TXT, MP3, Images, vidéos YouTube) et d'extraire du texte ou des informations utiles à partir de ceux-ci.

## Comment utiliser l'application :

1. **Téléchargez un fichier** : Utilisez l'outil de téléchargement pour importer des fichiers de type PDF, TXT, MP3, images ou entrez l'URL d'une vidéo YouTube pour en extraire la transcription.
2. **Posez vos questions** :  Vous pouvez poser vos questions sur le cours
3. **Générer** : Générer un cours, une fiche de révision, ou un podcast
""", unsafe_allow_html=True)

# Input pour la clé API
openai_api_key = st.text_input("Entrez votre clé API OpenAI", type="password")

if st.button("Vérifier la clé API"):
    if openai_api_key:
        client = OpenAI(
        api_key=openai_api_key,
        )
        valid, message = test_openai_api_key(openai_api_key, client)
        if valid:
            st.session_state.api_key_valid = True
            st.success("La clé API est valide !")

        else:
            st.session_state.api_key_valid = False
            st.error("Erreur avec la clé API : " + message)
    else:
        st.warning("Veuillez entrer une clé API.")


#Injection de CSS
st.markdown("""
<style>
/* Changer le fond de la page en noir */
body {
    background-color: black;
}

/* Changer la couleur du texte en blanc */
h1, h2, h3, h4, h5, h6 {
    color: black;
}
div.stButton > button {
    background-color: #009622;  /* Couleur de fond */
    color: white;               /* Couleur du texte */
    font-weight: bold;          /* Texte en gras */
    font-size: 20px;            /* Augmenter la taille du texte */
    padding: 15px 30px;         /* Ajouter de l'espace à l'intérieur du bouton */
    border-radius: 10px;        /* Coins arrondis pour un style moderne */
}
div.stButton > button:hover {
    background-color: #007B33;  /* Couleur légèrement différente au survol */
    font-size: 22px;            /* Augmenter la taille du texte au survol */
    font-weight: bold;          /* Toujours en gras */
}
div.stDownloadButton > button {
    background-color: #FF6347;
    color: white;
}
div.stDownloadButton > button:hover {
    background-color: #000000;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# # Section d'introduction avec l'arrière-plan
# st.markdown('<div class="intro-section">', unsafe_allow_html=True)
# st.title("Bienvenue au cours")
# st.subheader("Veuillez uploader vos fichiers : PDF, TXT, MP3, YouTube")
# st.markdown('</div>', unsafe_allow_html=True)

# # Section pour afficher les types de fichiers
# st.markdown('<div class="file-icons">', unsafe_allow_html=True)

# # Ajout d'images avec une structure HTML
# st.markdown('''
# <div><img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" alt="PDF" width="80"></div>
# <div><img src="https://cdn-icons-png.flaticon.com/512/104/104647.png" alt="TXT" width="80"></div>
# <div><img src="https://cdn-icons-png.flaticon.com/512/136/136466.png" alt="MP3" width="80"></div>
# <div><img src="https://cdn-icons-png.flaticon.com/512/1384/1384060.png" alt="YouTube" width="80"></div>
# ''', unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

################################## 
###########Streamlit##############
##################################

# st.title("Ton assitant de cours")

# Uploader de plusieurs documents
uploaded_files = st.file_uploader(
    "Uploader plusieurs documents (PDF, TXT, MP3, Images)", 
    type=["pdf", "txt", "mp3", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)
# Demander un lien YouTube si nécessaire
youtube_url = st.text_input("Entrer un lien YouTube pour récupérer la transcription (optionnel)")

#################### Initialisation ############################
# Initialisation de valeurs dans le cache
if 'response' not in st.session_state:
    st.session_state.response = None 
if 'sujet' not in st.session_state:
    st.session_state.sujet = None 
if 'relevant_docs' not in st.session_state:
    st.session_state.relevant_docs = None 
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None 

###############################################################    
###################### Introduction et stockage des documents et les embeddings
###############################################################
# persist_directory = "chroma_db"
vectordb = None

generer_cours=False

if 'api_key_valid' in st.session_state and st.session_state.api_key_valid :
    if st.button("Charger et traiter les fichiers") and open:
        if uploaded_files and isinstance(uploaded_files, list) and len(uploaded_files) > 0 or youtube_url is not None:

            all_docs = []

            # Parcourir chaque fichier téléchargé
            for uploaded_file in uploaded_files:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()

                if file_extension == ".pdf":
                    # Charger et splitter chaque document PDF
                    file_path = os.path.join("temp_uploaded_file.pdf")
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Splitter les documents
                    docs = load_pdf(file_path)
                    all_docs.extend(docs)

                elif file_extension == ".txt":
                    # Charger le fichier texte
                    text = load_txt_file(uploaded_file)
                    docs = [Document(page_content=text)]
                    all_docs.extend(docs)

                elif file_extension == ".mp3":
                    # Transcrire le fichier audio MP3
                    file_path = os.path.join("temp_uploaded_audio.mp3")
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    transcription = transcribe_audio(file_path)
                    docs = [Document(page_content=transcription)]
                    print(f"Voilà la TRANSCRIPTION MP3 : {docs}")
                    all_docs.extend(docs)

                elif file_extension in [".jpg", ".jpeg", ".png"]:
                    # Charger l'image
                    image = Image.open(uploaded_file)
                    file_type = uploaded_file.name.split('.')[-1].upper()
                    # Extraire le texte de l'image avec l'OCR
                    extracted_text = extract_text_from_image(image, file_type)

                    # Ajouter le texte extrait dans les documents pour un traitement ultérieur
                    docs = [Document(page_content=extracted_text)]
                    all_docs.extend(docs)
            if youtube_url:
                st.session_state.youtube_url = youtube_url
                try:
                    transcription = get_youtube_transcription(youtube_url)
                    docs = [Document(page_content=transcription)]
                    print(f"Voilà la TRANSCRIPTION YOUTUBE : {docs}")
                    all_docs.extend(docs)
                except Exception as e:
                    st.error(f"Erreur lors de la récupération de la transcription YouTube : {e}")

            # Splitter tous les documents collectés (PDF, TXT, MP3, YouTube)
            split_docs = load_and_split_documents(all_docs)
            # Afficher le nombre de morceaux de documents chargés
            st.session_state.all_docs = split_docs

            # Stocker les embeddings dans Chroma
            if split_docs:

                vectordb = store_embeddings(split_docs)
                # embeddings = vectordb.index.reconstruct_n(0, vectordb.index.ntotal)
                # print(embeddings[:5]) 
                if vectordb is None:
                    st.error("Erreur lors de l'initialisation de vectordb.")
                else:
                    st.session_state.vectordb = vectordb
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
                st.session_state.qa = qa

            else:
                st.error("Aucun document valide trouvé pour le stockage.")
        else:
            st.error("Veuillez télécharger des fichiers.")

        st.write(f"Nombre total de morceaux de documents chargés : {len(st.session_state.all_docs)}")
else :
    st.error("Votre clé OpenAI n'est pas valide")

################################## 
################################## Interface utilisateur pour poser des questions

st.write("## Poses ta questions ici")

####################### Champ de texte pour entrer une question

user_question = st.text_input("Ton message ici")

if st.button("Envoyer"):
    # Vérification des fichiers uploadés ou de l'URL YouTube
    if (uploaded_files is not None and len(uploaded_files) > 0) or \
       ('youtube_url' in st.session_state and st.session_state['youtube_url'] is not None):

        # Vérification de la question de l'utilisateur et de la base de données vectorielle
        if user_question and st.session_state['vectordb']:

            relevant_docs = get_relevant_docs_chat(st.session_state['vectordb'], user_question, len(st.session_state['all_docs']))
            response = generate_response(st.session_state.qa, f"""
            Répondre à la question posée à l'aide des documents. Vous serez pédagogue et expliquerez de manière claire la réponse à la question.
            Soyez le plus précis possible.
            Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas.    
            {relevant_docs}

            Question : {user_question}
            Répondre en français
            """)

            st.write("### 🤖 Réponse")
            st.write(response)
        else:
            st.error("Veuillez poser une question et vous assurer que la base de données vectorielle est chargée.")
    else:
        st.error("Veuillez télécharger des fichiers ou entrer une URL YouTube valide.")

################################ barre latérale pour les boutons de téléchargement #################################################

st.sidebar.header("Téléchargement")

##################### Générer le cours
if st.sidebar.button("Générer le cours"):
    if (uploaded_files is not None and len(uploaded_files) > 0) or \
       ('youtube_url' in st.session_state and st.session_state['youtube_url'] is not None):
        #Pour dire que le cours à déjà été générer
        generer_cours=True
        #Selection de 10 documents parmis les documents dont le 1er
        docs_randoms = select_random_documents(st.session_state['all_docs'], 10)
        #Génère le sujet et la structure du plan
        sujet = generate_structure_plan(docs_randoms)
        #Prend les différentes parties du sujet
        subject_parts = sujet.split("##")
        subjet=subject_parts[1]
        #Récupère les principaux documents concernant le sujet
        relevant_docs = get_relevant_docs_cours(st.session_state['vectordb'], subjet, len(st.session_state['all_docs']))
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