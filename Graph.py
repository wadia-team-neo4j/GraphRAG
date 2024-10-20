#This python code will create an interactive Chatbot to talk to documents.
import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from streamlit_option_menu import option_menu
#Let's integrate langsmith
from dotenv import load_dotenv, find_dotenv
from langsmith import Client
#Import related to KnowledgeGraph
from py2neo import Graph
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

from KnowledgeGrpah_Neo4j import RAG_Graph


#load langsmith API key
load_dotenv(find_dotenv())
os.environ["LANGSMITH_API_KEY"] = str(os.getenv("LANGSMITH_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

#Initialize the Client

#Create temporary folder location for document storage
TMP_DIR = Path(__file__).resolve().parent.parent.joinpath('data', 'tmp')

# Ensure TMP_DIR exists
TMP_DIR.mkdir(parents=True, exist_ok=True)


header = st.container()

def streamlit_ui():

    with st.sidebar:
        choice = option_menu('Navigation',['RAG with Neo4J'])

    if choice == 'Home':
        st.title("RAG tutorial using multiple techniques")

    elif choice == 'Simple RAG':
        with header:
            st.title('Simple RAG with vector')  
            st.write("""This is a simple RAG process where user will upload a document then the document
                     will go through RecursiveCharacterSplitter and embedd in FAISS DB""")
            
            source_docs = st.file_uploader(label ="Upload a document", type=['pdf'], accept_multiple_files=True)
            if not source_docs:
                st.warning('Please upload a document')
           
    
    elif choice == 'RAG with Neo4J':
        with header:
            st.title('Chat with Me On Chhatrapati Shivaji Maharaj')
            
            RAG_Neo4j()
           
            


def RAG_Neo4j():
    rag_graph = RAG_Graph()
    choice = option_menu('Options',["Upload document",'Graph(Skip document upload)'])
    
    if choice == 'Upload document':
        source_docs = st.file_uploader(label="Upload document", type=['docx'],accept_multiple_files=True)
        if not source_docs:
            st.warning("Please upload a document")
        else:
            rag_graph.create_graph(source_docs,TMP_DIR)
    else:
        show_graph()

    
    st.session_state.messages1 = []
    #Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages1 =[]
    
    #Display chat messages from history on app rerun
    for message in st.session_state.messages1:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    #React to user input
    if prompt1 := st.chat_input("Ask question to document assistant"):
        #Display user message in chat message container
        st.chat_message("user").markdown(prompt1)
        #Add user message to chat history
        st.session_state.messages1.append({"role":"user","context":prompt1})

        response1 = f"Echo: {prompt1}"
        #Display assistant response in chat message container
        response1 = rag_graph.ask_question_chain(prompt1)

        with st.chat_message("assistant"):
            st.markdown(response1)

        st.session_state.messages1.append({'role':"assistant", "content":response1})

def RAG_Neo4j1(docs,TMP_DIR):
    rag_graph = RAG_Graph()
    #rag_graph.create_graph(docs,TMP_DIR)
    show_graph()

    chat_history1 = []
    st.session_state.messages1 = []
    #Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages1 =[]
    
    #Display chat messages from history on app rerun
    for message in st.session_state.messages1:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    #React to user input
    if prompt1 := st.chat_input("Ask question to document assistant"):
        #Display user message in chat message container
        st.chat_message("user").markdown(prompt1)
        #Add user message to chat history
        st.session_state.messages1.append({"role":"user","context":prompt1})

        response1 = f"Echo: {prompt1}"
        #Display assistant response in chat message container
        #response = qa_chain({'question':prompt,'chat_history':chat_history})
        response1 = rag_graph.ask_question_chain(prompt1)

        with st.chat_message("assistant"):
            st.markdown(response1['answer'])

        st.session_state.messages1.append({'role':"assistant", "content":response1})
        chat_history1.append({prompt1,response1['answer']})



def show_graph():
    st.title("Neo4j Graph Visualization")

    #user input for Neo4J credential
    uri = st.text_input("Neo4j URI", "bolt://localhost:7687")
    user = st.text_input("Neo4j username", "neo4j")
    password = st.text_input("Neo4j password", type="password")

    #Create a load graph button
    if st.button("Load Graph"):
        try:
            data = get_graph_data(uri,user,password)
            G = create_networkx_graph(data)
            visualize_graph(G)

            HtmlFile = open("graph.html", "r", encoding="utf-8")
            source_code = HtmlFile.read()
            components.html(source_code,height=600, scrolling=True)
        except Exception as e:
            st.error(f"Error loading page:  {e}")

def get_graph_data(uri,user,password):
    graph = Graph(uri,auth=(user,password))
    query = """
    MATCH (n)-[r]->(m)
    RETURN n,r,m
    LIMIT 100
    """

    data = graph.run(query).data()
    return data

def create_networkx_graph(data):
    G = nx.DiGraph()
    for record in data:
        n = record['n']
        m = record['m']
        r = record['r']
        G.add_node(n['id'], label=n['name'])
        G.add_node(m['id'], label=m['name'])
        G.add_edge(n['id'], m['id'], label=r['type'])
    return G

def visualize_graph(G):
    net = Network(notebook=True)
    net.from_nx(G)
    net.show("graph.html")


streamlit_ui()




