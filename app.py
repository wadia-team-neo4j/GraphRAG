import os
import streamlit as st
import tempfile
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from streamlit_option_menu import option_menu
from dotenv import load_dotenv, find_dotenv
from langsmith import Client
from py2neo import Graph
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# Import your custom Neo4j-based knowledge graph class
from KnowledgeGrpah_Neo4j import RAG_Graph

# Load LangSmith API key and configure environment variables
load_dotenv(find_dotenv())
os.environ["LANGSMITH_API_KEY"] = str(os.getenv("LANGSMITH_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Apply custom CSS styling
def apply_custom_css():
    st.markdown(
        """
        <style>
        /* Style the chat container */
        .stChatMessage {
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        
        /* Different background for user and assistant messages */
        .stChatMessage.user {
            background-color: #d1e7dd;
            text-align: left;
        }
        .stChatMessage.assistant {
            background-color: #f8d7da;
            text-align: left;
        }

        /* Customize chat input field */
        div[data-baseweb="input"] > div {
            border-radius: 10px;
            background-color: #f1f3f4;
        }

        /* Adjust width of the Streamlit option menu */
        .css-18e3th9 {
            padding: 1rem;
        }
        
        /* Page title style */
        h1 {
            color: #4b6584;
            text-align: center;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background-color: #4b6584;
            border-radius: 10px;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

def RAG_Neo4j():
    # Initialize the RAG Graph class instance
    rag_graph = RAG_Graph()

    # Set up chat history within session state
    if "messages1" not in st.session_state:
        st.session_state.messages1 = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages1:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capture user input from chat interface
    if prompt1 := st.chat_input("Ask a question to the document assistant"):
        # Display the user's message
        st.chat_message("user").markdown(prompt1)
        # Add user's message to chat history
        st.session_state.messages1.append({"role": "user", "content": prompt1})

        # Get the response from the Neo4j-based RAG system
        response1 = rag_graph.ask_question_chain(prompt1)

        # Display the assistant's response
        with st.chat_message("assistant"):
            st.markdown(response1)

        # Add the assistant's response to chat history
        st.session_state.messages1.append({"role": "assistant", "content": response1})

def home():
    # Adjust the path to match your project structure
    image_path = Path("Ch. S M project/data/GRchatbot.jpg")

    if image_path.exists():
        st.image(image_path, caption="Chhatrapati Shivaji Maharaj", use_column_width=True)
    else:
        st.error("Image not found. Please check the file path.")

# Streamlit app main entry point
def main():
    apply_custom_css()  # Apply CSS styles

    st.title("Chat With Me On Chhatrapati Shivaji Maharaj")

    # Display an option menu (if you need navigation)
    option = option_menu(
        menu_title="Main Menu",
        options=["Home", "Chat"],
        icons=["house", "chat"],
        menu_icon="cast",
        default_index=1,
    )

    if option == "Home":
        home()
    elif option == "Chat":
        RAG_Neo4j()

if __name__ == "__main__":
    main()
