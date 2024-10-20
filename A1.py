import streamlit as st
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import openai
import time
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Streamlit app title
st.title("Chhatrapati Shivaji Maharaj Knowledge Graph Chatbot")

# Streamlit sidebar for configuration
st.sidebar.header("Configuration")

# Input fields for OpenAI API Key
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

# Input fields for Neo4j configuration
neo4j_uri = st.sidebar.text_input("Neo4j URI", value=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
neo4j_user = st.sidebar.text_input("Neo4j Username", value=os.getenv("NEO4J_USER", "neo4j"))
neo4j_password = st.sidebar.text_input("Neo4j Password", type="password", value=os.getenv("NEO4J_PASSWORD", ""))

# Initialize session state for chat history and embeddings
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = []  # Stores embeddings from Neo4j data
if 'documents' not in st.session_state:
    st.session_state.documents = []  # Stores documents from Neo4j
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = None  # Stores vector index for retrieval

# Function to connect to Neo4j Database and retrieve data
def get_data_from_neo4j():
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    query = """
    MATCH (n:Document) RETURN n.content AS content LIMIT 10
    """
    with driver.session() as session:
        results = session.run(query)
        documents = [record["content"] for record in results]
    driver.close()
    return documents

# Function to embed the documents using OpenAI's embedding model with retry logic
def embed_documents(documents):
    openai.api_key = openai_api_key
    embeddings = []
    retries = 3  # Number of retry attempts
    for attempt in range(retries):
        try:
            # Use the new OpenAI API for embeddings
            responses = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Ensure you're using a suitable model
                messages=[{"role": "user", "content": doc} for doc in documents],
                temperature=0,
                max_tokens=150
            )
            embeddings = [response['choices'][0]['message']['content'] for response in responses['choices']]
            return embeddings  # Return embeddings if successful
        except Exception as e:  # Catch any exception
            st.error(f"Error embedding documents (Attempt {attempt + 1}/{retries}): {e}")
            time.sleep(2)  # Wait before retrying
    return embeddings  # Return empty if all retries fail

# Function to use OpenAI LLM to generate content based on the most relevant document
def generate_output(input_text, relevant_document):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"The following information is about Chhatrapati Shivaji Maharaj:\n{relevant_document}\n\nUser Query: {input_text}\n\nAnswer:"}
            ],
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:  # Catch any exception
        st.error(f"Error generating response: {e}")  # Detailed OpenAI API error
        return None

# Display chat history in a WhatsApp-like format
def display_chat_history():
    st.header("Chat History")
    for speaker, message in st.session_state.chat_history:
        if speaker == "AI":
            st.markdown(
                f'<div style="background-color:#f1f0f0; color:#000000; padding:10px; border-radius:10px; margin-bottom:5px; width:fit-content;">{message}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background-color:#dcf8c6; color:#000000; padding:10px; border-radius:10px; margin-bottom:5px; width:fit-content; margin-left:auto;">{message}</div>',
                unsafe_allow_html=True
            )

# Main application logic
if st.sidebar.button("Connect to Neo4j and Retrieve Data"):
    if not openai_api_key or not neo4j_uri or not neo4j_user or not neo4j_password:
        st.error("Please provide all necessary credentials.")
    else:
        # Retrieve data from Neo4j
        documents = get_data_from_neo4j()
        if documents:
            st.write(f"Retrieved {len(documents)} documents from Neo4j.")
            st.write(documents)  # Display all retrieved documents for debugging
            
            # Embed documents and store in session state
            embeddings = embed_documents(documents)
            if embeddings:
                st.session_state.embeddings = embeddings
                st.session_state.documents = documents
                
                # Create a vector index from existing graph
                st.session_state.vector_index = Neo4jVector.from_existing_graph(
                    embeddings,
                    search_type="hybrid",
                    node_label="Document",
                    text_node_properties=["content"],  # Ensure this matches your Neo4j schema
                    embedding_node_property="embedding"
                )
                st.success("Documents embedded and vector index created successfully!")
            else:
                st.error("Embedding failed. Please check your API key or document input.")
        else:
            st.warning("No documents found in Neo4j.")

# Display chat history before the input field
display_chat_history()

# Input field and button to generate content
input_text = st.text_input("Enter your query:")

if st.button("Generate Response"):
    if input_text:
        if st.session_state.vector_index:
            try:
                # Embed user input query
                query_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": input_text}],
                    temperature=0,
                    max_tokens=150
                )
                query_embedding = query_response['choices'][0]['message']['content']

                # Use the vector index to retrieve relevant documents
                retriever = st.session_state.vector_index.as_retriever()
                relevant_documents = retriever.get_relevant_documents(query_embedding)

                if relevant_documents:
                    # Assume we only need the first relevant document
                    relevant_document = relevant_documents[0]
                    
                    # Use LLM to generate output
                    ai_response = generate_output(input_text, relevant_document['text'])

                    if ai_response:
                        st.session_state.chat_history.append(("User", input_text))
                        st.session_state.chat_history.append(("AI", ai_response))
                        # Refresh chat history display
                        st.experimental_rerun()
                    else:
                        st.warning("No content generated.")
                else:
                    st.warning("No relevant documents found.")
            except Exception as e:
                st.error(f"Error generating response: {e}")
        else:
            st.warning("No documents embedded. Please retrieve and embed documents first.")
    else:
        st.warning("Please enter your query.")

# Adjust page layout to align messages as per chat interface
st.markdown("""
    <style>
    .streamlit-expanderHeader {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)
