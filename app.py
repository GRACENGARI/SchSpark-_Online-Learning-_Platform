import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from dotenv import load_dotenv
import os
import time

# Load environment variables (e.g., API keys)
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Ensure the API key is set
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set. Check your .env file.")

# Set page config for wider layout
st.set_page_config(
    page_title="SchSpark Chat",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for chat interface
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .message {
    color: #fff;
    font-size: 1rem;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.app-title {
    text-align: center;
    color: #4b86b4;
    font-size: 2.5rem;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='app-title'>SchSpark Online Learning Platform</h1>", unsafe_allow_html=True)

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to handle query submission
def handle_query(query):
    if query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Create RAG chain and get response
        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # Invoke the RAG chain and get the response
        response = rag_chain.invoke({"input": query})
        answer = response["answer"]
        
        # Store conversation in langchain memory
        memory.save_context({"input": query}, {"output": answer})
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Load the document
@st.cache_resource
def load_and_process_document():
    loader = PyPDFLoader("C:/Users/grace/Desktop/VIN/langachain-cbc-cahatbot/FAQs SchSpark.pdf")
    data = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)
    
    # Create embeddings using GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Store the document embeddings in FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    return vectorstore

vectorstore = load_and_process_document()

# Set up a retriever for similarity search
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize the Google Gemini model for the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Add Memory to retain conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define system prompt for the chatbot
system_prompt = (
    "You are an assistant for the SchSpark Online Learning Platform. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Provide thorough but concise answers."
    "\n\n"
    "{context}"
)

# Create the chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Define a tool for document retrieval
def retrieve_documents(query):
    docs = retriever.get_relevant_documents(query)
    return docs

retrieval_tool = Tool(
    name="Document Retrieval",
    func=retrieve_documents,
    description="Retrieves relevant documents based on the user's query"
)

# Create an agent that can use retrieval as a tool
agent = initialize_agent(
    tools=[retrieval_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.container():
            st.markdown(f"""
            <div class="chat-message user">
                <div class="message">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown(f"""
            <div class="chat-message bot">
                <div class="message">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

# Chat input
with st.container():
    # Create a form for the input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask about SchSpark:", key="user_input", placeholder="Type your question here...")
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            # Process the query
            with st.spinner("Thinking..."):
                handle_query(user_input)
                # Force a rerun to update the chat display
                st.rerun()

# Add a sidebar with information
with st.sidebar:
    st.title("About SchSpark")
    st.info(
        """
        This chatbot uses RAG (Retrieval-Augmented Generation) to provide 
        accurate information about SchSpark Online Learning Platform using 
        the FAQ document as its knowledge base.
        """
    )
    
    # Add option to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()