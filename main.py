import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType

# ----------------------------
# Streamlit UI Setup
# ----------------------------
st.set_page_config(page_title="🩺 Medical AI Chatbot", page_icon="💊")
st.title("🧠 AI Medical Advisor")
st.markdown("Ask any health-related question. The bot uses PDFs + Web + LLaMA 3 💬")

# ----------------------------
# Load FAISS Vector DB
# ----------------------------
@st.cache_resource
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.load_local("medical_vector_db", embeddings, allow_dangerous_deserialization=True)
    return vectordb, embeddings

vectordb, embeddings = load_vectordb()

# ----------------------------
# Setup Tools: PDFs + Web
# ----------------------------
web_search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="VectorMedicalSearch",
        func=lambda q: "\n\n".join([doc.page_content for doc in vectordb.similarity_search(q, k=3)]),
        description="Fetches medical context from uploaded PDFs"
    ),
    Tool(
        name="WebSearch",
        func=web_search.run,
        description="Fetches latest medical info from web"
    )
]

# ----------------------------
# LLM Configuration (Groq + LLaMA3)
# ----------------------------
llm = ChatGroq(
    temperature=0.3,
    model_name="llama3-70b-8192",
    groq_api_key="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"  # Replace with your actual key
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# ----------------------------
# Chat Input / Output Handling
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Describe your symptoms or ask a medical question...")

# Show chat history only if no new query
if not user_query:
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

if user_query:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append(("user", user_query))

    # Add instruction to make it detailed with bullet points
    styled_query = (
        f"As a medical advisor, please give a detailed paragraph-based response to the following question. "
        f"If relevant, include bullet points for clarity (e.g., symptoms, causes, precautions, treatments). "
        f"Here is the question:\n\n{user_query}"
    )

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.run(styled_query)
            st.markdown(response)
    st.session_state.chat_history.append(("assistant", response))
