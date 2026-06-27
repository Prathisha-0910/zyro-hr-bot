import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. PAGE SETUP & UI STYLE ---
st.set_page_config(page_title="Zyro HR Compliance Assistant", page_icon="🏢", layout="wide")
st.title("🏢 Zyro Dynamics HR Compliance Assistant")
st.markdown("Interact with our verified corporate policy documents instantly. All chats are tracked securely via LangSmith.")

# --- 2. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    selected_model = st.selectbox(
        "Select LLM Backbone:",
        ["llama-3.1-8b-instant", "llama3-70b-8192"],
        index=0
    )
    temperature = st.slider("Temperature (Creativity/Strictness):", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    st.markdown("---")
    st.info("💡 **System Rules Active:** Off-topic prompt guardrails are hardcoded to block non-HR queries automatically.")

# --- 3. ENVIRONMENT & SECRETS INGESTION ---
# Fetch keys globally from Streamlit Secrets to enforce availability
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
except Exception as e:
    st.error("Missing configuration values in Advanced Settings -> Secrets! Please check your keys.")

# --- Rebuild Retriever From Saved Files ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Look directly in the main directory for index.faiss and index.pkl
if os.path.exists("index.faiss"):
    vector_store = FAISS.load_local(".", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
else:
    st.error("Vector database files not found in repository! Please upload index.faiss and index.pkl.")
    st.stop() # Freeze app execution if database can't load

# --- 4. CONVERSATION HISTORY MEMORY INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Viewed Compliance Sources"):
                for source in msg["sources"]:
                    st.markdown(f"- {source}")

# --- 5. SYSTEM PROMPT DESIGN ---
system_prompt = (
    "You are an expert HR compliance chatbot for Zyro Dynamics Pvt. Ltd.\n"
    "Your objective is to answer employee queries using the facts found in the provided context below.\n\n"
    "⚠️ CRITICAL INSTRUCTION FOR HARD REFUSALS:\n"
    "You must ONLY output the single refusal line if the question is completely off-topic and unrelated to corporate/HR policies "
    "(such as general coding questions, cooking recipes, global politics, or other company's financial data).\n"
    "For these off-topic queries, your response must be exactly this single line and nothing else:\n"
    "I can only answer HR-related questions from Zyro Dynamics policy documents.\n\n"
    "🏢 INSTRUCTION FOR CORPORATE/HR TOPICS:\n"
    "If the query is about an HR or corporate policy topic (like travel allowances, notice periods, core hours, reimbursements) "
    "but the specific detail or exact number cannot be verified in the context, DO NOT use the hard refusal string above.\n"
    "Instead, state naturally that the specific details are not explicitly covered or mentioned in the provided policy text.\n\n"
    "Context:\n{context}"
)

# --- 6. LIVE CONVERSATION FLOW ---
if user_query := st.chat_input("Ask about leave balances, allowances, or WFH guidelines..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing compliance references..."):
            try:
                llm = ChatGroq(model=selected_model, temperature=temperature)
                prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
                
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                
                sources = []
                if "context" in response:
                    for doc in response["context"]:
                        source_name = doc.metadata.get('source', 'Zyro HR Policy Document')
                        if source_name not in sources:
                            sources.append(source_name)
                
                st.write(answer)
                if sources:
                    with st.expander("📄 Viewed Compliance Sources"):
                        for source in sources:
                            st.markdown(f"- {source}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"Execution Error. Details: {e}")
