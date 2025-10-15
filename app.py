import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun


# ---------------- Helper Functions ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    google_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_key,
        temperature=0.5
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=(
            "You are a helpful assistant having a conversation with a human.\n"
            "Use the following context and chat history to answer the current question.\n"
            "If the user shares personal info (like their name), remember and use it in follow-ups.\n\n"
            "Context:\n{context}\n\n"
            "Conversation History:\n{chat_history}\n\n"
            "Question: {question}\n\n"
            "Answer helpfully:"
        )
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    return conversation_chain



def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"🧑 You: {message.content}")
        else:
            st.write(f"🤖 Bot: {message.content}")


def get_web_llm():
    google_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", google_api_key=google_key, temperature=0.7
    )
    return llm


def web_search(question):
    search_tool = DuckDuckGoSearchRun()
    memory = st.session_state.web_memory

    history_text = ""
    for msg in memory.buffer:
        role = "User" if msg.type == "human" else "Assistant"
        history_text += f"{role}: {msg.content}\n"

    st.markdown("🌐 Searching the web...")
    search_results = search_tool.run(question)

    context = f"Conversation so far:\n{history_text}\n\nSearch Results:\n{search_results}\n\n"
    prompt = (
        f"You are a helpful assistant. Use both the previous conversation and "
        f"the web search results below to answer the user's question clearly and concisely.\n\n"
        f"{context}\n"
        f"User question: {question}\n\n"
        f"Answer:"
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.6
    )
    response = llm.invoke(prompt)

    memory.save_context({"input": question}, {"output": response.content})

    st.write("🌐 **Web Chat:**")
    for msg in memory.buffer:
        if msg.type == "human":
            st.write(f"🧑 You: {msg.content}")
        elif msg.type == "ai":
            st.write(f"🤖 Bot: {msg.content}")



def summarize_pdfs():
    try:
        if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
            st.warning("Please process PDFs first.")
            return

        google_key = os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", google_api_key=google_key, temperature=0.3
        )

        docs = st.session_state.vectorstore.similarity_search("", k=40)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = "Summarize the following text into 5 concise bullet points:\n\n" + context

        summary = llm.invoke(prompt)
        st.write("📝 **Summary:**")
        st.write(summary.content)

    except Exception as e:
        st.error(f"Error while summarizing: {e}")


# ---------------- Main App ----------------
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs & Web", page_icon="📚")
    st.title("📚 Chat with Multiple PDFs + 🌐 Web Search")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "web_llm" not in st.session_state:
        st.session_state.web_llm = get_web_llm()
    if "web_memory" not in st.session_state:
        st.session_state.web_memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    # === Main Input Section ===
    user_question = st.text_input("💬 Ask a question:")

    # Buttons in one row
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        search_pdfs = st.button("🔎 Search PDFs")
    with col2:
        web_search_btn = st.button("🌐 Web Search")
    with col3:
        summarize_btn = st.button("📝 Summarize PDFs")

    # === Results appear right below buttons ===
    if search_pdfs:
        if user_question:
            if st.session_state.conversation:
                handle_userinput(user_question)
            else:
                st.warning("Please upload and process PDFs first.")
        else:
            st.warning("Please enter a question.")

    elif web_search_btn:
        if user_question:
            web_search(user_question)
        else:
            st.warning("Please enter a question.")

    elif summarize_btn:
        summarize_pdfs()

    # === Sidebar for PDF Upload ===
    with st.sidebar:
        st.subheader("📂 Upload your PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDFs and click Process", accept_multiple_files=True
        )
        if st.button("🚀 Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("✅ PDFs processed successfully!")
            else:
                st.warning("Please upload PDFs.")


if __name__ == "__main__":
    main()
