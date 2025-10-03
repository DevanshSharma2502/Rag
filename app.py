import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import os

# ----------------------------
# Helper functions
# ----------------------------
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
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature":0.5, "max_length":512},
        huggingfacehub_api_token=hf_key
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"🧑 You: {message.content}")
        else:
            st.write(f"🤖 Bot: {message.content}")

# ----------------------------
# Main app
# ----------------------------
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    st.title("📚 Chat with Multiple PDFs")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("💬 Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("📂 Upload your PDFs")
        pdf_docs = st.file_uploader("Upload PDFs and click Process", accept_multiple_files=True)
        if st.button("🚀 Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
            else:
                st.warning("Please upload PDFs.")

if __name__ == '__main__':
    main()
