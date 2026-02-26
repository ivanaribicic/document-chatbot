import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings 
from langchain_community.vectorstores.faiss import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import UnstructuredURLLoader
import validators
from dotenv import load_dotenv


hf_api_key = os.getenv("HF_TOKEN")

llm_model ="Featherless-Chat-Models/Mistral-7B-Instruct-v0.2:featherless-ai"
embedding_model = "hkunlp/instructor-xl"


def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    return "".join(page.extract_text() for page in pdf_reader.pages)

def get_url_text(urls):
    try:
        loader = UnstructuredURLLoader(urls=urls)
        documents = loader.load()
        return "\n\n".join([doc.page_content for doc in documents])
    except Exception as e:
        st.error(f"Error loading URLs: {str(e)}")
        return ""


def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_chunks = []
    position = 0
    
    while position < len(text):
        start_index = max(0, position - chunk_overlap)
        end_index = position + chunk_size
        chunk = text[start_index:end_index]
        text_chunks.append(chunk)
        position = end_index - chunk_overlap
    return text_chunks


def get_vectorstore(text_chunks):

    embeddings = HuggingFaceInstructEmbeddings(model_name=embedding_model)
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    print(type(vector_store))

    return vector_store


def get_conversation_chain(vectorstore):

    model_kwargs = {
    "max_new_tokens": 500, 
    "temperature": 0.1, 
    "timeout": 6000,
    "repetition_penalty": 1.0,
    }

    llm = HuggingFaceEndpoint(repo_id=llm_model, 
                              huggingfacehub_api_token = hf_api_key,
                                  **model_kwargs,
    )
    chat_model = ChatHuggingFace(llm=llm)

    print("Creating conversation chain...")
    print("Conversation chain created")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    return ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(),  
        memory=memory,
    )


def handle_userinput(user_question):
    if st.session_state.conversation is not None: 

        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            st.write(message.content)
    else:
        st.write("Please upload documents and click process")

def validate_urls(url_string):
    if not url_string:
        return []
    
    urls = [url.strip() for url in url_string.split(',') if url.strip()]
    valid_urls = []
    invalid_urls = []
    
    for url in urls:
        if validators.url(url):
            valid_urls.append(url)
        else:
            invalid_urls.append(url)
    
    return valid_urls, invalid_urls


def process_files(pdf_files, urls, st):
    all_text = ""
    
    if pdf_files:
        for file in pdf_files:
            with st.spinner(f"Processing PDF: {file.name}"):
                raw_text = get_pdf_text(file)
                all_text += raw_text + "\n\n"
                st.write(f"✅ Processed PDF: {file.name}")
    
    if urls:
        valid_urls, invalid_urls = validate_urls(urls)
        
        if invalid_urls:
            st.warning(f"Invalid URLs skipped: {', '.join(invalid_urls)}")
        
        if valid_urls:
            with st.spinner(f"Processing {len(valid_urls)} URLs..."):
                url_text = get_url_text(valid_urls)
                if url_text:
                    all_text += url_text + "\n\n"
                    for url in valid_urls:
                        st.write(f"✅ Processed URL: {url[:50]}...")
    
    if not all_text:
        st.error("No content to process. Please upload PDFs or enter valid URLs.")
        return
    
    with st.spinner("Creating text chunks..."):
        text_chunks = get_text_chunks(all_text)
        st.write(f"📝 Created {len(text_chunks)} text chunks")
    
    with st.spinner("Creating vector store..."):
        vector_store = get_vectorstore(text_chunks)
        st.write("✅ Vector store created")
    
    with st.spinner("Setting up conversation chain..."):
        st.session_state.conversation = get_conversation_chain(vector_store)
        st.write("✅ Ready to chat! You can now ask questions about your documents.")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with documents",
                       page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with documents :books:")
    
    if user_question := st.text_input("Ask a question about your documents:"):
        with st.container():
            st.markdown("---")
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("📁 Upload PDFs")
        pdf_docs = st.file_uploader(
            "Choose PDF files", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        st.subheader("🔗 Add URLs")
        st.markdown("Enter URLs (comma-separated)")
        urls_input = st.text_area(
            "URLs",
            placeholder="https://example.com/page1, https://example.com/page2",
            height=100
        )
        
        st.markdown("---")
        
        if st.button("🚀 Process Documents", type="primary"):
            if not pdf_docs and not urls_input:
                st.error("Please upload PDFs or enter URLs to process.")
            else:
                with st.spinner("Processing your documents..."):
                    process_files(pdf_docs, urls_input, st)
                st.success("Processing complete!")
        
        if st.session_state.conversation:
            st.markdown("---")
            st.subheader("📊 Status")
            st.info("✅ Documents processed")


if __name__ == '__main__':

    main()
