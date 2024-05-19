import os
# from groq import Groq
import streamlit as st
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

from utils import *
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader, ServiceContext, StorageContext, VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import Document
from llama_index.readers.file import PDFReader

import faiss
from processing import *
from chat import *

if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None
def save_uploaded_file(data):
    dir_path = os.path.join(os.path.abspath('.'),'temp_file_store')
    path = os.path.join(os.path.abspath('.'),'temp_file_store',data.name)
    with open(path,"wb") as f:
        f.write(data.getbuffer())
    return dir_path

def main():
    with st.sidebar:
        st.markdown("##### Step 1:")
        platform = st.selectbox('Select Platform',options=['Groq','OpenAI'])
        if platform is not None:
            if platform=='Groq':
                chat_model = st.selectbox('Select Model',options=groq_llm_list)
                embedding_model = st.selectbox('Select Model',options=groq_embed_list)
            else: #openai
                chat_model = st.selectbox('Select Model',options=openai_llm_list)
                #st.text_input("OpenAI Chat Model Name")
                embedding_model = st.selectbox('Select Model',options=openai_embed_list)
                #st.text_input("OpenAI Embedding Name")
        key = st.text_input("API KEY",key="api_key",type="password")
        data = st.file_uploader("Upload your PDF Chapter here!",type=['pdf'],help='Only PDF files are supported')
        submit = st.button("Get Started!")
    if submit:
        if platform and chat_model and embedding_model and key and data:
            with st.spinner("Processing your file, feel free to grab a coffee while waiting :coffee:"):
                st.session_state.docs = data
                st.session_state.docs_bytes = data.read()
                st.session_state.processed = submit

                # SAVE PDF FILE
                filepath = save_uploaded_file(data)
                # INITIALIZE LLAMAINDEX VECTOR STORE
                llm = create_llm(chat_model,key)
                embedding_model,d = embed(embedding_model,key)
                service_context = ServiceContext.from_defaults(llm=llm,embed_model = embedding_model)
                faiss_index = faiss.IndexFlatL2(d)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                # READ THE PDF FILE
                docs = SimpleDirectoryReader(filepath).load_data()
                # CHUNK DOCUMENT
                docs_text = "\n\n".join([d.get_content() for d in docs])
                text = [Document(text=docs_text)]
                # INITIALIZE THE VECTOR INDEX FROM THE PREDEFINED VECTOR STORE WE CREATED EARLIER
                vector_index = VectorStoreIndex.from_documents(docs,storage_context=storage_context,service_context=service_context)
                # SETTING UP OUR QUERY ENGINE TO START ASKING QUESTIONS
                st.session_state.query_engine = vector_index.as_query_engine(streaming=True)
                with st.chat_message("user"):
                    st.write("Hi! Ask me any questions about your document. I will try my best to find the most relevant insights for you :smile:")

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for message in st.session_state.messages:
                    with st.chat_message(message['role']):
                        st.markdown(message['content'])
    else:
        if not platform:
            st.error("Please select a platform")
        elif not chat_model:
            st.error("Please select an LLM model")
        elif not embedding_model:
            st.error("Please select an Embedding model")
        elif not key:
            st.error("Please fill in your API key!")
        elif not data:
            st.error("Please upload a PDF document!")
        else:
            st.success("Click the 'Get Started!' button to start chatting with your document!")
    prompt = st.chat_input("Ask me a question!",disabled=not st.session_state.query_engine)
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})
        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            full_response = ""
            response = st.session_state.query_engine.query(prompt)
            for chunk in response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({'role':'assistant','content':full_response})
if __name__=="__main__":
    main()