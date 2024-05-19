from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
groq_llm_list = ['llama3-8b-8192','llama3-70b-8192','mixtral-8x7b-32768','gemma-7b-it']
groq_embed_list = ['BAAI/bge-small-en-v1.5','BAAI/bge-base-en-v1.5']
embed_dict = {'BAAI/bge-small-en-v1.5':384,'BAAI/bge-base-en-v1.5':768,'text-embedding-ada-002':1536,'text-embedding-3-small':1536}
openai_llm_list = ['gpt-3.5-turbo','gpt-4-turbo','gpt-4o']
openai_embed_list = ['text-embedding-ada-002','text-embedding-3-small']
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def create_llm(model,key):
    if model in openai_llm_list:
        llm = OpenAI(model=model,api_key=key)
    if model in groq_llm_list:
        llm = Groq(model=model,api_key=key)
    return llm

def embed(embed,key):
    if embed in embed_dict:
        if embed in openai_embed_list:
            embedding = OpenAIEmbedding(model=embed,api_key=key)
            d = embed_dict[embed]
        else:
            embedding = HuggingFaceEmbedding(model_name=embed)
            d = embed_dict[embed]
    return embedding,d

def completion(llm,query):
    return llm.complete(query)

