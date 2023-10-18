import os
import sys
import openai
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

from transformers import pipeline
import requests

load_dotenv()

def test():
    llm = OpenAI(temperature = 0.6)
    name = llm("Who is Srk?")
    print(name)

def data_ingestion():
    print("Starting Data Ingestion")
    loader = PyPDFDirectoryLoader("data_dir/") # check PyPDFDirectoryLoader #https://github.com/mgleavitt/my_directory_loader/blob/master/my_directory_loader.py
    print("loader ready")
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    print("Data Ingestion Done")

def ask():

    question = input("Ask anything about the data\n")

    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True,
        )
    chat_history = []
    result  = chain({"question": question,"chat_history":chat_history})
    #print(result)
    print(result['answer'])
    source_documents = result['source_documents']
    for doc in source_documents:
        print(doc.metadata['source'])
        print(doc.metadata['page'])
        print(doc.page_content)

def hugging_face():
    call_api = True # we can download the model or directly call the api from hugging face
    # https://huggingface.co/tasks/image-to-text
    print("HF")

    if call_api:
        print("Calling API")
        hugging_face_api_key = os.environ['HUGGING_FACE_API']
        API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base" # https://huggingface.co/Salesforce/blip-image-captioning-base?inference_api=true
        headers = {"Authorization": "Bearer " + hugging_face_api_key}
        path_of_image = "C:\\Users\\varun.verma\\OneDrive - Shell\\Documents\\My Pictures\\me-maa.png"

        with open(path_of_image, "rb") as f:
            data = f.read()
            response = requests.post(API_URL, headers=headers, data=data)
            print(response.json()[0]['generated_text'])

    else:
        print("Using Downloaded Module")
        url="https://avatars.githubusercontent.com/u/8264476?v=4"
        image_to_text = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base") # "https://huggingface.co/Salesforce/blip-image-captioning-base" #https://huggingface.co/Salesforce/blip-image-captioning-large?library=true
        text = image_to_text(url)[0]["generated_text"]
        print(text)


if __name__ == "__main__":
    # test()
    # data_ingestion()
    # ask()
    hugging_face()