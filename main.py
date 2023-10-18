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

if __name__ == "__main__":
    #test()
    #data_ingestion()
    ask()