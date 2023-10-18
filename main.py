import os
import sys
import openai
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator

from dotenv import load_dotenv

load_dotenv()

def test():
    llm = OpenAI(temperature = 0.6)
    name = llm("Who is Srk?")
    print(name)

def data_ingestion():
    print("Starting Data Ingestion")
    loader = DirectoryLoader("data_dir/")
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    print("Data Ingestion Done")

if __name__ == "__main__":
    #test()
    data_ingestion()