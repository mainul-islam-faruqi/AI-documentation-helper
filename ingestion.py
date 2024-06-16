import os

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from consts import INDEX_NAME

# from langchain import VectorDBQA, OpenAI

load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

def ingest_docs()->None:
    loader = UnstructuredHTMLLoader("./langchain-docs/index.html")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])

    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    # for doc in documents:
    #     old_path = doc.metadata["source"]
    #     new_url = old_path.replace("langchain-docs", "https:/")
    #     doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)

    print("******* Added to Pinecone Vectorstore")
    

if __name__ == '__main__':
    ingest_docs()