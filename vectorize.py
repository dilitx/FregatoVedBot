import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

if __name__ == "__main__":

    model = SentenceTransformer('cointegrated/rubert-tiny2')
    chunks = load_and_split("final_knowledge_base.txt")
    embeddings = model.encode(chunks)
    client = chromadb.Client()
    collection = client.create_collection(name="ifrigate_knowledge")

    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
