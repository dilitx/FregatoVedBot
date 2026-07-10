import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

model = SentenceTransformer('cointegrated/rubert-tiny2')

with open("final_knowledge_base.txt", "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
print(f"Получилось {len(chunks)} чанков")

embeddings = model.encode(chunks)

client = chromadb.Client()

questions = [
    "Сколько стоит интернет-магазин?",
    "Что такое Флюгер-Продажи?",
    "Склько стит саит?",
    "Что за пушка для продаж?",
    "Какая погода в Москве?"
]

metrics = ["cosine", "l2", "ip"]

print("Сравнение метрик")

for metric in metrics:
    print(f"\n Метрика: {metric}")

    collection = client.create_collection(
        name=f"test_{metric}",
        metadata={"hnsw:space": metric}
    )

    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    for q in questions:
        query_emb = model.encode([q]).tolist()

        results = collection.query(
            query_embeddings=query_emb,
            n_results=3
        )

        print(f"\nВопрос: {q}")

        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            dist = results['distances'][0][i]
            doc_clean = doc.replace('\n', ' ')
            print(f"  {i + 1}. dist={dist:.4f}")
            print(f"     Текст: {doc_clean}")

    client.delete_collection(f"test_{metric}")
