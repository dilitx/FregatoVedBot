from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()


def split_into_chunks(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)
    return chunks



if __name__ == "__main__":
    raw_text = load_text("final_knowledge_base.txt")

    print(f"Размер текста: {len(raw_text)} символов")
    print("Разбиваем на чанки...")

    chunks = split_into_chunks(raw_text)

    print(f"\nГотово! Текст разбит на {len(chunks)} кусочков.\n")



    for i in range(min(5, len(chunks))):
        print(f"\n--- ЧАНК №{i + 1} (символов: {len(chunks[i])}) ---")
        print(chunks[i])
        print()


    with open("chunks_preview.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"=== ЧАНК №{i + 1} ===\n")
            f.write(chunk)
            f.write("\n\n")

    print(f"\nВсе чанки сохранены в файл 'chunks_preview.txt'")
