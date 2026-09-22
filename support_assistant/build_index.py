from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DB = ROOT / "chroma_db"

def main():
    client = chromadb.PersistentClient(path=str(DB))
    collection = client.get_or_create_collection(
        name="zepto_policies",
        metadata={"hnsw:space": "cosine"}
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    ids, documents, metadatas = [], [], []
    for path in sorted(DOCS.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        ids.append(path.stem)
        documents.append(text)
        metadatas.append({"source": path.name})

    embeddings = model.encode(documents, normalize_embeddings=True).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Indexed {len(ids)} documents.")
    print("Collection:", collection.name)

if __name__ == "__main__":
    main()
