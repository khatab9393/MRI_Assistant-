"""
Load the medical knowledge base (text only, NO images) into the local database.
Edit knowledge.json to add or correct conditions, then run:
    python ingest_knowledge.py
"""
import json
import ollama, chromadb

EMBED_MODEL = "nomic-embed-text"

client = chromadb.PersistentClient(path="./knowledge_db")
try:
    client.delete_collection("knowledge")
except Exception:
    pass
col = client.create_collection("knowledge", metadata={"hnsw:space": "cosine"})

items = json.load(open("knowledge.json", encoding="utf-8"))
texts = [f"{i['name']}. MRI findings: {i['mri_findings']}" for i in items]
embs = ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]

col.add(
    ids=[str(n) for n in range(len(items))],
    embeddings=embs,
    documents=[
        f"Condition: {i['name']}\nMRI findings: {i['mri_findings']}\n"
        f"Differential: {i['differential']}\nNext steps: {i['next_steps']}"
        for i in items
    ],
)
print("Loaded", col.count(), "conditions")
