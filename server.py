"""
Backend for the custom MRI Assistant page.
  ResNet (model.pt) + knowledge DB + qwen2.5, served locally with FastAPI.

Run:  python server.py     then open http://127.0.0.1:8000
"""
import io
import json
import os
import ollama
import chromadb
import uvicorn
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from torchvision import models, transforms

TEXT_MODEL = "qwen2.5"
EMBED_MODEL = "nomic-embed-text"
LOW_CONFIDENCE = 0.60

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier, CLASSES = None, []
if os.path.exists("model.pt"):
    ckpt = torch.load("model.pt", map_location=device)
    CLASSES = ckpt["classes"]
    classifier = models.resnet18()
    classifier.fc = nn.Linear(classifier.fc.in_features, len(CLASSES))
    classifier.load_state_dict(ckpt["state"])
    classifier.to(device).eval()

tf = transforms.Compose([
    transforms.Grayscale(3), transforms.Resize(
        (224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

col = chromadb.PersistentClient(path="./knowledge_db").get_or_create_collection(
    "knowledge", metadata={"hnsw:space": "cosine"}
)

SYSTEM = """You are a clinical decision-support assistant speaking directly to a physician
(an MRI specialist). Never address a patient. Use professional medical language.

When a classifier result is provided, answer in this format:
1. Most likely diagnosis: <name> - confidence: <X>%  (use the classifier's percentage exactly)
2. Other possibilities with their percentages (from the classifier)
3. Typical MRI features of the top diagnosis (from the reference knowledge)
4. Differential diagnoses to consider
5. Suggested next steps
6. Limitations

Rules:
- You are an aid, not a replacement for the physician's reading. Never claim certainty.
- Do NOT invent your own percentages. Use only the classifier's numbers.
- The classifier only knows the listed classes. If confidence is low, or the case may be
  outside those classes, say so clearly and recommend manual review.
- Say that you cannot see the image yourself; you rely on the classifier output.
- The reader is the physician, never a patient and never a layperson. Address them as
  "Doctor" and write like a colleague giving a second opinion.
- Do not give the reader instructions or commands such as "please provide". If clinical
  history or sequence type is missing, state it as a note, for example: "Clinical history
  and sequence type were not provided; they would help refine this assessment."
- Do not repeat that you are an AI or that you cannot see the image more than once."""


def classify(img):
    with torch.no_grad():
        probs = torch.softmax(classifier(
            tf(img).unsqueeze(0).to(device)), 1)[0].cpu()
    return sorted(zip(CLASSES, probs.tolist()), key=lambda t: -t[1])


def lookup(label, k=3):
    if col.count() == 0:
        return "Knowledge base is empty (run ingest_knowledge.py)."
    emb = ollama.embed(model=EMBED_MODEL, input=label.replace("_", " "))[
        "embeddings"][0]
    r = col.query(query_embeddings=[emb], n_results=min(k, col.count()))
    return "\n\n".join(r["documents"][0])


app = FastAPI()


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/analyze")
def analyze(message: str = Form(""), history: str = Form("[]"),
            image: UploadFile | None = File(None)):
    try:
        hist = json.loads(history)
        content, results = message, []

        if image is not None and image.filename:
            if classifier is None:
                return JSONResponse({"error": "No trained model found. Run train.py first."}, 400)
            img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
            top = classify(img)
            results = [{"label": n, "p": p} for n, p in top]
            flag = "LOW CONFIDENCE - recommend manual review.\n" if top[0][1] < LOW_CONFIDENCE else ""
            listing = "\n".join(f"- {n}: {p * 100:.1f}%" for n, p in top)
            content = (
                f"Doctor's message: {message or '(none)'}\n\n"
                f"[Classifier output (ResNet, trained classes: {', '.join(CLASSES)})]\n{flag}{listing}\n\n"
                f"[Reference knowledge for the top class]\n{lookup(top[0][0])}\n\n"
                "Write the structured answer for the doctor."
            )

        msgs = [{"role": "system", "content": SYSTEM}] + \
            hist + [{"role": "user", "content": content}]
        answer = ollama.chat(model=TEXT_MODEL, messages=msgs)[
            "message"]["content"]
        return {"answer": answer, "results": results}
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, 500)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
