# MRI Assistant

A local, private chatbot that helps a physician review brain MRI scans. The doctor uploads a scan and gets a suggested diagnosis with real probabilities, typical MRI features, differential diagnoses and next steps, written for a clinician. Everything runs on your own computer: no cloud, no data sent anywhere.

> **Research prototype. Not a medical device.** It is not validated for clinical use. A qualified physician must verify every result.

## How it works

```
MRI image -> ResNet18 classifier -> probabilities per condition
                                  -> knowledge base lookup (text only)
                                  -> Qwen2.5 (via Ollama) writes the answer for the doctor
```

- **ResNet18** (PyTorch) is fine-tuned on labeled brain MRI images and outputs a probability for each class.
- **Knowledge base** (`knowledge.json`, searched with ChromaDB and `nomic-embed-text`) holds text descriptions of each condition: MRI features, differential diagnosis, next steps. It stores no images.
- **Qwen2.5** runs in Ollama and turns the classifier output and the knowledge into a clinician-facing answer. It is told to use the classifier's percentages and not to invent its own.
- **FastAPI** serves a simple black-and-white web page (`index.html`).

## Features

- Local and private: the uploaded scan is processed in memory and not saved by the app
- Probabilities for every class, with a low-confidence warning below 60%
- Drag and drop, paste, or attach a scan; click a scan to enlarge it
- Follow-up buttons (differentials, next steps, short summary) and copy-answer
- Editable knowledge base: change `knowledge.json` and re-run the ingest step

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com)
- An NVIDIA GPU is recommended for training (CPU works, but slowly)

## Setup

```bash
# 1. Models
ollama pull qwen2.5
ollama pull nomic-embed-text

# 2. Python environment
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
# For an NVIDIA GPU, install the CUDA build of PyTorch from https://pytorch.org

# 3. Train the classifier (once)
python train.py dataset/Training

# 4. Build the knowledge base
python ingest_knowledge.py

# 5. Run
python server.py
```

Open http://127.0.0.1:8000

### Dataset

Training expects one folder per class:

```
dataset/Training/
  glioma/  meningioma/  notumor/  pituitary/
```

The public Brain Tumor MRI Dataset on Kaggle (by Masoud Nickparvar) uses this layout. Check its license and terms before use. For real use, train on your own anonymized, confirmed cases. The dataset and the trained `model.pt` are not included in this repository.

## Project structure

| File | Purpose |
|---|---|
| `train.py` | Fine-tunes ResNet18, saves `model.pt` |
| `ingest_knowledge.py` | Loads `knowledge.json` into the local vector database |
| `knowledge.json` | Condition descriptions (starter set of 9 brain conditions) |
| `server.py` | FastAPI backend: classifier, knowledge lookup, Qwen |
| `index.html` | Web interface |
| `requirements.txt` | Python packages |

## Limitations

- The classifier only knows its training classes. Any other condition (stroke, MS, spine, etc.) is forced into one of them, sometimes with high confidence.
- Softmax probabilities can be overconfident. A high percentage is not a guarantee.
- Public datasets are cleaner than hospital scans, so accuracy on real scans may be lower. Evaluate on held-out images before trusting results.
- The starter `knowledge.json` was written from general knowledge and must be reviewed by a specialist.

## Privacy

Never commit patient images, reports or identifiers. Remove all identifying information (including DICOM metadata) before using any scan, and follow your institution's and country's rules.
