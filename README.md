# Visual Prompt Injection Detection Pipeline

A modular, lightweight Machine Learning pipeline designed to detect visual prompt injection (typographic jailbreaks, instruction overrides, and prompt extraction attacks) embedded in images targeting Multimodal Large Language Models (MLLMs). 

The system acts as a security firewall, parsing images via Computer Vision/OCR, analyzing textual payload intent using TF-IDF NLP vectorization, and blocking malicious inputs.

## 🚀 Key Features
* **Hybrid CV & NLP Stack**: Defeats advanced typographic camouflage (Delta-RGB $\le$ 25) using Otsu Adaptive Binarization before extracting text.
* **Massive Dataset**: Auto-generates a 5,000-image dataset (Signage, Memos, Receipts) blended with real-world adversarial jailbreaks (AdvBench).
* **High-Accuracy Classification**: Linear Support Vector Machine (SVM) pipeline that prioritizes Recall to prevent zero-day adversarial payloads from leaking through to the VLM.
* **Real-World Benchmarked**: Evaluated securely against Meta's *CyberSecEval 3* Out-of-Distribution (OOD) benchmark.

## 📈 Performance Metrics

| Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|----------|-----------|--------|----------|---------|
| **Synthetic Test Set (In-Distribution)** | 99.8% | 99.8% | 99.8% | 99.8% | 1.000 |
| **CyberSecEval 3 (Out-of-Distribution)** | 93.3% | 100% | 90.0% | 94.7% | 1.000 |

## ⚙️ Installation
1. Clone the repository and navigate to the root directory.
2. Ensure you have Python 3.10+ installed.
3. Install the dependencies:
```bash
pip install -r requirements.txt
```
*(Note: Requires `tesseract-ocr` installed on the host machine)*

## 💻 Usage

The primary entry point is the CLI tool `src/main.py` which provides a rich terminal interface.

### Train the Pipeline
To regenerate the 5,000-image dataset, extract all text via OCR, train the NLP classifiers, and generate visualization figures:
```bash
python src/main.py --train
```

### Live Prediction
To pass an image through the firewall and retrieve a verdict (`SAFE` vs `MALICIOUS`):
```bash
python src/main.py --predict path/to/your/image.jpg
```

### Benchmark Evaluation
To re-evaluate the serialized models (`models/best_supervised_model.joblib`) against the test set and CyberSecEval 3:
```bash
python src/main.py --evaluate
```

## 🏗️ Architecture
For deep architectural details, dataset split metrics, and interface contracts, refer to [PROJECT.md](PROJECT.md). 

For the academic thesis/report associated with this project, refer to `Visual_Prompt_Injection_Detection_Report.pdf`.
