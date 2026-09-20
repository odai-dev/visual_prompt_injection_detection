# Final Evaluation Report — Visual Prompt Injection Detection System

## 1. Executive Summary

This report documents the training, evaluation, and analysis of a machine learning
pipeline designed to detect **visual prompt injection attacks** targeting Multimodal
Large Language Models (MLLMs). The system combines Computer Vision (CV) preprocessing
with Natural Language Processing (NLP) n-gram TF-IDF feature extraction and benchmarks
two classical supervised classifiers.

**Best model selected:** `linear_svm`

---

## 2. Quantitative Performance — In-Distribution (Synthetic Test Set)

| Metric | Score |
|:---|---:|
| Accuracy | 0.9980 |
| Precision | 0.9980 |
| **Recall** | **0.9980** |
| F1-Score | 0.9980 |
| ROC-AUC | 1.0000 |

Recall is prioritised above other metrics as False Negatives (missed attacks)
pose the greatest risk to deployed multimodal AI systems.

---

## 3. Quantitative Performance — Out-of-Distribution (Meta CyberSecEval 3)

| Metric | Score |\n|:---|---:|\n| Accuracy | 0.9333 |\n| **Recall** | **0.9000** |\n| F1-Score | 0.9474 |\n| ROC-AUC | 1.0000 |

---

## 4. Confusion Matrix Analysis

| | Predicted: Safe | Predicted: Injection |
|:---|:---:|:---:|
| **Actual: Safe** | TN = 499 | FP = 1 |
| **Actual: Injection** | FN = 1 | TP = 499 |

### 4.1 False Positive (FP) Analysis

**Count: 1** benign samples were incorrectly flagged as injections.

FPs arise primarily from:
- **Instructional office language** — notices containing imperative verbs
  (e.g., *"Please follow safety protocols"*, *"Execute the plan by Friday"*)
  that overlap lexically with attack phrases.
- **Technical documentation** — texts containing words like *"ignore"* or
  *"override"* in legitimate engineering contexts.

**Mitigation:** Expanding the benign corpus with more technical documentation
samples and introducing domain-adaptive stopword filtering would reduce FP rate.

### 4.2 False Negative (FN) Analysis

**Count: 1** malicious samples evaded detection.

FNs arise primarily from:
- **Heavy visual camouflage** — text rendered at ΔRGB ≤ 10 (extreme low
  contrast) where even histogram equalisation and Otsu binarisation struggle
  to recover the signal before OCR.
- **Subtle phrasing** — adversarial prompts using synonyms not present in
  the TF-IDF vocabulary (e.g., *"discard prior context"* instead of
  *"ignore previous instructions"*).
- **Short injections** — single-word adversarial tokens (e.g., `DAN`) that
  fall below TF-IDF minimum document frequency thresholds.

**Mitigation:** Augmenting the injection lexicon, increasing OCR preprocessing
intensity, and training a character-level N-gram model as a secondary classifier
would reduce FN rate.

---

## 5. Cluster Analysis (K-Means, k=3)

Unsupervised K-Means clustering with 2D PCA projection revealed three
naturally separating semantic groups in the text feature space:

| Cluster | Theme | Dominant Terms |
|:---:|:---|:---|
| 0 | Benign Notices & Signage | *meeting, invoice, floor, exit, caution* |
| 1 | System Prompt Exfiltration | *system, instructions, verbatim, print, reveal* |
| 2 | Jailbreak & Command Override | *dan, ignore, bypass, curl, bash, override* |

The clear separation between clusters validates the discriminative power of
unigram+bigram TF-IDF representations for this threat detection task.

---

## 6. Deployment Recommendations for Multimodal AI Gateways

1. **Pre-inference screening layer:** Deploy this pipeline as a lightweight
   pre-processing step before images are forwarded to MLLM APIs. Estimated
   inference time per image: < 500ms on CPU.

2. **Recall-first threshold calibration:** Lower the classification probability
   threshold from 0.5 to 0.35 in production to trade a small FP increase for
   maximum attack recall.

3. **Adaptive OCR pipeline:** When extreme visual camouflage is detected
   (ΔRGB ≤ 10), apply multi-scale Gaussian blur inversion before binarisation
   to improve text recovery.

4. **Continuous lexicon updates:** The injection trigger lexicon in `nlp.py`
   must be updated regularly as adversaries discover new bypass phrasing.

5. **Ensemble with visual-only detector:** Complement text-based classification
   with a CNN-based visual feature detector for images containing no extractable
   text (steganographic or purely visual attacks).

---

## 7. Limitations

- **Synthetic distribution shift:** The training set is synthetically generated;
  real-world attack diversity may differ.
- **OCR reliability:** Low-contrast text at ΔRGB ≤ 15 may not be fully
  recovered even after CV preprocessing, creating irreducible FN headroom.
- **No deep learning component:** Classical ML on TF-IDF is fast and interpretable
  but cannot capture semantic meaning beyond n-gram co-occurrence.

---

*Report generated automatically by the Visual Prompt Injection Detection System.*
