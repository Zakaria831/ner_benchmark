# French Named Entity Recognition Benchmark (WiNER)

This repository implements a **comparative benchmark of Named Entity Recognition (NER) methods for French**, evaluated on the **WiNER (Wikinews for Named Entity Recognition)** corpus.

The benchmark focuses on **off-the-shelf models and tools** and aims to analyze **which methods perform best for which entity types**, with both **research and industrial perspectives**.

---

## Objectives

- Benchmark multiple **NER paradigms** on a common French dataset
- Compare models using **entity-level exact match evaluation**
- Analyze **per-entity-type performance**
- Study **model biases, coverage, and limitations**
- Provide insights relevant for **industrial NLP applications** (e.g. information extraction, voice-bots)

---

## Dataset

### WiNER (Wikinews for Named Entity Recognition)

- **Language:** French  
- **Domain:** News (Wikinews)  
- **Annotation format:** BRAT (`.txt` + `.ann`)  
- **Special feature:** Supports **nested named entities**

### Entity Types in WiNER

- `Person`
- `Location`
- `Organization`
- `Date`
- `Event`
- `Product`
- `Hour`

Dataset location: data/WiNER-fr/


---

##  Evaluated Methods

### 1️ spaCy (Industrial NLP Pipeline)

Models:
- `fr_core_news_sm`
- `fr_core_news_md` (or `lg`)

Characteristics:
- CNN-based architecture
- Fast inference
- Widely used in production systems

---

### 2️ Hugging Face Transformers (NER Fine-Tuned)

Models:
- `Jean-Baptiste/camembert-ner` (French-specific transformer)
- `cmarkea/distilcamembert-base-ner` (lightweight French transformer)
- `Davlan/bert-base-multilingual-cased-ner-hrl` (multilingual baseline)

Characteristics:
- Transformer-based models
- High accuracy
- Higher computational cost

---

### 3️ Flair

Model:
- `flair/ner-french`

Characteristics:
- BiLSTM + CRF architecture
- Strong pre-transformer baseline
- Sentence-level sequence tagging

---

### 4️ Rule-Based Baseline (DATE / HOUR)

Method:
- Regular-expression-based extraction

Target entities:
- `Date`
- `Hour`

Purpose:
- Provide a **lower-bound baseline**
- Highlight strengths and limitations of rule-based systems
- Compare specialist rules vs statistical models

---


## 🗂 Project Structure

ner_benchmark/
│
├── README.md
│
├── data/
│ └── WiNER-fr/
│
├── src/
│ ├── load_winer.py
│ ├── run_spacy.py
│ ├── run_hf.py
│ ├── run_flair.py
│ ├── run_rules.py
│ ├── evaluate.py
│ └── utils.py
│
├── scripts/
│ ├── run_spacy_.sh
│ ├── run_hf_.sh
│ ├── run_flair_.sh
│ └── run_rules_.sh
│
├── results/
│ ├── predictions/
│ └── eval/



---

## Inference Scripts (`src/`)

| Script | Description |
|------|-------------|
| `load_winer.py` | Load WiNER BRAT annotations |
| `run_spacy.py` | Run spaCy NER models |
| `run_hf.py` | Run Hugging Face NER models |
| `run_flair.py` | Run Flair NER |
| `run_rules.py` | Rule-based DATE/HOUR extraction |
| `evaluate.py` | Evaluation and analysis |

---

##  Evaluation Protocol

### ✔ Core Metrics (Entity-Level Exact Match)

- **Precision**
- **Recall**
- **F1-score**

A prediction is considered correct **only if**:
- Start offset matches
- End offset matches
- Entity label matches

---

### ✔ Averaging Schemes

- **Micro-average**
  - Aggregates all entities
  - Dominated by frequent classes
- **Macro-average**
  - Average across entity types
  - Highlights performance on rare entities

---

### ✔ Per-Entity-Type Evaluation

Metrics computed independently for each entity type:

- Person
- Location
- Organization
- Date
- Event
- Product
- Hour

This analysis allows:
- Identification of **easy vs hard entities**
- Detection of **model biases**

---

### ✔ Nested Entity Analysis (WiNER-Specific)

- Proportion of nested entities in gold annotations
- Exact-match detection rate for nested entities
- Prediction overlap rate (proxy for nesting support)

> Most NER tools output **flat entities** and do not explicitly support nesting.