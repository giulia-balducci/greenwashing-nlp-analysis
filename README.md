# Greenwashing NLP Analysis: Reef-Safe Sunscreen Claims

**NLP analysis of greenwashing claims in reef-safe sunscreen marketing**  
VADER · RoBERTa · LDA · spaCy NER | Anchored to ACCC v Edgewell Personal Care (2025)

---

## Project Overview
This project was developed as one of the assignments for the Applied AI Bootcamp at AllWomen (Barcelona, April 2026). It applies NLP techniques to analyse greenwashing claims in reef-safe sunscreen marketing, anchored to the ACCC v Edgewell Personal Care Federal Court proceedings (July 2025).

Edgewell's Banana Boat and Hawaiian Tropic products were labelled "reef friendly" while containing oxybenzone and other chemical UV filters linked to coral bleaching.

This project analyses public discourse around these claims using a combination of topic modelling, sentiment analysis, named entity recognition, and transformer-based classification.

The dataset combines 8 real Reddit documents with 260 synthetic documents generated to augment the corpus around the ACCC case, for a total of 268 documents.

## Pipeline

| Notebook | Description | Output |
|----------|-------------|--------|
| `01_data_collection.ipynb` | Loads raw Reddit CSVs from 11 subreddits, filters by keyword, combines posts and comments | `greenwashing_dataset_filtered.csv` |
| `02_preprocessing_topic_modelling.ipynb` | Text preprocessing, LDA topic modelling (n=7), coherence score analysis, greenwashing subset extraction | `df_greenwashing_topic1.csv` |
| `03_data_augmentation.ipynb` | Keyword filtering of sunscreen subset (8 docs), synthetic data generation (260 docs) via Claude API | `df_sunscreen_augmented.csv` |
| `04_sentiment_ner.ipynb` | VADER sentiment analysis, spaCy NER with custom entity ruler, brand sentiment comparison, ACCC case analysis, dashboard | `df_sunscreen_final.csv` |
| `05_roberta.ipynb` | RoBERTa transformer sentiment analysis, VADER vs RoBERTa comparison | figures |

Notebooks are designed to run in sequence.

## Results

### Topic Modelling
LDA on the full filtered corpus (n=27,851, 7 topics) isolated a greenwashing-dominant topic of **3,602 documents (12.9% of the corpus)**. A second-level LDA on this subset (6 subtopics) informed the stratified synthetic data generation in notebook 03.

### Dataset
The sunscreen-specific corpus combines **8 real Reddit documents** with **260 synthetic documents**, for a total of **268 documents** (real + synthetic).

### Sentiment Analysis: VADER vs RoBERTa
RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) is substantially more critical than VADER on this corpus:

| | VADER (rule-based) | RoBERTa (transformer) |
|---|---|---|
| Positive | 67.5% (181) | 22.0% (59) |
| Negative | 23.9% (64) | 41.0% (110) |
| Neutral | 8.6% (23) | 36.9% (99) |

The gap is consistent with VADER over-crediting sarcastic or hedged language as positive — a limitation RoBERTa's transformer architecture is better equipped to handle.

### Brand-Level Sentiment (RoBERTa, entity-based)
Using spaCy entity mentions (not string matching) to attribute documents to brands, all four brands named in the ACCC case score negative, and all five reef-safe brands score positive:

| Brand | Category | Mean RoBERTa score |
|---|---|---|
| Banana Boat | Greenwashing | -0.53 |
| Hawaiian Tropic | Greenwashing | -0.52 |
| Coppertone | Greenwashing | -0.50 |
| Neutrogena | Greenwashing | -0.08 |
| Raw Elements | Reef-safe | +0.05 |
| Stream2Sea | Reef-safe | +0.17 |
| Sun Bum | Reef-safe | +0.25 |
| Thinksport | Reef-safe | +0.27 |
| Badger Balm | Reef-safe | +0.44 |

### ACCC Case Study
Of the 268 documents, **75 mention the ACCC case** (`ACCC`, `Edgewell`, `Federal Court`). This subset shows the sharpest VADER/RoBERTa divergence in the whole analysis:

| | VADER | RoBERTa |
|---|---|---|
| Positive | 72.0% (54) | 10.7% (8) |
| Negative | 26.7% (20) | 41.3% (31) |
| Neutral | 1.3% (1) | 48.0% (36) |

VADER reads celebratory reactions to the ACCC ruling ("finally called out", "about time") as lexically positive; RoBERTa correctly identifies the underlying sentiment toward Edgewell/Banana Boat/Hawaiian Tropic as negative. This is the clearest evidence in the dataset for VADER's sarcasm-detection limitation flagged in notebook 04.

### Limitations
Both models score sentiment at the document level, not the entity level: reef-safe brands mentioned inside otherwise negative documents (e.g. "betrayed by Banana Boat, switched to Raw Elements") have their positive mention pulled toward neutral by the surrounding negativity — this is the likely explanation for Raw Elements' comparatively weak +0.05 score. RoBERTa's 512-token truncation affects only 2 of the 268 documents (0.7%), so it has negligible impact on these results.

## Repository Structure

```
greenwashing-nlp-analysis/
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing_topic_modelling.ipynb
│   ├── 02b_filtered_coherence_score_colab.ipynb   # run on Google Colab
│   ├── 02c_subset_coherence_score_colab.ipynb     # run on Google Colab
│   ├── 03_data_augmentation.ipynb
│   ├── 04_sentiment_ner.ipynb
│   └── 05_roberta.ipynb
├── data/
│   ├── raw/                             # not tracked (too large for GitHub)
│   │   └── <SubredditName>/
│   │       ├── Articles/
│   │       ├── Comments/
│   │       └── TextPosts/
│   └── processed/
│       ├── greenwashing_dataset_filtered.csv   # output of notebook 01
│       ├── df_greenwashing_topic1.csv          # output of notebook 02
│       ├── df_sunscreen_real.csv               # output of notebook 03 (real subset)
│       ├── df_sunscreen_synthetic.csv          # output of notebook 03 (synthetic)
│       ├── df_sunscreen_augmented.csv          # output of notebook 03 (combined)
│       └── df_sunscreen_final.csv              # output of notebook 04
├── figures/                             # generated by notebooks 04 and 05
├── .gitignore
├── requirements.txt
└── README.md
```

All processed datasets are tracked in the repository so that any notebook can be run independently without regenerating upstream outputs.

## Setup

```bash
# Clone the repository
git clone https://github.com/giulia-balducci/greenwashing-nlp-analysis.git
cd greenwashing-nlp-analysis

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

```python
# Download NLTK data (run once)
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
```

**Note:** `gensim` (coherence score analysis in notebook 02) requires Python ≤ 3.12. If your local environment uses Python 3.13+, run the coherence analysis on Google Colab using the precomputed scores included in the notebook markdown cells.

**Note:** `05_roberta.ipynb` downloads `cardiffnlp/twitter-roberta-base-sentiment-latest` and runs it locally via `transformers`. It works on CPU but is faster with a GPU/MPS-enabled PyTorch install.

## Data

### Raw data (not included)
Reddit posts and comments were collected from 11 subreddits related to sustainability and greenwashing:
`r/greenwashing`, `r/environment`, `r/climate`, `r/climatechange`, `r/sustainability`, `r/green`, `r/Anticonsumption`, `r/ClimateActionPlan`, `r/ClimateOffensive`, `r/SustainableFashion`, `r/ZeroWaste`.

The raw CSVs are not tracked in this repository due to size. The processed outputs (starting from `greenwashing_dataset_filtered.csv`) are included and sufficient to reproduce all downstream analysis.

### Processed data
All files in `data/processed/` are tracked and ready to use:

| File | Rows | Description |
|------|------|-------------|
| `greenwashing_dataset_filtered.csv` | 27,851 | Full Reddit corpus after keyword filtering and outlier removal |
| `df_greenwashing_topic1.csv` | 3,602 | Greenwashing-dominant topic subset (LDA topic 1) |
| `df_sunscreen_real.csv` | 8 | Real Reddit documents mentioning reef-safe sunscreen |
| `df_sunscreen_synthetic.csv` | 260 | Synthetic documents generated via Claude API |
| `df_sunscreen_augmented.csv` | 268 | Combined real + synthetic corpus |
| `df_sunscreen_final.csv` | 268 | Final dataset with sentiment scores and NER output |

## Author

Giulia Balducci  
Applied AI Bootcamp — AllWomen, Barcelona (April 2026)  
GitHub: [giulia-balducci](https://github.com/giulia-balducci)
