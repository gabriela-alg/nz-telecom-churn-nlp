# NZ Telecom Churn Risk — NLP and Machine Learning on Public Customer Reviews

This project applies Natural Language Processing and machine learning to public customer feedback from New Zealand telecommunications providers, with the goal of identifying churn risk drivers without access to internal customer data. It was developed as an Applied Project for a Master of Business Informatics at ICL Graduate Business School, New Zealand (2026).

---

## Problem

Churn prediction research typically relies on internal customer records that are not accessible outside the companies that hold them. This project investigates whether publicly available reviews can serve as an alternative source for identifying the service factors most associated with customer dissatisfaction and switching risk.

---

## Data

- **1,929 entries** collected from two public platforms between June 2023 and June 2026
- **Trustpilot**: 1,502 reviews across Spark (316), One NZ (596) and 2degrees (590)
- **Geekzone**: 427 forum posts across Spark (155), One NZ (128) and 2degrees (144)
- Churn risk proxy: 1–2 star ratings = high risk, 4–5 stars = low risk, 3 stars excluded
- Classification dataset: 1,483 Trustpilot reviews (1,330 high risk, 153 low risk)
- Geekzone posts carry no star ratings and were used for descriptive and NLP analysis only

---

## Methods

| Stage | Tool / Library |
|---|---|
| Data collection (Trustpilot) | Playwright + BeautifulSoup (manual page saving) |
| Data collection (Geekzone) | BeautifulSoup + requests |
| Data storage | SQLite |
| Sentiment analysis | VADER (vaderSentiment) |
| Text pre-processing | spaCy (lemmatisation, POS filtering) |
| Term frequency features | TF-IDF (scikit-learn, 300 terms, unigrams and bigrams) |
| Topic modelling | LDA via gensim (Trustpilot k=5, Geekzone k=4, fitted separately) |
| Classification | Logistic regression and decision tree (scikit-learn, class-weighted) |
| Visualisation | Matplotlib, Seaborn |

---

## Results

### Model performance

| Model | Features | ROC-AUC |
|---|---|---|
| Logistic regression | TF-IDF + VADER | **0.957** |
| Decision tree | TF-IDF + VADER | 0.851 |
| Logistic regression | TF-IDF only | 0.935 |
| Decision tree | TF-IDF only | 0.774 |

The TF-IDF-only logistic regression retaining ROC-AUC 0.935 indicates that the vocabulary of the reviews carries most of the churn-risk signal independently of the sentiment scores.

### Top churn risk drivers (by prevalence difference between classes)

| Term | High risk (%) | Low risk (%) | Difference (pp) |
|---|---|---|---|
| bad | 23.3 | 3.3 | +20.0 |
| month | 18.8 | 3.9 | +14.9 |
| time | 28.2 | 13.7 | +14.5 |
| account | 16.3 | 2.6 | +13.7 |
| charge | 16.2 | 3.3 | +12.9 |
| hour | 14.7 | 3.3 | +11.4 |
| customer service | 31.6 | 20.3 | +11.3 |
| cancel | 11.4 | 0.7 | +10.8 |

### Topic themes

**Trustpilot (5 themes):** Contracts and In-Store Experience (287 docs), Billing and Payments (276), Mobile Plans and Data Usage (218), Network Speed and Connectivity (178), General Service Evaluation (144). All themes carry negative average sentiment (corpus mean: −0.23).

**Geekzone (4 themes):** Fibre and Connection Configuration (88 posts), Mobile Plans and Account Issues (70), App and Customer Service Problems (62), Home Network Setup and Equipment (56). All themes carry positive average sentiment (corpus mean: +0.34).

---

## Repository structure

```
├── 01_data_collection.ipynb      # Trustpilot (Playwright + BeautifulSoup) and Geekzone scraping
├── 02_data_preparation.ipynb     # Cleaning, deduplication, VADER scoring, spaCy pre-processing
├── 03_nlp_analysis.ipynb         # TF-IDF term ranking, LDA topic modelling, driver proportion analysis
├── 04_modelling.ipynb            # Logistic regression and decision tree, evaluation, provider-level drivers
├── data/
│   ├── reviews.csv               # Raw consolidated dataset (1,929 entries)
│   ├── reviews_processed.csv     # Processed dataset with VADER scores and topic assignments
│   ├── model_metrics.csv         # ROC-AUC, precision, recall and F1 for all four model configurations
│   ├── model_drivers.csv         # Logistic regression coefficients and decision tree importances
│   ├── provider_drivers.csv      # Driver term proportions per provider
│   └── quarterly_summary.csv    # Quarterly sentiment and high-risk proportions by source
```

---

## How to reproduce

1. Clone the repository
```bash
git clone https://github.com/gabriela-alg/nz-telecom-churn-nlp.git
cd nz-telecom-churn-nlp
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn gensim spacy vaderSentiment beautifulsoup4 requests matplotlib seaborn playwright
python -m spacy download en_core_web_sm
playwright install chromium
```

3. Run the notebooks in order
```
01_data_collection.ipynb
02_data_preparation.ipynb
03_nlp_analysis.ipynb
04_modelling.ipynb
```

> **Note on Trustpilot collection:** the collection notebook documents the Playwright setup used to render and save pages locally. The saved HTML files are not included in this repository. To reproduce the collection step, the pages need to be saved manually from the browser before running the parsing cells.

---

## Key findings

- Billing accuracy is the strongest and most cohesive churn-risk driver, with account, charge, bill, payment and credit all showing prevalence gaps above six percentage points between the high and low-risk classes.
- Customer service quality is the cross-cutting factor: its failure markers (hour on hold, unanswered emails, repeated contacts) rank among the strongest high-risk terms, while its success markers (helpful, friendly, patient) dominate the low-risk vocabulary.
- The two platforms produce opposite sentiment profiles from identical processing, reflecting the function of the platform rather than a difference in service quality, which is a methodological finding relevant to any study that mines public feedback.
- Provider signatures are visible around a shared core: connectivity reliability for Spark, response times for One NZ, and fixed broadband and hold times for 2degrees.

---

## Limitations

- Churn risk is approximated through star ratings, not confirmed switching behaviour.
- Trustpilot functions mainly as a complaint channel in the New Zealand market (86.3% of reviews are one-star), so the dataset over-represents dissatisfied customers.
- Geekzone posts carry no star ratings and were excluded from the supervised classification stage.
- The low-risk class is small (153 reviews), which limits the precision of the low-risk characterisation.

---

## Author

Gabriela Goulart  
Master of Business Informatics — ICL Graduate Business School, New Zealand (2026)  
[github.com/gabriela-alg](https://github.com/gabriela-alg)
