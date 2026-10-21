# Customer-Support Intent Classification with DistilBERT (Banking77)

Fine-tuned a **Transformer (DistilBERT)** to route customer-support messages to the right intent across **77 fine-grained banking intents**, compared it with a classical **TF-IDF + Logistic Regression** baseline, measured how much routing it can **automate safely**, and validated the error analysis with **unsupervised clustering**. All runs are tracked with **MLflow**.

**Notebook:** [`banking77_intent_classifier.ipynb`](banking77_intent_classifier.ipynb) (runs end to end on a free Google Colab T4 GPU in about 15 minutes)

## Headline results

| | |
|---|---|
| **Test accuracy / macro-F1** | **91.7% / 0.917** (baseline 88.5% / 0.885) |
| **Automation at 0.8 confidence** | **81.3%** of messages routed automatically at **97.9%** accuracy; 18.7% go to an agent |
| **Clustering check** | **6 of the 10** most-confused intent pairs share a K-Means cluster (fewer than 1 expected by chance) |

## Why this problem

Support teams receive thousands of short, varied messages a day. Routing each one to the right intent ("lost card", "transfer not received", "why was I charged a fee") is the first step of any support workflow, whether it goes to a human agent, a FAQ answer or a RAG assistant. The challenge is that many intents differ only in small details, and a wrong route costs a customer time.

## Data

[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI): about 13,000 real online-banking customer queries labelled with 77 intents.

| Split | Messages |
|---|---|
| Train (90% of official train, stratified) | ~9,000 |
| Validation (10% of official train, stratified) | ~1,000 |
| Test (official, untouched until the end) | 3,080 |

The classes are moderately imbalanced, so **macro-F1** (every intent weighted equally) is reported alongside accuracy.

## Approach

1. **EDA:** examples per intent and message length (short messages, so a 64-token limit is enough)
2. **Baseline:** TF-IDF (unigrams + bigrams) + Logistic Regression, with the vectorizer fit on training data only
3. **Feature selection:** chi-squared `SelectKBest` on the baseline's TF-IDF features
4. **Deep learning:** fine-tuned `distilbert-base-uncased` (~66M parameters) with a 77-way classification head using the Hugging Face `Trainer`: 4 epochs, learning rate 5e-5, batch size 32, weight decay 0.01, learning-rate warm-up, mixed precision (fp16), **best epoch selected by validation macro-F1**
5. **Evaluation** on the held-out test set, then **error analysis** with per-class F1 and the confusion matrix
6. **Business impact:** confidence-threshold analysis of automation rate versus accuracy
7. **Unsupervised validation:** intent embeddings (`all-MiniLM-L6-v2`), **PCA** dimensionality reduction, **K-Means** clustering with the number of clusters chosen by silhouette score
8. **Experiment tracking:** every run's parameters and metrics logged to **MLflow**

## Results (held-out test set, 3,080 messages), as tracked in MLflow

| Run | Features | Test accuracy | Test macro-F1 |
|---|---|---|---|
| **DistilBERT, fine-tuned** | — | **0.9169** | **0.9168** |
| TF-IDF + Logistic Regression (baseline) | 9,640 | 0.8851 | 0.8854 |
| TF-IDF + chi² + Logistic Regression | 5,000 | 0.8831 | 0.8833 |
| TF-IDF + chi² + Logistic Regression | 2,500 | 0.8792 | 0.8796 |
| TF-IDF + chi² + Logistic Regression | 1,000 | 0.8406 | 0.8423 |

- Fine-tuning improved on the baseline by **+3.2 accuracy points** and **+3.1 macro-F1 points**.
- **Feature selection:** keeping only the top 5,000 of 9,640 n-grams (48% fewer) costs just 0.2 accuracy points, so about half the features carry almost all the signal. Cutting to 1,000 costs 4.5 points.

## Business impact: how much routing can be automated?

The model's highest softmax probability is used as a confidence score. Messages above the threshold are routed automatically; the rest go to a human agent.

| Confidence threshold | Auto-routed | Accuracy on auto-routed | Sent to an agent |
|---|---|---|---|
| none | 100.0% | 91.7% | 0.0% |
| 0.5 | 94.6% | 94.6% | 5.4% |
| 0.7 | 87.9% | 97.0% | 12.1% |
| **0.8** | **81.3%** | **97.9%** | **18.7%** |
| 0.9 | 67.6% | 98.8% | 32.4% |
| 0.95 | 44.9% | 99.6% | 55.1% |

At a **0.8** threshold the model would handle **about 4 in 5 messages on its own while being right 97.9% of the time**, leaving agents only the ambiguous fifth. The threshold is a business dial: a team that values accuracy over automation can raise it.

## Error analysis

**Hardest intents (lowest test F1):**

| Intent | Precision | Recall | F1 |
|---|---|---|---|
| why_verify_identity | 0.794 | 0.675 | 0.730 |
| topping_up_by_card | 0.795 | 0.775 | 0.785 |
| balance_not_updated_after_bank_transfer | 0.816 | 0.775 | 0.795 |
| compromised_card | 0.825 | 0.825 | 0.825 |
| verify_my_identity | 0.745 | 0.950 | 0.835 |

**Most frequent confusions (true → predicted):**

| Count | True intent | Predicted intent | Same K-Means cluster? |
|---|---|---|---|
| 12 | why_verify_identity | verify_my_identity | yes |
| 5 | balance_not_updated_after_bank_transfer | transfer_timing | yes |
| 4 | unable_to_verify_identity | why_verify_identity | yes |
| 4 | transfer_into_account | topping_up_by_card | no |
| 4 | get_disposable_virtual_card | getting_virtual_card | no |
| 4 | declined_transfer | declined_card_payment | no |
| 3 | wrong_amount_of_cash_received | declined_cash_withdrawal | yes |
| 3 | transfer_timing | balance_not_updated_after_bank_transfer | yes |
| 3 | topping_up_by_card | compromised_card | no |
| 3 | top_up_reverted | top_up_failed | yes |

The model rarely confuses *topics*. Most errors are between **near-duplicate intents within one topic**. The identity-verification group alone (*why* / *how* / *unable to* verify) accounts for 16 of these errors.

## Unsupervised validation: do the confused intents really overlap?

To test the error analysis independently of the classifier, each intent was represented by the mean sentence embedding of its training messages (`all-MiniLM-L6-v2`, 384 dimensions).

- **PCA:** 27 of 384 dimensions keep 90% of the variance between intents.
- **K-Means:** the number of clusters was chosen by silhouette score over k = 8 to 24, which selected **k = 24**. Silhouette scores are low overall (about 0.2), as expected when many intents genuinely overlap. The best score was at the top of the tested range, so a wider range is worth trying.
- **Result:** **6 of the 10 most-confused pairs fall in the same cluster.** With 24 clusters, two random intents share a cluster only about 1 time in 24, so fewer than 1 match would be expected by chance.

This confirms that the remaining errors come from genuinely overlapping intents rather than random model mistakes.

## What I would do next in production

- **Confidence threshold:** auto-route above about 0.8 and send the rest to an agent or ask a clarifying question
- **Two-stage routing:** predict the cluster first, then the exact intent within it
- **Taxonomy clean-up:** merge or relabel near-duplicate intents that the business routes to the same team anyway
- **More data** for the weakest intents, and **monitoring** of the confidence distribution in production to catch drift

## Experiment tracking

All runs are logged to MLflow (`sqlite:///mlflow.db`) under the experiment `banking77-intent-classification`, with model parameters (learning rate, epochs, batch size, number of features), per-epoch validation macro-F1 and loss for DistilBERT, and test accuracy, macro-F1 and automation metrics.

## Try it

The notebook includes a `predict_intent()` helper that returns the top-3 intents with probabilities:
```python
predict_intent("I lost my card yesterday, what should I do?")
```

## Tech stack

Python · pandas · scikit-learn (TF-IDF, Logistic Regression, chi² feature selection, PCA, K-Means, silhouette) · PyTorch · Hugging Face Transformers (`Trainer`) · Hugging Face Datasets · sentence-transformers · MLflow · Matplotlib · Google Colab (T4 GPU)
