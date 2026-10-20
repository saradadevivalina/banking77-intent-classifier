# Customer-Support Intent Classification with DistilBERT (Banking77)

Fine-tuned a **Transformer (DistilBERT)** to route customer-support messages to the right intent, across **77 fine-grained banking intents**, and compared it with a classical **TF-IDF + Logistic Regression** baseline.

**Notebook:** [`banking77_intent_classifier.ipynb`](banking77_intent_classifier.ipynb) (runs end to end on a free Google Colab T4 GPU in about 10 minutes)

## Why this problem

Support teams receive thousands of short, varied messages a day. Routing each one to the right intent ("lost card", "transfer not received", "why was I charged a fee") is the first step of any support workflow, whether it hands off to a human agent, a FAQ answer or a RAG assistant. The challenge is that many intents differ only in small details.

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
3. **Deep learning:** fine-tuned `distilbert-base-uncased` (~66M parameters) with a 77-way classification head, using the Hugging Face `Trainer`: 4 epochs, learning rate 5e-5, batch size 32, weight decay 0.01, learning-rate warm-up, mixed-precision (fp16) on GPU, and the **best epoch selected by validation macro-F1**
4. **Evaluation** on the held-out test set, then **error analysis** with per-class F1 and the confusion matrix

## Results (held-out test set, 3,080 messages)

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.8851 | 0.8854 |
| **DistilBERT, fine-tuned** | **0.9198** | **0.9198** |

Fine-tuning improved the baseline by **+3.5 accuracy points** and **+3.4 macro-F1 points**.

## Error analysis

**Hardest intents (lowest test F1):**

| Intent | Precision | Recall | F1 |
|---|---|---|---|
| why_verify_identity | 0.763 | 0.725 | 0.744 |
| balance_not_updated_after_bank_transfer | 0.789 | 0.750 | 0.769 |
| verify_my_identity | 0.750 | 0.825 | 0.786 |
| pending_transfer | 0.838 | 0.775 | 0.805 |
| topping_up_by_card | 0.861 | 0.775 | 0.816 |

**Most frequent confusions (true → predicted):**

| Count | True intent | Predicted intent |
|---|---|---|
| 10 | why_verify_identity | verify_my_identity |
| 5 | verify_my_identity | why_verify_identity |
| 5 | fiat_currency_support | exchange_via_app |
| 4 | wrong_exchange_rate_for_cash_withdrawal | card_payment_wrong_exchange_rate |
| 4 | get_disposable_virtual_card | getting_virtual_card |
| 4 | card_delivery_estimate | card_arrival |
| 4 | card_arrival | card_delivery_estimate |
| 4 | balance_not_updated_after_bank_transfer | transfer_timing |

**What this shows:** the model rarely confuses *topics*. Almost all errors are between **near-duplicate intents within the same topic**. The single biggest source is *why* identity verification is needed versus *how* to do it (15 errors in total), which differ only by the question word. "Where is my card?" versus "When will my card arrive?" behaves the same way.

**What I would do next in production:**
- Merge or relabel near-duplicate intent pairs, if the business routes them to the same team anyway
- Add a **confidence threshold**: low-confidence predictions go to a human agent, or trigger a clarifying question
- Collect more examples for the weakest intents

## Try it

The notebook ends with a `predict_intent()` helper that returns the top-3 intents with probabilities, for example:
```python
predict_intent("I lost my card yesterday, what should I do?")
```

## Tech stack

Python · pandas · scikit-learn · PyTorch · Hugging Face Transformers (`Trainer`, `AutoModelForSequenceClassification`) · Hugging Face Datasets · Matplotlib · Google Colab (T4 GPU)
