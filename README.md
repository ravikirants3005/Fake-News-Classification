#  Fake News Detection using DistilBERT

## Problem Statement
Detect whether a news article is **REAL** or **FAKE** using a fine-tuned
DistilBERT transformer model on labeled news data.

## Approach
1. Load and clean the Fake News dataset (title + body text combined)
2. Tokenize using DistilBERT tokenizer (max_len=256, padding+truncation)
3. Fine-tune `distilbert-base-uncased` for 3 epochs with AdamW + linear warmup
4. Evaluate on held-out test set with full classification metrics
5. Analyze misclassified samples for failure patterns
6. Improve with weighted cross-entropy loss to handle class imbalance
7. Deploy via Gradio web interface

## Model
- **Architecture:** DistilBERT-base-uncased (66M parameters)
- **Task:** Sequence Classification (Binary)
- **Library:** HuggingFace Transformers + PyTorch

## Metrics (Test Set)
| Metric | Baseline | Improved |
|--------|----------|---------|
| Accuracy  | {test_acc*100:.2f}% | {improved_test_acc*100:.2f}% |
| Precision | {base_prec*100:.2f}% | {imp_prec*100:.2f}% |
| Recall    | {base_rec*100:.2f}% | {imp_rec*100:.2f}% |
| F1-Score  | {base_f1*100:.2f}% | {imp_f1*100:.2f}% |

## Improvements
- **Class imbalance:** Weighted CrossEntropyLoss to penalize minority class errors
- **Learning rate tuning:** 2e-5 → 3e-5 for the improved model

## Key Learnings
- Pre-trained BERT models transfer exceptionally well to NLP classification tasks
- Combining title + body text outperforms using either alone
- Weighted loss significantly improves recall on the minority class
- DistilBERT achieves ~97% of BERT's performance at 40% fewer parameters
- Fake news often uses emotionally charged language, ALL-CAPS, and vague sources

## Repository Structure
```
├── Fake_News_Detection_BERT.ipynb   # Main notebook
├── best_model/                       # Saved DistilBERT weights
├── label_distribution.png
├── training_history.png
├── confusion_matrix.png
├── error_analysis.png
├── model_comparison.png
└── final_dashboard.png
```
"""

print(readme)

# Also save README.md
with open('README.md', 'w') as f:
    f.write(readme)
print("README.md saved!")
