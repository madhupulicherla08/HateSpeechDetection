# Hate Speech Detection on Twitter using Transformer Embeddings and XGBoost

## Project Overview
This project aims to classify tweets into three categories: Hate Speech, Offensive Language, and Neither. It leverages advanced transformer-based embeddings from the `mxbai-embed-large-v1` model to extract contextual and semantic features, which are then used to train an XGBoost classifier. A comparative analysis with traditional N-gram + SVM and sentiment-based features is also presented.

## Key Features
- Transformer-based semantic feature extraction (1024-dim embeddings)
- Sentiment analysis using polarity and subjectivity scores
- Multi-class classification with XGBoost
- Class imbalance addressed via ADASYN oversampling
- Comprehensive evaluation using confusion matrix, ROC curves, and classification reports

## Project Files
| File | Description |
|------|-------------|
| `data_ingestion_preprocessed.ipynb` | Preprocessing steps including cleaning and tokenization |
| `feature_extraction.ipynb` | Embedding generation and sentiment feature extraction |
| `ensembel_model.ipynb` | XGBoost training, hyperparameter tuning, ADASYN oversampling |
| `Analysis.ipynb` | Model evaluation, visualization of metrics and results |

## Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/your-username/hate-speech-xgboost-transformer.git
cd hate-speech-xgboost-transformer
