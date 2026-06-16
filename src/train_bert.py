import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.utils import logger, get_torch_device, setup_directories

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class NewsDataset(Dataset):
    """Custom PyTorch Dataset for loading news articles."""
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def compute_metrics(eval_pred):
    """Calculates accuracy, precision, recall, and F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def train_bert(epochs: int = 3, batch_size: int = 8, quick: bool = False):
    setup_directories(BASE_DIR)
    device = get_torch_device()
    
    data_dir = os.path.join(BASE_DIR, "data")
    models_dir = os.path.join(BASE_DIR, "models")
    output_dir = os.path.join(models_dir, "distilbert")
    
    logger.info("Loading preprocessed datasets...")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv")).fillna("")
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv")).fillna("")
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv")).fillna("")
    
    if quick:
        logger.info("Running in QUICK mode. Truncating datasets to 20 samples each.")
        train_df = train_df.head(20)
        val_df = val_df.head(20)
        test_df = test_df.head(20)
        epochs = 1
        
    logger.info(f"Loaded train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    
    # Load DistilBERT tokenizer
    logger.info("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    logger.info("Tokenizing datasets...")
    train_encodings = tokenizer(train_df["clean_text"].tolist(), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_df["clean_text"].tolist(), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_df["clean_text"].tolist(), truncation=True, padding=True, max_length=512)
    
    # Create PyTorch datasets
    train_dataset = NewsDataset(train_encodings, train_df["label"].tolist())
    val_dataset = NewsDataset(val_encodings, val_df["label"].tolist())
    test_dataset = NewsDataset(test_encodings, test_df["label"].tolist())
    
    # Load DistilBERT model
    logger.info("Loading DistilBERT sequence classification model...")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(device)
    
    # Define training arguments
    try:
        training_args = TrainingArguments(
            output_dir=os.path.join(models_dir, "bert_checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir=os.path.join(models_dir, "bert_logs"),
            logging_steps=10 if quick else 100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            # Use CPU if device is CPU, otherwise use GPU/MPS config
            use_cpu=(device.type == "cpu"),
            report_to="none"
        )
    except TypeError:
        # Fallback for older versions of transformers
        training_args = TrainingArguments(
            output_dir=os.path.join(models_dir, "bert_checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir=os.path.join(models_dir, "bert_logs"),
            logging_steps=10 if quick else 100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            # Use CPU if device is CPU, otherwise use GPU/MPS config
            use_cpu=(device.type == "cpu"),
            report_to="none"
        )
    
    logger.info("Starting fine-tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if not quick else []
    )
    
    trainer.train()
    logger.info("Fine-tuning completed. Evaluating on test set...")
    
    # Evaluate model on test set
    eval_results = trainer.evaluate(test_dataset)
    logger.info(f"--- DistilBERT Test Evaluation ---")
    logger.info(f"Accuracy:  {eval_results['eval_accuracy']:.4f}")
    logger.info(f"F1-Score:  {eval_results['eval_f1']:.4f}")
    logger.info(f"Precision: {eval_results['eval_precision']:.4f}")
    logger.info(f"Recall:    {eval_results['eval_recall']:.4f}")
    
    # Get confusion matrix
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    cm = confusion_matrix(test_df["label"].tolist(), preds)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Save the best model and tokenizer
    logger.info(f"Saving fine-tuned model and tokenizer to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metrics summary
    torch.save(eval_results, os.path.join(output_dir, "eval_results.pt"))
    logger.info("DistilBERT model and metrics saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for Fake News Detection.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--quick", action="store_true", help="Run a quick training pass on a tiny subset")
    args = parser.parse_args()
    
    train_bert(epochs=args.epochs, batch_size=args.batch_size, quick=args.quick)
