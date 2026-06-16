import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.utils import logger, setup_directories

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(data_dir: str):
    """Loads clean train, validation, and test datasets."""
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    # Fill any potential NaN values
    train_df["clean_text"] = train_df["clean_text"].fillna("")
    val_df["clean_text"] = val_df["clean_text"].fillna("")
    test_df["clean_text"] = test_df["clean_text"].fillna("")
    
    return train_df, val_df, test_df

def evaluate_model(model, X_test, y_test, name: str):
    """Evaluates the model and prints key metrics."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")
    cm = confusion_matrix(y_test, preds)
    
    logger.info(f"--- {name} Evaluation ---")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }

def train_baselines():
    setup_directories(BASE_DIR)
    data_dir = os.path.join(BASE_DIR, "data")
    models_dir = os.path.join(BASE_DIR, "models")
    
    logger.info("Loading preprocessed datasets...")
    train_df, val_df, test_df = load_data(data_dir)
    
    X_train_raw = train_df["clean_text"].values
    y_train = train_df["label"].values
    
    X_test_raw = test_df["clean_text"].values
    y_test = test_df["label"].values
    
    logger.info("Fitting TF-IDF Vectorizer...")
    # Using unigrams and bigrams, limited to top 10000 features for optimal balance of speed and representation
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # Save the vectorizer
    vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Saved TF-IDF Vectorizer to {vectorizer_path}")
    
    # 1. Logistic Regression
    logger.info("Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # 2. Random Forest
    logger.info("Training Random Forest model (this can take a moment)...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Compare and save the best baseline
    if lr_metrics["f1"] >= rf_metrics["f1"]:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_metrics = lr_metrics
    else:
        best_model = rf_model
        best_name = "Random Forest"
        best_metrics = rf_metrics
        
    best_model_path = os.path.join(models_dir, "baseline_model.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model, f)
        
    logger.info(f"Saved the best baseline model ({best_name}) to {best_model_path}")
    
    # Save performance summary
    summary_path = os.path.join(models_dir, "baseline_summary.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump({"logistic_regression": lr_metrics, "random_forest": rf_metrics, "best_baseline_name": best_name}, f)

if __name__ == "__main__":
    train_baselines()
