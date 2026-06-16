import os
import re
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

from src.utils import logger, download_nltk_resources

# Initialize NLTK resource directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
download_nltk_resources(BASE_DIR)

# Get stopwords list
try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    # Fallback in case NLTK stopwords aren't loaded yet
    STOPWORDS = set()

# Initialize NLTK Lemmatizer as a fallback
nltk_lemmatizer = WordNetLemmatizer()

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception as e:
    logger.warning(f"Failed to load SpaCy model 'en_core_web_sm' ({e}). Will fall back to NLTK for lemmatization.")
    nlp = None

def clean_text(text: str, use_spacy: bool = True) -> str:
    """
    Cleans raw text by:
    1. Lowercasing
    2. Removing punctuation and special characters
    3. Tokenizing and removing stopwords
    4. Lemmatizing (using spaCy or NLTK fallback)
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase and remove URLs/handles
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"\@\w+", "", text)
    
    # 2. Remove punctuation
    # Keep alphanumeric characters and spaces
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # 3 & 4. Tokenize, Stopword removal, and Lemmatization
    if use_spacy and nlp is not None:
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in STOPWORDS and not token.is_space]
    else:
        # NLTK Fallback
        tokens = text.split()
        tokens = [nltk_lemmatizer.lemmatize(word) for word in tokens if word not in STOPWORDS]
        
    return " ".join(tokens)

def generate_mock_data(data_dir: str):
    """Generates synthetic Fake.csv and True.csv files to ensure pipeline can run out-of-the-box."""
    true_path = os.path.join(data_dir, "True.csv")
    fake_path = os.path.join(data_dir, "Fake.csv")
    
    if os.path.exists(true_path) and os.path.exists(fake_path):
        logger.info("True.csv and Fake.csv already exist. Skipping mock data generation.")
        return
        
    logger.info("Generating mock dataset for training and verification...")
    
    # Real news templates
    real_texts = [
        "The Senate passed a bipartisan budget bill today after weeks of intense negotiations between lawmakers.",
        "The President announced a new initiative to fund public schools and update national infrastructure.",
        "Scientists at the research institute have discovered a new method to recycle plastics efficiently.",
        "The Federal Reserve decided to keep interest rates steady, citing stable economic growth and low inflation.",
        "State officials are preparing for the upcoming election, reminding citizens of key voter registration deadlines.",
        "The governor signed a historic climate pact aiming to reduce state emissions by 40 percent in a decade.",
        "A new study published in the medical journal shows that regular exercise improves cognitive longevity.",
        "City officials approved the construction of a new public transit line connecting the suburbs to downtown.",
        "The foreign minister met with international allies to discuss global trade policies and security agreements.",
        "NASA successfully launched its latest satellite to monitor ocean temperatures and weather patterns."
    ] * 15 # 150 rows
    
    real_data = pd.DataFrame({
        "title": [f"Official Report: Senate passes major budget bill {i}" for i in range(len(real_texts))],
        "text": real_texts,
        "subject": "politics",
        "date": "June 16, 2026"
    })
    
    # Fake news templates
    fake_texts = [
        "Shocking secret report reveals that politicians are using mind control signals in public parks!",
        "You won't believe what the government is hiding: evidence of alien spacecraft in the desert!",
        "A secret organization has been secretly manipulating global weather patterns to ruin crops.",
        "Breaking: Miraculous herbal tea proven to cure all major diseases in less than 24 hours!",
        "Confidential source leaks plans of secret tax that will seize all private savings accounts next month.",
        "Unbelievable video shows celebrity admitting they are actually a holographic projection.",
        "Local man discovers ancient gold treasury hidden beneath city hall, officials try to cover it up.",
        "New research proves that drinking carbonated water causes immediate memory loss.",
        "Conspiracy exposed: the moon landing was filmed in a television studio in New Jersey.",
        "Urgent warning: massive solar flare to shut down the entire global internet forever tonight!"
    ] * 15 # 150 rows
    
    fake_data = pd.DataFrame({
        "title": [f"SHOCKING: Secret truth revealed about {i}!" for i in range(len(fake_texts))],
        "text": fake_texts,
        "subject": "US_News",
        "date": "June 16, 2026"
    })
    
    real_data.to_csv(true_path, index=False)
    fake_data.to_csv(fake_path, index=False)
    logger.info(f"Generated mock data at {true_path} and {fake_path}.")

def prepare_data(data_dir: str):
    """
    Loads, cleans, labels, and splits True.csv and Fake.csv.
    Saves clean_train.csv, clean_val.csv, clean_test.csv.
    """
    # Create directories first
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate mock data if missing
    generate_mock_data(data_dir)
    
    true_df = pd.read_csv(os.path.join(data_dir, "True.csv"))
    fake_df = pd.read_csv(os.path.join(data_dir, "Fake.csv"))
    
    logger.info(f"Loaded {len(true_df)} real articles and {len(fake_df)} fake articles.")
    
    # Assign labels
    true_df["label"] = 1  # REAL
    fake_df["label"] = 0  # FAKE
    
    # Merge
    df = pd.concat([true_df, fake_df], ignore_index=True)
    
    # Fill missing values and handle duplicates
    df = df.dropna(subset=["text", "title"])
    df = df.drop_duplicates(subset=["text"])
    
    # Combine title and text for classification
    logger.info("Combining title and text for feature extraction...")
    df["full_text"] = df["title"] + " " + df["text"]
    
    # Clean text
    logger.info("Cleaning and lemmatizing text (this might take a moment)...")
    # Using spacy if available, process in batch or row by row
    df["clean_text"] = df["full_text"].apply(lambda x: clean_text(x, use_spacy=(nlp is not None)))
    
    # Remove rows that ended up empty after cleaning
    df = df[df["clean_text"].str.strip() != ""]
    
    logger.info(f"Dataset size after cleaning: {len(df)}")
    
    # Split: Train/Val/Test (70% / 15% / 15%)
    # Stratified to ensure equal distribution of labels
    train_val_df, test_df = train_test_split(
        df[["clean_text", "label"]], 
        test_size=0.15, 
        random_state=42, 
        stratify=df["label"]
    )
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.1765,  # 0.15 / 0.85 approx 17.65% to get 15% of total
        random_state=42, 
        stratify=train_val_df["label"]
    )
    
    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Save datasets
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    logger.info("Saved train.csv, val.csv, and test.csv.")

if __name__ == "__main__":
    # Test script execution
    prepare_data(os.path.join(BASE_DIR, "data"))
