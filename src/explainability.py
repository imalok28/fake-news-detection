import re
import numpy as np
import torch
from lime.lime_text import LimeTextExplainer

from src.preprocessing import clean_text
from src.utils import logger, get_torch_device

class NewsExplainer:
    """Generates word-level explanations for fake news classification using LIME."""
    
    def __init__(self, model, model_type="baseline", vectorizer=None, tokenizer=None):
        self.model = model
        self.model_type = model_type
        self.vectorizer = vectorizer
        self.tokenizer = tokenizer
        self.explainer = LimeTextExplainer(class_names=["FAKE", "REAL"])
        self.device = get_torch_device() if model_type == "bert" else None
        
        if model_type == "bert" and self.model is not None:
            self.model.eval()

    def _predict_proba_baseline(self, texts):
        """Prediction function for LIME when using TF-IDF baseline models."""
        cleaned_texts = [clean_text(text, use_spacy=False) for text in texts]
        tfidf_features = self.vectorizer.transform(cleaned_texts)
        return self.model.predict_proba(tfidf_features)

    def _predict_proba_bert(self, texts):
        """Prediction function for LIME when using the DistilBERT model."""
        # BERT takes cleaned or uncleaned text, but since it was trained on clean_text,
        # we will clean the perturbed texts to align preprocessing
        cleaned_texts = [clean_text(text, use_spacy=False) for text in texts]
        
        # Tokenize and run prediction in batches to avoid OOM
        batch_size = 32
        all_probs = []
        
        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i:i + batch_size]
            encodings = self.tokenizer(
                batch, 
                truncation=True, 
                padding=True, 
                max_length=512, 
                return_tensors="pt"
            )
            
            # Move inputs to device
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
                
        return np.vstack(all_probs)

    def explain(self, text: str, num_features: int = 10):
        """
        Explains the prediction for the given text.
        Returns:
            label: "REAL" or "FAKE"
            confidence: float (probability of the predicted class)
            influential_words: list of dicts with {"word": str, "weight": float, "influence": "REAL"/"FAKE"}
        """
        # Select prediction function
        if self.model_type == "bert":
            predict_fn = self._predict_proba_bert
        else:
            predict_fn = self._predict_proba_baseline
            
        # Get raw probabilities
        probs = predict_fn([text])[0]
        fake_prob, real_prob = probs[0], probs[1]
        
        # Determine label and confidence
        if real_prob >= 0.5:
            label = "REAL"
            confidence = float(real_prob)
        else:
            label = "FAKE"
            confidence = float(fake_prob)
            
        # Run LIME explainer
        # LIME perturbs the input text and computes feature importances
        try:
            explanation = self.explainer.explain_instance(
                text, 
                predict_fn, 
                num_features=num_features,
                labels=(1,)  # Explain the "REAL" class (class 1)
            )
            
            # Extract weights for the REAL class
            # Positive weights = supports REAL, Negative weights = supports FAKE
            lime_weights = explanation.as_list(label=1)
            
            influential_words = []
            for word, weight in lime_weights:
                # Classify influence
                influence = "REAL" if weight > 0 else "FAKE"
                influential_words.append({
                    "word": word,
                    "weight": float(weight),
                    "influence": influence
                })
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            influential_words = []
            
        return {
            "label": label,
            "authenticity_score": confidence,
            "top_influential_words": influential_words
        }
