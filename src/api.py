import os
import pickle
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.explainability import NewsExplainer
from src.utils import logger

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
BERT_PATH = os.path.join(MODELS_DIR, "distilbert")
BASELINE_PATH = os.path.join(MODELS_DIR, "baseline_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

# Global state for explainer and model status
explainer = None
model_info = {
    "model_loaded": False,
    "model_type": None,
    "detail": "No model loaded yet."
}

def load_inference_model():
    """Loads the best available model for prediction and initializes the explainer."""
    global explainer, model_info
    
    # 1. Try loading DistilBERT
    if os.path.exists(BERT_PATH) and os.path.exists(os.path.join(BERT_PATH, "config.json")):
        try:
            logger.info("Attempting to load fine-tuned DistilBERT model...")
            tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_PATH)
            model = DistilBertForSequenceClassification.from_pretrained(BERT_PATH)
            explainer = NewsExplainer(
                model=model,
                model_type="bert",
                tokenizer=tokenizer
            )
            model_info = {
                "model_loaded": True,
                "model_type": "DistilBERT",
                "detail": "Fine-tuned transformer model loaded successfully."
            }
            logger.info("DistilBERT model loaded successfully.")
            return
        except Exception as e:
            logger.error(f"Failed to load DistilBERT model: {e}. Falling back to baseline.")
            
    # 2. Try loading Baseline TF-IDF model
    if os.path.exists(BASELINE_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            logger.info("Attempting to load baseline TF-IDF model...")
            with open(BASELINE_PATH, "rb") as f:
                model = pickle.load(f)
            with open(VECTORIZER_PATH, "rb") as f:
                vectorizer = pickle.load(f)
                
            explainer = NewsExplainer(
                model=model,
                model_type="baseline",
                vectorizer=vectorizer
            )
            model_info = {
                "model_loaded": True,
                "model_type": "Baseline TF-IDF",
                "detail": "Baseline machine learning model loaded successfully."
            }
            logger.info("Baseline TF-IDF model loaded successfully.")
            return
        except Exception as e:
            logger.error(f"Failed to load baseline TF-IDF model: {e}")
            
    # 3. No models found
    model_info = {
        "model_loaded": False,
        "model_type": None,
        "detail": "No trained model files found. Please run training scripts first."
    }
    logger.warning("No trained models found on disk.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    load_inference_model()
    yield
    # Shutdown: Clean up if needed
    pass

app = FastAPI(
    title="Fake News Detection API",
    description="A FastAPI backend for classifying news articles and explaining predictions.",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic Schemas
class ArticleRequest(BaseModel):
    text: str = Field(..., min_length=10, description="The raw text of the news article to analyze.")

class InfluentialWord(BaseModel):
    word: str
    weight: float
    influence: str = Field(..., description="Either 'REAL' or 'FAKE' depending on which class it supports.")

class PredictionResponse(BaseModel):
    label: str = Field(..., description="The predicted class: 'REAL' or 'FAKE'")
    authenticity_score: float = Field(..., description="Confidence score / probability of the predicted class")
    top_influential_words: List[InfluentialWord] = Field(..., description="Top words/phrases that contributed to the prediction")
    model_used: str = Field(..., description="The model type used for prediction")

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Returns the health status of the API and model loading details."""
    return {
        "status": "healthy" if model_info["model_loaded"] else "degraded",
        **model_info
    }

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_article(request: ArticleRequest):
    """
    Classifies a news article as REAL or FAKE, returns a confidence score,
    and lists the top influential words that drove the decision.
    """
    if not model_info["model_loaded"]:
        # Attempt reloading in case a training run just completed
        load_inference_model()
        if not model_info["model_loaded"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please run the training pipeline first."
            )
            
    # Clean validation check for empty/blank text
    cleaned_input = request.text.strip()
    if not cleaned_input:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provided text must contain actual news content, not just whitespace."
        )
        
    try:
        # Run explainability pipeline (which also makes the prediction)
        explanation = explainer.explain(request.text, num_features=10)
        
        # Structure the response
        words = [
            InfluentialWord(
                word=w["word"], 
                weight=w["weight"], 
                influence=w["influence"]
            ) 
            for w in explanation["top_influential_words"]
        ]
        
        return PredictionResponse(
            label=explanation["label"],
            authenticity_score=explanation["authenticity_score"],
            top_influential_words=words,
            model_used=model_info["model_type"]
        )
    except Exception as e:
        logger.error(f"Inference error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during model prediction: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
