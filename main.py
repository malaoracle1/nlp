from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import joblib
from typing import Dict, Any
import uvicorn
from model_utils import ToxicCommentClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Toxic Comment Classification API",
    description="API for toxic comment classification using Naive Bayes",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

MODEL_PATH = "toxic_model.pkl"
classifier = None

class CommentRequest(BaseModel):
    text: str

class TrainingRequest(BaseModel):
    use_augmentation: bool = True

class PredictionResponse(BaseModel):
    text: str
    prediction: int
    probability: float
    confidence: str

@app.on_event("startup")
async def startup_event():
    """Load model if it exists"""
    global classifier
    if os.path.exists(MODEL_PATH):
        try:
            classifier = joblib.load(MODEL_PATH)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            classifier = None

@app.get("/")
async def root():
    """Serve the main web interface"""
    return FileResponse("index.html")

@app.get("/api/health")
async def health_check():
    """Health check endpoint for API"""
    return {"message": "Toxic Comment Classification API is running"}

@app.get("/model/status")
async def model_status():
    """Check if model exists and is loaded"""
    global classifier
    model_exists = os.path.exists(MODEL_PATH)
    model_loaded = classifier is not None

    return {
        "model_exists": model_exists,
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "status": "ready" if model_loaded else "needs_training"
    }

@app.post("/model/train")
async def train_model(request: TrainingRequest):
    """Train the toxic comment classification model"""
    global classifier

    # Check if training data exists
    if not os.path.exists("train.csv"):
        raise HTTPException(
            status_code=400,
            detail="Training data (train.csv) not found. Please upload the dataset first."
        )

    try:
        logger.info("Starting model training...")
        classifier = ToxicCommentClassifier()

        # Train the model
        metrics = classifier.train(
            data_path="train.csv",
            use_augmentation=request.use_augmentation
        )

        # Save the trained model
        joblib.dump(classifier, MODEL_PATH)
        logger.info("Model trained and saved successfully")

        return {
            "message": "Model trained successfully",
            "metrics": metrics,
            "model_saved": True,
            "augmentation_used": request.use_augmentation
        }

    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_toxicity(request: CommentRequest):
    """Predict if a comment is toxic"""
    global classifier

    # Check if model is loaded
    if classifier is None:
        if os.path.exists(MODEL_PATH):
            try:
                classifier = joblib.load(MODEL_PATH)
                logger.info("Model loaded for prediction")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail="Model exists but failed to load. Please retrain the model."
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Model not found. Please train the model first using /model/train endpoint."
            )

    try:
        # Make prediction
        prediction, probability = classifier.predict(request.text)

        # Determine confidence level
        if probability > 0.8:
            confidence = "high"
        elif probability > 0.6:
            confidence = "medium"
        else:
            confidence = "low"

        return PredictionResponse(
            text=request.text,
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence
        )

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(texts: list[str]):
    """Predict toxicity for multiple comments"""
    global classifier

    if classifier is None:
        raise HTTPException(
            status_code=400,
            detail="Model not found. Please train the model first."
        )

    try:
        results = []
        for text in texts:
            prediction, probability = classifier.predict(text)
            confidence = "high" if probability > 0.8 else "medium" if probability > 0.6 else "low"

            results.append({
                "text": text,
                "prediction": int(prediction),
                "probability": float(probability),
                "confidence": confidence
            })

        return {"predictions": results}

    except Exception as e:
        logger.error(f"Error making batch predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))