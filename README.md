# Toxic Comment Classification API

A FastAPI-based web service for detecting toxic comments using machine learning.

## Features

- **Model Training**: Train a Naive Bayes classifier with optional bad word augmentation
- **Toxicity Prediction**: Classify individual comments as toxic or non-toxic
- **Batch Prediction**: Process multiple comments at once
- **Model Management**: Check model status and retrain as needed
- **Cloud Ready**: Optimized for deployment on Render

## Setup

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare training data:**
   - Download the Kaggle Toxic Comment Classification dataset
   - Place `train.csv` in the project directory

3. **Run the application:**
   ```bash
   python main.py
   ```

   Or use uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access the API:**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

### Render Deployment

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Use the following settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.11

## API Endpoints

### Health Check
```
GET /
```
Returns API status.

### Model Status
```
GET /model/status
```
Check if model exists and is loaded.

**Response:**
```json
{
  "model_exists": true,
  "model_loaded": true,
  "model_path": "toxic_model.pkl",
  "status": "ready"
}
```

### Train Model
```
POST /model/train
```

**Request Body:**
```json
{
  "use_augmentation": true
}
```

**Response:**
```json
{
  "message": "Model trained successfully",
  "metrics": {
    "accuracy": 0.9437,
    "f1_score": 0.6320,
    "roc_auc": 0.9047,
    "true_positives": 1544,
    "true_negatives": 28573,
    "false_positives": 97,
    "false_negatives": 1701
  },
  "model_saved": true,
  "augmentation_used": true
}
```

### Predict Single Comment
```
POST /predict
```

**Request Body:**
```json
{
  "text": "This is a sample comment to analyze"
}
```

**Response:**
```json
{
  "text": "This is a sample comment to analyze",
  "prediction": 0,
  "probability": 0.15,
  "confidence": "low"
}
```

### Batch Prediction
```
POST /predict/batch
```

**Request Body:**
```json
[
  "This is comment 1",
  "This is comment 2",
  "This is comment 3"
]
```

**Response:**
```json
{
  "predictions": [
    {
      "text": "This is comment 1",
      "prediction": 0,
      "probability": 0.12,
      "confidence": "low"
    },
    {
      "text": "This is comment 2",
      "prediction": 1,
      "probability": 0.89,
      "confidence": "high"
    },
    {
      "text": "This is comment 3",
      "prediction": 0,
      "probability": 0.34,
      "confidence": "low"
    }
  ]
}
```

## Model Details

- **Algorithm**: Multinomial Naive Bayes with TF-IDF vectorization
- **Features**: Up to 10,000 TF-IDF features with 1-2 gram range
- **Preprocessing**: Text cleaning, tokenization, lemmatization, stop word removal
- **Augmentation**: Optional bad word augmentation from multiple sources
- **Performance**: ~94% accuracy, ~63% F1-score on test set

## Usage Flow

1. **First Time Setup:**
   ```bash
   curl -X POST "http://localhost:8000/model/train" \
        -H "Content-Type: application/json" \
        -d '{"use_augmentation": true}'
   ```

2. **Check Model Status:**
   ```bash
   curl -X GET "http://localhost:8000/model/status"
   ```

3. **Make Predictions:**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"text": "Your comment here"}'
   ```

## Error Handling

- **400**: Bad request (missing data, invalid input)
- **404**: Model not found (need to train first)
- **500**: Internal server error (training/prediction failure)

## Confidence Levels

- **High**: Probability > 0.8
- **Medium**: Probability 0.6-0.8
- **Low**: Probability < 0.6