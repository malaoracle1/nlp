from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import os
import json
from model_trainer_simple import SimpleToxicCommentClassifier

app = FastAPI(title="Toxic Comment Classifier", version="1.0.0")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Create directories if they don't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Global classifier instance
classifier = SimpleToxicCommentClassifier()

# Training status
training_status = {"is_training": False, "progress": "", "completed": False, "error": None}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "training_status": training_status})

@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    global training_status

    if training_status["is_training"]:
        return JSONResponse({"error": "Training already in progress"}, status_code=400)

    # Reset training status
    training_status = {"is_training": True, "progress": "Starting training...", "completed": False, "error": None}

    # Start training in background
    background_tasks.add_task(run_training)

    return JSONResponse({"message": "Training started successfully"})

async def run_training():
    global training_status
    try:
        training_status["progress"] = "Training simple model..."

        results = classifier.train_simple_model()

        if "error" in results:
            training_status["error"] = results["error"]
            training_status["is_training"] = False
            return

        training_status["progress"] = "Training completed!"
        training_status["completed"] = True
        training_status["is_training"] = False

        # Save results
        with open("models/training_results.json", "w") as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        training_status["error"] = str(e)
        training_status["is_training"] = False
        training_status["completed"] = False

@app.get("/training_status")
async def get_training_status():
    return JSONResponse(training_status)

@app.post("/predict")
async def predict_text(text: str = Form(...)):
    try:
        if not classifier.is_trained():
            # Try to load existing model
            if not classifier.load_model():
                return JSONResponse({"error": "Model not trained yet. Please train the model first."}, status_code=400)

        prediction = classifier.predict(text)
        return JSONResponse({
            "text": text,
            "prediction": prediction["prediction"],
            "confidence": prediction["confidence"],
            "model_used": prediction["model_used"]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/model_info")
async def get_model_info():
    if not classifier.is_trained():
        # Try to load existing model
        if not classifier.load_model():
            return JSONResponse({"error": "Model not trained yet"}, status_code=400)

    # Load training results if available
    results_file = "models/training_results.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
        return JSONResponse(results)

    return JSONResponse({
        "training_completed": True,
        "model_type": "Simple Naive Bayes",
        "message": "Model loaded but detailed results not available"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)