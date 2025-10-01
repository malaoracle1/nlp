from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import secrets
import re
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from io import StringIO
import csv
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import threading
import pickle
import os

app = FastAPI(title="Toxic Comment Classifier")
security = HTTPBasic()

# Global variables
models = {}
model_status = {
    "status": "not_ready",
    "message": "Models not generated yet"
}
model_lock = threading.Lock()

# Model storage directory
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Authentication
USERNAME = "admin"
PASSWORD = "admin123"

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Request/Response models
class TestRequest(BaseModel):
    text: str

class TestResponse(BaseModel):
    results: dict

# Download NLTK data
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Preprocessing functions
contractions = {}

def load_contractions():
    global contractions
    try:
        url = "https://raw.githubusercontent.com/andrewbury/contractions/refs/heads/master/contractions.json"
        response = requests.get(url, timeout=10)
        data = response.json()
        contractions = {key: value[0] for key, value in data.items()}
    except:
        contractions = {}

def basic_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', " ", text)
    text = re.sub(r'"[^"]*"', " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    for k, v in contractions.items():
        text = text.replace(k, v)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_lemmatize(text, lemmatizer, stop_words):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and t.isalpha()]
    lem = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lem)

def preprocess_text_pipeline(texts, lemmatizer, stop_words):
    processed_texts = []
    for text in texts:
        cleaned = basic_clean(text)
        processed = tokenize_and_lemmatize(cleaned, lemmatizer, stop_words)
        processed_texts.append(processed)
    return processed_texts

def load_bad_words():
    bad_words = set()
    try:
        ds = load_dataset("textdetox/multilingual_toxic_lexicon")
        en_lexicon = ds["en"]
        en_words_list = [word.strip().lower() for word in en_lexicon["text"] if word.strip()]
        bad_words.update(en_words_list)
    except:
        pass

    sources = [
        'https://www.cs.cmu.edu/~biglou/resources/bad-words.txt',
        'https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/refs/heads/master/en',
        'https://raw.githubusercontent.com/Orthrus-Lexicon/Toxic/refs/heads/main/Toxic%20words%20dictionary.txt'
    ]

    for url in sources:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                words = response.text.strip().split('\n')
                bad_words.update([word.strip().lower() for word in words if word.strip()])
        except:
            pass

    try:
        response = requests.get('https://raw.githubusercontent.com/pmathur5k10/Hinglish-Offensive-Text-Classification/refs/heads/main/Hinglish_Profanity_List.csv', timeout=10)
        if response.status_code == 200:
            csv_content = StringIO(response.text)
            csv_reader = csv.reader(csv_content)
            next(csv_reader, None)
            for row in csv_reader:
                if len(row) >= 2 and row[1].strip():
                    bad_words.add(row[1].strip().lower())
    except:
        pass

    return list(set([word for word in bad_words if word and word.strip() and len(word.strip()) > 0]))

def build_top_vocab(texts, max_features=10000):
    temp_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    temp_vectorizer.fit(texts)
    vocab = set(temp_vectorizer.get_feature_names_out())
    return vocab

def build_combined_vocab(train_texts, bad_words_list, lemmatizer, stop_words, max_features=10000):
    top_vocab = build_top_vocab(train_texts, max_features=max_features)
    bad_vocab = preprocess_text_pipeline(bad_words_list, lemmatizer, stop_words)
    combined_vocab = top_vocab.union(bad_vocab)
    return list(combined_vocab)

def save_models():
    """Save all trained models to disk"""
    for model_key, model_data in models.items():
        model_path = os.path.join(MODEL_DIR, f"{model_key}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    print(f"Models saved to {MODEL_DIR} directory")

def load_models():
    """Load models from disk if they exist"""
    global models, model_status

    model_files = ['model_1.pkl', 'model_2.pkl', 'model_3.pkl', 'model_4.pkl']
    all_exist = all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in model_files)

    if not all_exist:
        return False

    try:
        for model_file in model_files:
            model_key = model_file.replace('.pkl', '')
            model_path = os.path.join(MODEL_DIR, model_file)
            with open(model_path, 'rb') as f:
                models[model_key] = pickle.load(f)

        with model_lock:
            model_status["status"] = "ready"
            model_status["message"] = "Models loaded from disk"
        print("Models loaded successfully from disk")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def train_models_background():
    global models, model_status

    try:
        with model_lock:
            model_status["status"] = "training"
            model_status["message"] = "Loading data and preprocessing..."

        # Check if train.csv exists
        if not os.path.exists('train.csv'):
            with model_lock:
                model_status["status"] = "error"
                model_status["message"] = "train.csv file not found. Please place the file in the same directory."
            return

        # Load data
        df = pd.read_csv('train.csv')
        label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
        df['toxic'] = (df[label_cols].sum(axis=1) > 0).astype(int)

        # Load contractions
        load_contractions()

        # Initialize NLTK components
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        # Load bad words
        bad_words_list = load_bad_words()

        # Create bad words dataframe
        bad_words_df = pd.DataFrame({'comment_text': bad_words_list, 'toxic': 1})

        # Split data
        X = df['comment_text']
        y = df['toxic']
        X_train_orig, X_test, y_train_orig, y_test = train_test_split(
            X, y, test_size=0.2, random_state=10, stratify=y
        )

        # Create combined training set
        X_train_combined = pd.concat([
            pd.DataFrame({'comment_text': X_train_orig, 'toxic': y_train_orig}),
            bad_words_df
        ], ignore_index=True)

        # Preprocess
        with model_lock:
            model_status["message"] = "Preprocessing text data..."

        X_train_orig_processed = preprocess_text_pipeline(X_train_orig, lemmatizer, stop_words)
        X_train_combined_processed = preprocess_text_pipeline(X_train_combined['comment_text'], lemmatizer, stop_words)

        # Train Model 1
        with model_lock:
            model_status["message"] = "Training Model 1: Multinomial Naive Bayes (TF-IDF)..."

        model_1 = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('nb', MultinomialNB())
        ])
        model_1.fit(X_train_orig_processed, y_train_orig)
        models['model_1'] = {
            'pipeline': model_1,
            'name': 'Multinomial Naive Bayes (TF-IDF)',
            'lemmatizer': lemmatizer,
            'stop_words': stop_words
        }

        # Train Model 2
        with model_lock:
            model_status["message"] = "Training Model 2: Multinomial Naive Bayes (CountVectorizer + Lexicon)..."

        combined_vocab = build_combined_vocab(X_train_combined_processed, bad_words_list, lemmatizer, stop_words, max_features=10000)
        model_2 = Pipeline([
            ('countvec', CountVectorizer(vocabulary=combined_vocab, ngram_range=(1, 2))),
            ('nb', MultinomialNB())
        ])
        model_2.fit(X_train_combined_processed, X_train_combined['toxic'])
        models['model_2'] = {
            'pipeline': model_2,
            'name': 'Multinomial Naive Bayes (CountVectorizer + Lexicon)',
            'lemmatizer': lemmatizer,
            'stop_words': stop_words
        }

        # Train Model 3
        with model_lock:
            model_status["message"] = "Training Model 3: Logistic Regression (TF-IDF)..."

        model_3 = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('lr', LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'))
        ])
        model_3.fit(X_train_orig_processed, y_train_orig)
        models['model_3'] = {
            'pipeline': model_3,
            'name': 'Logistic Regression (TF-IDF)',
            'lemmatizer': lemmatizer,
            'stop_words': stop_words
        }

        # Train Model 4
        with model_lock:
            model_status["message"] = "Training Model 4: Logistic Regression (CountVectorizer + Lexicon)..."

        model_4 = Pipeline([
            ('countvec', CountVectorizer(vocabulary=combined_vocab, ngram_range=(1, 2))),
            ('lr', LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'))
        ])
        model_4.fit(X_train_combined_processed, X_train_combined['toxic'])
        models['model_4'] = {
            'pipeline': model_4,
            'name': 'Logistic Regression (CountVectorizer + Lexicon)',
            'lemmatizer': lemmatizer,
            'stop_words': stop_words
        }

        # Save models to disk
        save_models()

        with model_lock:
            model_status["status"] = "ready"
            model_status["message"] = "All 4 models trained and saved successfully"

    except Exception as e:
        with model_lock:
            model_status["status"] = "error"
            model_status["message"] = f"Error training models: {str(e)}"

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Toxic Comment Classifier</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .status-section {
                margin: 20px 0;
                padding: 15px;
                border-radius: 5px;
                background-color: #f0f0f0;
            }
            .status-ready {
                background-color: #d4edda;
                color: #155724;
            }
            .status-training {
                background-color: #fff3cd;
                color: #856404;
            }
            .status-error {
                background-color: #f8d7da;
                color: #721c24;
            }
            .status-not-ready {
                background-color: #d1ecf1;
                color: #0c5460;
            }
            button {
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #0056b3;
            }
            button:disabled {
                background-color: #6c757d;
                cursor: not-allowed;
            }
            .test-section {
                margin-top: 30px;
            }
            textarea {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
                box-sizing: border-box;
            }
            .results {
                margin-top: 20px;
            }
            .model-result {
                margin: 10px 0;
                padding: 15px;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
            .toxic {
                background-color: #f8d7da;
            }
            .non-toxic {
                background-color: #d4edda;
            }
            .model-name {
                font-weight: bold;
                margin-bottom: 5px;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ Toxic Comment Classifier</h1>

            <div class="status-section" id="statusSection">
                <h3>Model Status</h3>
                <p id="statusMessage">Loading...</p>
            </div>

            <div>
                <button id="generateBtn" onclick="generateModels()">Generate Models</button>
            </div>

            <div class="test-section">
                <h3>Test Models</h3>
                <textarea id="testText" rows="5" placeholder="Enter text to classify..."></textarea>
                <br><br>
                <button id="testBtn" onclick="testModels()" disabled>Test</button>
                <div class="loading" id="loading">Processing...</div>
                <div class="results" id="results"></div>
            </div>
        </div>

        <script>
            let checkStatusInterval;

            async function checkStatus() {
                try {
                    const response = await fetch('/status');
                    const data = await response.json();

                    const statusSection = document.getElementById('statusSection');
                    const statusMessage = document.getElementById('statusMessage');
                    const generateBtn = document.getElementById('generateBtn');
                    const testBtn = document.getElementById('testBtn');

                    statusMessage.textContent = data.message;

                    statusSection.className = 'status-section status-' + data.status.replace('_', '-');

                    if (data.status === 'ready') {
                        generateBtn.disabled = true;
                        testBtn.disabled = false;
                        if (checkStatusInterval) {
                            clearInterval(checkStatusInterval);
                        }
                    } else if (data.status === 'training') {
                        generateBtn.disabled = true;
                        testBtn.disabled = true;
                    } else {
                        generateBtn.disabled = false;
                        testBtn.disabled = true;
                    }
                } catch (error) {
                    console.error('Error checking status:', error);
                }
            }

            async function generateModels() {
                const username = prompt('Username:');
                const password = prompt('Password:');

                if (!username || !password) {
                    alert('Username and password are required');
                    return;
                }

                const credentials = btoa(username + ':' + password);

                try {
                    const response = await fetch('/generate-models', {
                        method: 'POST',
                        headers: {
                            'Authorization': 'Basic ' + credentials
                        }
                    });

                    if (response.status === 401) {
                        alert('Invalid username or password');
                        return;
                    }

                    const data = await response.json();
                    alert(data.message);

                    checkStatusInterval = setInterval(checkStatus, 2000);
                    checkStatus();
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }

            async function testModels() {
                const text = document.getElementById('testText').value;

                if (!text.trim()) {
                    alert('Please enter text to test');
                    return;
                }

                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').innerHTML = '';

                try {
                    const response = await fetch('/test', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ text: text })
                    });

                    const data = await response.json();

                    document.getElementById('loading').style.display = 'none';

                    let resultsHtml = '';
                    for (const [modelKey, result] of Object.entries(data.results)) {
                        const toxicClass = result.prediction === 'TOXIC' ? 'toxic' : 'non-toxic';
                        resultsHtml += `
                            <div class="model-result ${toxicClass}">
                                <div class="model-name">${result.model_name}</div>
                                <div>Prediction: <strong>${result.prediction}</strong></div>
                                <div>Confidence: ${result.probability.toFixed(2)}%</div>
                            </div>
                        `;
                    }

                    document.getElementById('results').innerHTML = resultsHtml;
                } catch (error) {
                    document.getElementById('loading').style.display = 'none';
                    alert('Error: ' + error.message);
                }
            }

            checkStatus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/status")
async def get_status():
    with model_lock:
        return model_status

@app.post("/generate-models")
async def generate_models(username: str = Depends(verify_credentials)):
    with model_lock:
        if model_status["status"] == "training":
            raise HTTPException(status_code=400, detail="Models are already being trained")
        if model_status["status"] == "ready":
            raise HTTPException(status_code=400, detail="Models are already trained")

    thread = threading.Thread(target=train_models_background)
    thread.start()

    return {"message": "Model training started"}

@app.post("/test", response_model=TestResponse)
async def test_text(request: TestRequest):
    with model_lock:
        if model_status["status"] != "ready":
            raise HTTPException(status_code=400, detail="Models are not ready yet")

    results = {}

    for model_key, model_data in models.items():
        pipeline = model_data['pipeline']
        lemmatizer = model_data['lemmatizer']
        stop_words = model_data['stop_words']

        # Preprocess
        processed_text = preprocess_text_pipeline([request.text], lemmatizer, stop_words)

        # Predict
        prediction = pipeline.predict(processed_text)[0]
        probability = pipeline.predict_proba(processed_text)[0]

        results[model_key] = {
            'model_name': model_data['name'],
            'prediction': 'TOXIC' if prediction == 1 else 'NON-TOXIC',
            'probability': float(probability[1] * 100)
        }

    return TestResponse(results=results)

@app.on_event("startup")
async def startup_event():
    """Load models on startup if they exist"""
    load_models()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
