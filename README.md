# 🛡️ Toxic Comment Classifier

A web-based toxic comment classification system built with FastAPI and machine learning. This application trains two Naive Bayes models to detect toxic comments, with and without bad words augmentation.

## ✨ Features

- **Interactive Web Interface**: Easy-to-use web UI for model training and text testing
- **Dual Model Training**: Compares models with and without bad words augmentation
- **Real-time Prediction**: Instant toxic comment detection with confidence scores
- **Background Training**: Non-blocking model training with progress tracking
- **Performance Metrics**: Detailed model performance visualization
- **RESTful API**: Complete API for programmatic access

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Internet connection (for downloading bad words lists and NLTK data)

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - Go to [Kaggle Jigsaw Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)
   - Download `train.csv` and place it in the project root directory

4. **Run the application**:
   ```bash
   python run.py
   ```

5. **Open your browser** and navigate to `http://localhost:8000`

## 🎯 Usage

### Training the Model

1. **Ensure you have `train.csv`** in the project root directory
2. **Click "Start Training"** button on the web interface
3. **Wait for completion** - training typically takes several minutes depending on your hardware
4. **View results** - performance metrics will be displayed automatically

### Testing Text

1. **Enter text** in the text area under "Test Text"
2. **Click "Analyze Text"** button
3. **View results** - see if the text is classified as toxic or non-toxic with confidence scores

## 📊 Model Details

### Dataset Processing
- **Original Dataset**: Kaggle Jigsaw Toxic Comment Classification Challenge
- **Label Combination**: Multiple toxic categories combined into binary classification
- **Text Preprocessing**: Cleaning, tokenization, lemmatization, and stopword removal
- **Data Augmentation**: Addition of curated bad words lists from multiple sources

### Models Trained

1. **Baseline Model**: Naive Bayes without augmentation
2. **Augmented Model**: Naive Bayes with bad words augmentation

### Bad Words Sources
- CMU Bad Words List
- GitHub LDNOOBW English List
- Hinglish Profanity List

### Performance Metrics
- Accuracy
- F1-Score
- ROC-AUC
- Precision/Recall
- Confusion Matrix components

## 🌐 API Endpoints

### Web Interface
- `GET /` - Main web interface

### Training
- `POST /train` - Start model training
- `GET /training_status` - Get current training status

### Prediction
- `POST /predict` - Predict toxicity of text
- `GET /model_info` - Get model performance metrics

### Example API Usage

```python
import requests

# Start training
response = requests.post("http://localhost:8000/train")
print(response.json())

# Check training status
status = requests.get("http://localhost:8000/training_status")
print(status.json())

# Make prediction
data = {"text": "Your text here"}
prediction = requests.post("http://localhost:8000/predict", data=data)
print(prediction.json())
```

## 🏗️ Project Structure

```
toxic/
├── main.py                 # FastAPI application
├── model_trainer.py        # Model training and preprocessing logic
├── run.py                  # Startup script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Web interface template
├── models/                # Saved models and training results
└── train.csv             # Dataset (download separately)
```

## 🔧 Configuration

The application uses the following default settings:
- **Server**: `0.0.0.0:8000`
- **Model Type**: Naive Bayes with TF-IDF vectorization
- **Max Features**: 10,000
- **N-gram Range**: (1, 2)
- **Test Split**: 20%

## 📈 Expected Results

Based on the original Jupyter notebook analysis:
- **Accuracy**: ~94.4%
- **F1-Score**: ~63.2% (with augmentation)
- **ROC-AUC**: ~90.5%
- **Training Time**: 5-15 minutes (depending on hardware)

## ⚠️ Important Notes

1. **Dataset Required**: You must download `train.csv` from Kaggle
2. **Internet Connection**: Required for downloading NLTK data and bad words lists
3. **Training Time**: Initial training can take several minutes
4. **Model Persistence**: Trained models are saved and can be reloaded
5. **Background Training**: Training runs in background, web interface remains responsive

## 🐛 Troubleshooting

### Common Issues

1. **"train.csv not found"**
   - Download the dataset from the Kaggle link provided
   - Ensure the file is named exactly `train.csv` in the project root

2. **NLTK download errors**
   - Ensure you have an internet connection
   - The application will automatically download required NLTK data

3. **Training fails**
   - Check that you have enough disk space and memory
   - Verify the dataset file is not corrupted

4. **Port already in use**
   - Change the port in `run.py` or stop other applications using port 8000

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is for educational and research purposes. Please respect the original dataset license terms.