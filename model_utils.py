import re
import numpy as np
import pandas as pd
import requests
import csv
from io import StringIO
import logging
from typing import List, Tuple, Dict, Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class ToxicCommentClassifier:
    def __init__(self):
        self.pipeline = None
        self.lemmatizer = None
        self.stop_words = None
        self.contractions = None
        self._initialize_nltk()
        self._load_contractions()

    def _initialize_nltk(self):
        """Download required NLTK data and initialize components"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            logger.info("NLTK components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing NLTK: {e}")
            raise

    def _load_contractions(self):
        """Load contractions dictionary"""
        try:
            url = "https://raw.githubusercontent.com/andrewbury/contractions/refs/heads/master/contractions.json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.contractions = {}
                for key, value in data.items():
                    self.contractions[key] = value[0]
                logger.info("Contractions loaded successfully")
            else:
                logger.warning("Failed to load contractions, using empty dict")
                self.contractions = {}
        except Exception as e:
            logger.warning(f"Error loading contractions: {e}, using empty dict")
            self.contractions = {}

    def basic_clean(self, text: str) -> str:
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        # Remove numbers
        text = re.sub(r'\d+', " ", text)
        # Remove quoted text
        text = re.sub(r'"[^"]*"', " ", text)
        # Remove URLs
        text = re.sub(r"http\S+|www\S+", " ", text)
        # Remove HTML tags
        text = re.sub(r"<.*?>", " ", text)

        # Expand contractions
        for k, v in self.contractions.items():
            text = text.replace(k, v)

        # Remove punctuation except hashtags/mentions
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize_and_lemmatize(self, text: str) -> str:
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        # Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words and t.isalpha()]
        # Lemmatize
        lem = [self.lemmatizer.lemmatize(t) for t in tokens]

        return " ".join(lem)

    def load_bad_words(self) -> List[str]:
        """Load bad words from various sources"""
        bad_words = set()

        # Source 1: CMU bad words list
        try:
            response = requests.get('https://www.cs.cmu.edu/~biglou/resources/bad-words.txt', timeout=10)
            if response.status_code == 200:
                words = response.text.strip().split('\n')
                bad_words.update([word.strip().lower() for word in words if word.strip()])
                logger.info(f"Loaded {len(words)} words from CMU list")
        except Exception as e:
            logger.warning(f"Error loading CMU bad words: {e}")

        # Source 2: GitHub bad words list
        try:
            response = requests.get('https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/refs/heads/master/en', timeout=10)
            if response.status_code == 200:
                words = response.text.strip().split('\n')
                github_words = [word.strip().lower() for word in words if word.strip()]
                bad_words.update(github_words)
                logger.info(f"Loaded {len(github_words)} words from GitHub English list")
        except Exception as e:
            logger.warning(f"Error loading GitHub English bad words: {e}")

        # Source 3: Hinglish profanity list
        try:
            response = requests.get('https://raw.githubusercontent.com/pmathur5k10/Hinglish-Offensive-Text-Classification/refs/heads/main/Hinglish_Profanity_List.csv', timeout=10)
            if response.status_code == 200:
                csv_content = StringIO(response.text)
                csv_reader = csv.reader(csv_content)

                # Skip header row if present
                next(csv_reader, None)

                hinglish_words = []
                for row in csv_reader:
                    if len(row) >= 2 and row[1].strip():
                        word = row[1].strip().lower()
                        if word:
                            hinglish_words.append(word)
                            bad_words.add(word)

                logger.info(f"Loaded {len(hinglish_words)} words from Hinglish CSV")
        except Exception as e:
            logger.warning(f"Error loading Hinglish profanity CSV: {e}")

        # Convert set to list and filter
        final_bad_words = [word for word in bad_words if word and word.strip() and len(word.strip()) > 0]

        return list(set(final_bad_words))

    def create_bad_words_df(self, bad_words_list: List[str]) -> pd.DataFrame:
        """Create dataframe from bad words list"""
        df = pd.DataFrame({'comment_text': bad_words_list})
        df['toxic'] = 1
        return df

    def preprocess_text_pipeline(self, texts: List[str]) -> List[str]:
        """Process a list of texts"""
        processed_texts = []
        for text in texts:
            # Basic cleaning
            cleaned = self.basic_clean(text)
            # Tokenize and lemmatize
            processed = self.tokenize_and_lemmatize(cleaned)
            processed_texts.append(processed)
        return processed_texts

    def train(self, data_path: str, use_augmentation: bool = True) -> Dict[str, Any]:
        """Train the toxic comment classification model"""
        logger.info("Loading training data...")

        # Load data
        df = pd.read_csv(data_path)

        # Create single toxic label
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        df['toxic'] = (df[label_cols].sum(axis=1) > 0).astype(int)

        # Split data
        X = df['comment_text']
        y = df['toxic']
        X_train_orig, X_test, y_train_orig, y_test = train_test_split(
            X, y, test_size=0.2, random_state=25, stratify=y
        )

        logger.info(f"Original training set: {len(X_train_orig)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # Prepare training data
        if use_augmentation:
            logger.info("Loading bad words for augmentation...")
            bad_words_list = self.load_bad_words()
            bad_words_df = self.create_bad_words_df(bad_words_list)

            # Create augmented training set
            X_train_augmented = pd.concat([
                pd.DataFrame({'comment_text': X_train_orig, 'toxic': y_train_orig}),
                bad_words_df
            ], ignore_index=True)

            X_train = X_train_augmented['comment_text']
            y_train = X_train_augmented['toxic']
            logger.info(f"Augmented training set: {len(X_train)} samples")
        else:
            X_train = X_train_orig
            y_train = y_train_orig
            logger.info(f"Using original training set: {len(X_train)} samples")

        # Preprocess texts
        logger.info("Preprocessing training texts...")
        X_train_processed = self.preprocess_text_pipeline(X_train.tolist())
        logger.info("Preprocessing test texts...")
        X_test_processed = self.preprocess_text_pipeline(X_test.tolist())

        # Create and train pipeline
        logger.info("Training model...")
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('nb', MultinomialNB())
        ])

        self.pipeline.fit(X_train_processed, y_train)

        # Evaluate model
        y_pred = self.pipeline.predict(X_test_processed)
        y_pred_proba = self.pipeline.predict_proba(X_test_processed)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': len(y_test)
        }

        logger.info("Model training completed successfully")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1-Score: {f1:.4f}")
        logger.info(f"Test ROC-AUC: {roc_auc:.4f}")

        return metrics

    def predict(self, text: str) -> Tuple[int, float]:
        """Predict if a single text is toxic"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Please train the model first.")

        # Preprocess the text
        processed_text = self.preprocess_text_pipeline([text])

        # Make prediction
        prediction = self.pipeline.predict(processed_text)[0]
        probability = self.pipeline.predict_proba(processed_text)[0][1]  # Probability of toxic class

        return prediction, probability

    def predict_batch(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        """Predict toxicity for multiple texts"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Please train the model first.")

        # Preprocess the texts
        processed_texts = self.preprocess_text_pipeline(texts)

        # Make predictions
        predictions = self.pipeline.predict(processed_texts)
        probabilities = self.pipeline.predict_proba(processed_texts)[:, 1]

        return predictions.tolist(), probabilities.tolist()