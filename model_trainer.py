import re
import numpy as np
import pandas as pd
import requests
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import csv
from io import StringIO

class ToxicCommentClassifier:
    def __init__(self):
        self.model_without_bad_words = None
        self.model_with_bad_words = None
        self.lemmatizer = None
        self.stop_words = None
        self.contractions = None
        self._download_nltk_data()
        self._load_contractions()
        self._initialize_nltk_components()

    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except Exception as e:
            print(f"Error downloading NLTK data: {e}")

    def _load_contractions(self):
        """Load contractions dictionary"""
        try:
            url = "https://raw.githubusercontent.com/andrewbury/contractions/refs/heads/master/contractions.json"
            response = requests.get(url)
            data = response.json()
            self.contractions = {}
            for key, value in data.items():
                self.contractions[key] = value[0]
        except Exception as e:
            print(f"Error loading contractions: {e}")
            self.contractions = {}

    def _initialize_nltk_components(self):
        """Initialize NLTK components"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def basic_clean(self, text):
        """Basic text cleaning function"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'\d+', " ", text)
        text = re.sub(r'"[^"]*"', " ", text)
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"<.*?>", " ", text)

        for k, v in self.contractions.items():
            text = text.replace(k, v)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and t.isalpha()]
        lem = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(lem)

    def load_bad_words(self):
        """Load bad words from various sources"""
        bad_words = set()

        # Source 1: CMU bad words list
        try:
            response = requests.get('https://www.cs.cmu.edu/~biglou/resources/bad-words.txt')
            if response.status_code == 200:
                words = response.text.strip().split('\n')
                bad_words.update([word.strip().lower() for word in words if word.strip()])
                print(f"Loaded {len(words)} words from CMU list")
        except Exception as e:
            print(f"Error loading CMU bad words: {e}")

        # Source 2: GitHub English bad words
        try:
            response = requests.get('https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/refs/heads/master/en')
            if response.status_code == 200:
                words = response.text.strip().split('\n')
                github_words = [word.strip().lower() for word in words if word.strip()]
                bad_words.update(github_words)
                print(f"Loaded {len(github_words)} words from GitHub English list")
        except Exception as e:
            print(f"Error loading GitHub English bad words: {e}")

        # Source 3: Hinglish profanity list
        try:
            response = requests.get('https://raw.githubusercontent.com/pmathur5k10/Hinglish-Offensive-Text-Classification/refs/heads/main/Hinglish_Profanity_List.csv')
            if response.status_code == 200:
                csv_content = StringIO(response.text)
                csv_reader = csv.reader(csv_content)
                next(csv_reader, None)

                hinglish_words = []
                for row in csv_reader:
                    if len(row) >= 2 and row[1].strip():
                        word = row[1].strip().lower()
                        if word:
                            hinglish_words.append(word)
                            bad_words.add(word)

                print(f"Loaded {len(hinglish_words)} words from Hinglish CSV")
        except Exception as e:
            print(f"Error loading Hinglish profanity CSV: {e}")

        final_bad_words = [word for word in bad_words if word and word.strip() and len(word.strip()) > 0]
        return list(set(final_bad_words))

    def create_bad_words_df(self, bad_words_list):
        """Create bad words dataframe"""
        df = pd.DataFrame({'comment_text': bad_words_list})
        df['toxic'] = 1
        return df

    def preprocess_text_pipeline(self, texts):
        """Text preprocessing pipeline"""
        processed_texts = []
        for text in texts:
            cleaned = self.basic_clean(text)
            processed = self.tokenize_and_lemmatize(cleaned)
            processed_texts.append(processed)
        return processed_texts

    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test, model_name):
        """Train and evaluate a model"""
        print(f"\n=== Training {model_name} ===")

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('nb', MultinomialNB())
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")

        return {
            'model': pipeline,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
            }
        }

    def train_models(self):
        """Main training function"""
        print("Starting model training...")

        # Load data
        df = pd.read_csv('train.csv')

        # Create toxic column
        label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
        df['toxic'] = (df[label_cols].sum(axis=1) > 0).astype(int)

        print(f"Dataset shape: {df.shape}")
        print(f"Toxic samples: {df['toxic'].sum()} ({df['toxic'].mean()*100:.1f}%)")

        # Load bad words
        bad_words_list = self.load_bad_words()
        bad_words_df = self.create_bad_words_df(bad_words_list)
        print(f"Bad words dataset: {len(bad_words_df)} samples")

        # Split data
        X = df['comment_text']
        y = df['toxic']
        X_train_orig, X_test, y_train_orig, y_test = train_test_split(
            X, y, test_size=0.2, random_state=25, stratify=y
        )

        # Create augmented training set
        X_train_augmented = pd.concat([
            pd.DataFrame({'comment_text': X_train_orig, 'toxic': y_train_orig}),
            bad_words_df
        ], ignore_index=True)

        print(f"Original training: {len(X_train_orig)} samples")
        print(f"Augmented training: {len(X_train_augmented)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Preprocess text
        print("Preprocessing texts...")
        X_train_orig_processed = self.preprocess_text_pipeline(X_train_orig)
        X_train_augmented_processed = self.preprocess_text_pipeline(X_train_augmented['comment_text'])
        X_test_processed = self.preprocess_text_pipeline(X_test)

        # Train models
        model_1_results = self.train_and_evaluate_model(
            X_train_orig_processed, y_train_orig, X_test_processed, y_test,
            "Naive Bayes (Without Bad Words)"
        )

        model_2_results = self.train_and_evaluate_model(
            X_train_augmented_processed, X_train_augmented['toxic'], X_test_processed, y_test,
            "Naive Bayes (With Bad Words Augmentation)"
        )

        # Save models
        self.model_without_bad_words = model_1_results['model']
        self.model_with_bad_words = model_2_results['model']

        # Save to disk
        with open('models/model_without_bad_words.pkl', 'wb') as f:
            pickle.dump(self.model_without_bad_words, f)

        with open('models/model_with_bad_words.pkl', 'wb') as f:
            pickle.dump(self.model_with_bad_words, f)

        # Determine better model
        better_model = "With Bad Words" if model_2_results['metrics']['f1_score'] > model_1_results['metrics']['f1_score'] else "Without Bad Words"

        results = {
            'training_completed': True,
            'dataset_info': {
                'total_samples': len(df),
                'toxic_samples': int(df['toxic'].sum()),
                'toxic_percentage': float(df['toxic'].mean() * 100),
                'training_samples': len(X_train_orig),
                'augmented_training_samples': len(X_train_augmented),
                'test_samples': len(X_test),
                'bad_words_added': len(bad_words_list)
            },
            'model_without_bad_words': model_1_results['metrics'],
            'model_with_bad_words': model_2_results['metrics'],
            'better_model': better_model,
            'f1_improvement': float(abs(model_2_results['metrics']['f1_score'] - model_1_results['metrics']['f1_score']))
        }

        print(f"\nTraining completed! Better model: {better_model}")
        return results

    def predict(self, text):
        """Make prediction on text"""
        if not self.is_trained():
            raise ValueError("Models not trained yet")

        # Preprocess text
        processed_text = self.preprocess_text_pipeline([text])[0]

        # Use the better model (with bad words augmentation)
        prediction = self.model_with_bad_words.predict([processed_text])[0]
        confidence = self.model_with_bad_words.predict_proba([processed_text])[0]

        return {
            'prediction': 'Toxic' if prediction == 1 else 'Non-toxic',
            'confidence': float(max(confidence)),
            'model_used': 'With Bad Words Augmentation',
            'processed_text': processed_text
        }

    def is_trained(self):
        """Check if models are trained"""
        return self.model_with_bad_words is not None and self.model_without_bad_words is not None

    def load_models(self):
        """Load saved models from disk"""
        try:
            with open('models/model_without_bad_words.pkl', 'rb') as f:
                self.model_without_bad_words = pickle.load(f)

            with open('models/model_with_bad_words.pkl', 'rb') as f:
                self.model_with_bad_words = pickle.load(f)

            return True
        except FileNotFoundError:
            return False