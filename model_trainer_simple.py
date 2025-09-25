import re
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import requests

class SimpleToxicCommentClassifier:
    def __init__(self):
        self.model = None
        self.contractions = {}
        self._load_contractions()

    def _load_contractions(self):
        """Load basic contractions"""
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }

    def basic_clean(self, text):
        """Basic text cleaning"""
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

    def simple_preprocess(self, texts):
        """Simple preprocessing without NLTK"""
        processed = []
        for text in texts:
            cleaned = self.basic_clean(text)
            # Simple word filtering - remove common stop words
            words = cleaned.split()
            # Basic stopwords
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'}
            filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
            processed.append(' '.join(filtered_words))
        return processed

    def load_bad_words(self):
        """Load a simple bad words list"""
        bad_words = []
        try:
            # Simple predefined list as fallback
            predefined_bad_words = [
                "hate", "stupid", "idiot", "moron", "dumb", "loser", "shut up",
                "kill yourself", "go die", "worthless", "pathetic", "disgusting"
            ]
            bad_words.extend(predefined_bad_words)
            print(f"Using predefined bad words: {len(bad_words)} words")
        except Exception as e:
            print(f"Using minimal bad words list: {e}")
            bad_words = ["hate", "stupid", "kill", "die"]

        return bad_words

    def create_bad_words_data(self, bad_words_list):
        """Create simple bad words dataset"""
        return [(word, 1) for word in bad_words_list]

    def train_simple_model(self):
        """Train a simple model without complex dependencies"""
        try:
            # Load from CSV if it exists
            if os.path.exists("train.csv"):
                print("Loading data from train.csv...")
                texts = []
                labels = []

                # Simple CSV reading using built-in csv module
                import csv
                with open("train.csv", 'r', encoding='utf-8') as f:
                    csv_reader = csv.DictReader(f)

                    print(f"CSV columns: {csv_reader.fieldnames}")

                    # Find the comment column
                    comment_col = None
                    for col in csv_reader.fieldnames:
                        if 'comment_text' in col.lower():
                            comment_col = col
                            break

                    if comment_col is None:
                        raise ValueError("Could not find comment_text column")

                    # Toxic columns
                    toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

                    row_count = 0
                    max_rows = 5000  # Process fewer rows for speed and memory

                    for row in csv_reader:
                        if row_count >= max_rows:
                            break

                        try:
                            comment = row.get(comment_col, '').strip()
                            if len(comment) > 10:  # Skip very short comments
                                # Check if any toxic column is 1
                                is_toxic = 0
                                for toxic_col in toxic_cols:
                                    if row.get(toxic_col, '0').strip() == '1':
                                        is_toxic = 1
                                        break

                                texts.append(comment)
                                labels.append(is_toxic)
                                row_count += 1

                                if row_count % 1000 == 0:
                                    print(f"Processed {row_count} samples...")

                        except Exception as e:
                            continue  # Skip problematic rows

                print(f"Loaded {len(texts)} samples from CSV")

                # Add bad words to augment data
                bad_words = self.load_bad_words()
                bad_words_data = self.create_bad_words_data(bad_words)
                for word, label in bad_words_data:
                    texts.append(word)
                    labels.append(label)

            else:
                print("Creating sample dataset...")
                sample_texts = [
                    ("This is a good comment", 0),
                    ("I like this post", 0),
                    ("Great work!", 0),
                    ("Thank you for sharing", 0),
                    ("This is helpful", 0),
                    ("You are stupid", 1),
                    ("I hate this", 1),
                    ("This is garbage", 1),
                    ("Go kill yourself", 1),
                    ("You're an idiot", 1)
                ]

                # Add bad words
                bad_words = self.load_bad_words()
                bad_words_data = self.create_bad_words_data(bad_words)
                sample_texts.extend(bad_words_data)

                texts, labels = zip(*sample_texts)

            print(f"Training with {len(texts)} samples")

            # Simple preprocessing
            processed_texts = self.simple_preprocess(texts)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                processed_texts, labels, test_size=0.2, random_state=42
            )

            # Create and train model
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
                ('nb', MultinomialNB())
            ])

            self.model.fit(X_train, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Save model
            os.makedirs("models", exist_ok=True)
            with open('models/simple_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)

            results = {
                'training_completed': True,
                'dataset_info': {
                    'total_samples': len(texts),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test)
                },
                'model_performance': {
                    'accuracy': float(accuracy),
                    'f1_score': float(f1)
                },
                'model_type': 'Simple Naive Bayes'
            }

            print(f"Training completed! Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
            return results

        except Exception as e:
            print(f"Training error: {e}")
            return {"error": str(e)}

    def predict(self, text):
        """Make prediction"""
        if not self.is_trained():
            raise ValueError("Model not trained yet")

        processed_text = self.simple_preprocess([text])[0]
        prediction = self.model.predict([processed_text])[0]
        confidence = self.model.predict_proba([processed_text])[0]

        return {
            'prediction': 'Toxic' if prediction == 1 else 'Non-toxic',
            'confidence': float(max(confidence)),
            'model_used': 'Simple Naive Bayes'
        }

    def is_trained(self):
        """Check if model is trained"""
        return self.model is not None

    def load_model(self):
        """Load saved model"""
        try:
            with open('models/simple_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            return True
        except FileNotFoundError:
            return False