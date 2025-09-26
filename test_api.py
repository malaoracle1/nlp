#!/usr/bin/env python3
"""
Simple test script for the Toxic Comment Classification API
"""
import sys
import traceback

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")

    try:
        print("[OK] Importing standard library modules...")
        import os
        import logging
        import json

        print("[OK] Testing FastAPI components...")
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel

        print("[OK] Testing sklearn components...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        print("[OK] Testing pandas and numpy...")
        import pandas as pd
        import numpy as np

        print("[OK] Testing NLTK...")
        import nltk

        print("[OK] Testing requests...")
        import requests

        print("[OK] Testing joblib...")
        import joblib

        print("[OK] All imports successful!")
        return True

    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def test_model_utils():
    """Test model_utils functionality"""
    print("\nTesting model_utils...")

    try:
        from model_utils import ToxicCommentClassifier

        print("[OK] ToxicCommentClassifier imported successfully")

        # Initialize classifier
        classifier = ToxicCommentClassifier()
        print("[OK] ToxicCommentClassifier initialized")

        # Test text preprocessing
        test_text = "This is a test comment with URLs http://example.com and numbers 123!"
        cleaned = classifier.basic_clean(test_text)
        print(f"[OK] Text cleaning works: '{test_text}' -> '{cleaned}'")

        # Test tokenization and lemmatization
        processed = classifier.tokenize_and_lemmatize(cleaned)
        print(f"[OK] Tokenization and lemmatization works: '{cleaned}' -> '{processed}'")

        return True

    except Exception as e:
        print(f"[ERROR] Model utils error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def create_sample_data():
    """Create a small sample dataset for testing"""
    print("\nCreating sample training data...")

    try:
        import pandas as pd

        # Create sample data
        sample_data = {
            'comment_text': [
                'This is a nice comment',
                'This is another good comment',
                'Great work everyone!',
                'I love this community',
                'Thank you for sharing this',
                # Add some potentially toxic examples (but keeping them mild for testing)
                'This is bad content',
                'I hate this thing',
                'This is terrible work',
                'Stop doing this',
                'This is wrong and bad'
            ],
            'toxic': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            'severe_toxic': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'obscene': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'threat': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'insult': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            'identity_hate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }

        df = pd.DataFrame(sample_data)
        df.to_csv('train.csv', index=False)

        print(f"[OK] Sample training data created: {len(df)} samples")
        print(f"[OK] Toxic samples: {df['toxic'].sum()}")
        print(f"[OK] Non-toxic samples: {(df['toxic'] == 0).sum()}")

        return True

    except Exception as e:
        print(f"[ERROR] Error creating sample data: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Toxic Comment Classification API Tests ===\n")

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False
        print("\n[WARNING]  Some imports failed. You may need to install dependencies:")
        print("   pip install -r requirements.txt")

    # Test model utilities
    if all_passed and not test_model_utils():
        all_passed = False

    # Create sample data
    if not create_sample_data():
        all_passed = False

    print("\n" + "="*50)
    if all_passed:
        print("[SUCCESS] All tests passed! The API should work correctly.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the API: python main.py")
        print("3. Test training: POST /model/train")
        print("4. Test prediction: POST /predict")
    else:
        print("[FAILED] Some tests failed. Please check the errors above.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)