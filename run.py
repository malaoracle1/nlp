#!/usr/bin/env python3
"""
Startup script for the Toxic Comment Classifier web application.
This script initializes the application and loads pre-trained models if available.
"""

import uvicorn
import os
from main import app, classifier

def main():
    print("🛡️ Starting Toxic Comment Classifier...")
    print("=" * 50)

    # Check for required files
    if not os.path.exists("train.csv"):
        print("⚠️  WARNING: train.csv not found!")
        print("   Please download the dataset from:")
        print("   https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data")
        print("   and place train.csv in the project root directory.")
        print()

    # Try to load existing models
    if classifier.load_models():
        print("✅ Pre-trained models loaded successfully!")
    else:
        print("📝 No pre-trained models found. You'll need to train the model first.")

    print()
    print("🚀 Starting web server...")
    print("   Web interface will be available at: http://localhost:8000")
    print("   API documentation: http://localhost:8000/docs")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 50)

    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[".", "templates"]
    )

if __name__ == "__main__":
    main()