#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Execute file for Credit Card Fraud Detection Project.
This script first runs the preprocessing step to download and prepare the data,
then trains and evaluates the CNN model.
"""

import os
import numpy as np
import time
from pathlib import Path

# Import preprocessing module
from preprocess_data.preprocess import CreditCardFraudPreprocessor

# Import CNN model module
from model.cnn_model import CreditCardFraudCNN

# Set the paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
PREPROCESSED_DATA_DIR = OUTPUT_DIR / "preprocessed_data"

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

def save_preprocessed_data(X_train, X_test, y_train, y_test):
    """Save preprocessed data to files"""
    print("\nSaving preprocessed data...")
    np.save(PREPROCESSED_DATA_DIR / "X_train.npy", X_train)
    np.save(PREPROCESSED_DATA_DIR / "X_test.npy", X_test)
    np.save(PREPROCESSED_DATA_DIR / "y_train.npy", y_train)
    np.save(PREPROCESSED_DATA_DIR / "y_test.npy", y_test)
    
    with open(PREPROCESSED_DATA_DIR / "data_info.txt", "w") as f:
        f.write(f"X_train shape: {X_train.shape}\n")
        f.write(f"X_test shape: {X_test.shape}\n")
        f.write(f"y_train shape: {y_train.shape}\n")
        f.write(f"y_test shape: {y_test.shape}\n")
        f.write(f"Class distribution in training set: {np.bincount(y_train)}\n")
        f.write(f"Class distribution in test set: {np.bincount(y_test)}\n")
    
    print(f"Preprocessed data saved to {PREPROCESSED_DATA_DIR}")

def load_preprocessed_data():
    """Load preprocessed data from files"""
    print("\nLoading preprocessed data...")
    try:
        X_train = np.load(PREPROCESSED_DATA_DIR / "X_train.npy")
        X_test = np.load(PREPROCESSED_DATA_DIR / "X_test.npy")
        y_train = np.load(PREPROCESSED_DATA_DIR / "y_train.npy")
        y_test = np.load(PREPROCESSED_DATA_DIR / "y_test.npy")
        
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("Error: Preprocessed data not found at expected location.")
        return None, None, None, None

def preprocess_data(force_reprocess=False):
    """Preprocess the credit card fraud data"""
    if not force_reprocess and (PREPROCESSED_DATA_DIR / "X_train.npy").exists():
        print("Preprocessed data already exists. Loading from files...")
        return load_preprocessed_data()
    
    print("\n" + "="*50)
    print("Starting data preprocessing...")
    print("="*50)
    
    start_time = time.time()
    
    preprocessor = CreditCardFraudPreprocessor(
        download_data=True,
        save_plots=True
    )
    
    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    save_preprocessed_data(X_train, X_test, y_train, y_test)
    
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    return X_train, X_test, y_train, y_test

def train_and_evaluate_cnn_model(X_train, X_test, y_train, y_test, force_retrain=False):
    """Train and evaluate CNN model"""
    print("\n" + "="*50)
    print("Starting CNN model training and evaluation...")
    print("="*50)
    
    # Train and evaluate Neural Network
    nn_output_dir = OUTPUT_DIR / "cnn_model"
    os.makedirs(nn_output_dir, exist_ok=True)
    
    nn_model = CreditCardFraudCNN(output_dir=str(nn_output_dir), validation_split=0.2)
    
    if not force_retrain and (nn_output_dir / "best_cnn_fraud_model.h5").exists():
        print(f"Loading existing CNN model from {nn_output_dir}")
        nn_model.load_model("best_cnn_fraud_model.h5")
    else:
        print("Training CNN model...")
        nn_model.train_model(X_train, y_train, epochs=100, batch_size=32, verbose=1)
        nn_model.plot_training_history()
    
    print("\nEvaluating CNN model...")
    nn_metrics = nn_model.evaluate_model(X_test, y_test)
    
    return nn_model, nn_metrics

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION - WITH PREPROCESSING")
    print("="*70)
    
    # First, preprocess the data (or load if already preprocessed)
    X_train, X_test, y_train, y_test = preprocess_data(force_reprocess=False)
    
    if X_train is None:
        print("Error: Could not get preprocessed data. Exiting.")
        return
    
    # Train and evaluate CNN model
    nn_model, nn_metrics = train_and_evaluate_cnn_model(
        X_train, X_test, y_train, y_test, force_retrain=False
    )
    
    # Print final evaluation metrics
    print("\n" + "="*50)
    print("CNN MODEL PERFORMANCE")
    print("="*50)
    
    print(f"Accuracy:  {nn_metrics['accuracy']:.4f}")
    print(f"Precision: {nn_metrics['precision']:.4f}")
    print(f"Recall:    {nn_metrics['recall']:.4f}")
    print(f"F1 Score:  {nn_metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {nn_metrics['roc_auc']:.4f}")
    print(f"PR AUC:    {nn_metrics['pr_auc']:.4f}")

if __name__ == "__main__":
    main()
