import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    precision_score, 
    recall_score, 
    f1_score,
    accuracy_score
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
import os
import time
import joblib

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CreditCardFraudCNN:
    """
    Advanced Neural Network for Credit Card Fraud Detection with enhanced visualizations
    """
    
    def __init__(self, output_dir='outputs/cnn_model', validation_split=0.2):
        """
        Initialize the fraud detection model.
        
        Args:
            output_dir: Directory to save model outputs
            validation_split: Proportion of training data to use for validation
        """
        self.output_dir = output_dir
        self.validation_split = validation_split
        self.model = None
        self.history = None
        self.best_threshold = 0.5
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def build_model(self, input_shape, learning_rate=0.001):
        """
        Build a neural network model
        
        Args:
            input_shape: Shape of input features
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=input_shape, name='input_layer')
        
        # Feature extraction branch
        x = Dense(128, activation='relu', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Removed problematic attention mechanism
        
        x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid', name='output_layer')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='fraud_detection_nn')
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.AUC(name='pr_auc', curve='PR')
            ]
        )
        
        return model
    
    def train_model(self, X_train, y_train, epochs=100, batch_size=32, verbose=1):
        """
        Train the neural network model
        
        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity mode
        """
        print("\nTraining Neural Network model...")
        start_time = time.time()
        
        # Build model
        self.model = self.build_model(input_shape=(X_train.shape[1],))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_pr_auc', patience=10, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_pr_auc', factor=0.5, patience=5, min_lr=1e-6, mode='max'),
            ModelCheckpoint(
                filepath=os.path.join(self.output_dir, 'best_cnn_fraud_model.h5'),
                monitor='val_pr_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=self.validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save the final model
        self.save_model('final_cnn_fraud_model.h5')
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation with threshold analysis"""
        print("\nEvaluating Neural Network model...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test).flatten()
        
        # Find optimal threshold
        self.best_threshold = self._find_optimal_threshold(y_test, y_pred_proba)
        y_pred = (y_pred_proba > self.best_threshold).astype(int)
        
        # Generate evaluation metrics
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Generate visualizations
        self._plot_roc_curve(fpr, tpr, roc_auc)
        self._plot_pr_curve(precision, recall, pr_auc, np.mean(y_test))
        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_probability_distribution(y_test, y_pred_proba)
        
        # Print and return metrics
        print("\nOptimal Threshold:", self.best_threshold)
        print(classification_report(y_test, y_pred))
        
        return {
            'best_threshold': self.best_threshold,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def _find_optimal_threshold(self, y_true, y_pred_proba, thresholds=None):
        """Find optimal threshold based on F1 score"""
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 50)
        
        best_threshold = 0.5
        best_f1 = 0
        
        for thresh in thresholds:
            y_pred = (y_pred_proba > thresh).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
                
        return best_threshold
    
    def plot_training_history(self):
        """Plot training metrics visualization"""
        if self.history is None:
            print("No training history available.")
            return
        
        # Plot training history
        plt.figure(figsize=(15, 10))
        
        # Accuracy plot
        plt.subplot(2, 3, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Loss plot
        plt.subplot(2, 3, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Precision plot
        plt.subplot(2, 3, 3)
        plt.plot(self.history.history['precision'], label='Train Precision')
        plt.plot(self.history.history['val_precision'], label='Validation Precision')
        plt.title('Model Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Recall plot
        plt.subplot(2, 3, 4)
        plt.plot(self.history.history['recall'], label='Train Recall')
        plt.plot(self.history.history['val_recall'], label='Validation Recall')
        plt.title('Model Recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend()
        
        # AUC plot
        plt.subplot(2, 3, 5)
        plt.plot(self.history.history['auc'], label='Train AUC')
        plt.plot(self.history.history['val_auc'], label='Validation AUC')
        plt.title('Model ROC AUC')
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        plt.legend()
        
        # PR AUC plot
        plt.subplot(2, 3, 6)
        plt.plot(self.history.history['pr_auc'], label='Train PR AUC')
        plt.plot(self.history.history['val_pr_auc'], label='Validation PR AUC')
        plt.title('Model PR AUC')
        plt.ylabel('PR AUC')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=300)
        plt.close()
    
    def _plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot ROC curve"""
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic', fontsize=16)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'), dpi=300)
        plt.close()
    
    def _plot_pr_curve(self, precision, recall, pr_auc, prevalence):
        """Plot Precision-Recall curve"""
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.plot([0, 1], [prevalence, prevalence], color='red', linestyle='--', 
                 label=f'Baseline ({prevalence:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve', fontsize=16)
        plt.legend(loc='lower left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pr_curve.png'), dpi=300)
        plt.close()
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Fraud', 'Fraud'],
                    yticklabels=['Non-Fraud', 'Fraud'])
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)
        plt.title('Confusion Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
    
    def _plot_probability_distribution(self, y_true, y_pred_proba):
        """Plot probability distribution by class"""
        plt.figure(figsize=(10, 6))
        for label in [0, 1]:
            sns.kdeplot(y_pred_proba[y_true == label], label=f'Class {label}')
        plt.axvline(x=self.best_threshold, color='k', linestyle='--', label=f'Threshold ({self.best_threshold:.2f})')
        plt.xlabel('Predicted Probability', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('Probability Distribution by Class', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'probability_distribution.png'), dpi=300)
        plt.close()
    
    def save_model(self, filename):
        """Save the trained model"""
        if self.model is not None:
            path = os.path.join(self.output_dir, filename)
            self.model.save(path)
            print(f"Model saved to {path}")
    
    def load_model(self, filename='best_cnn_fraud_model.h5'):
        """Load a trained model"""
        path = os.path.join(self.output_dir, filename)
        try:
            self.model = tf.keras.models.load_model(path)
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, X, threshold=None):
        """
        Make predictions with adjustable threshold
        
        Args:
            X: Input features
            threshold: Decision threshold (defaults to optimal threshold)
            
        Returns:
            predictions, probabilities
        """
        if self.model is None:
            print("Model not loaded or trained")
            return None, None
            
        if threshold is None:
            threshold = self.best_threshold
            
        proba = self.model.predict(X).flatten()
        return (proba > threshold).astype(int), proba
