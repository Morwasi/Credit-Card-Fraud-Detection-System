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
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import os
import time
import joblib

# Set random seed for reproducibility
np.random.seed(42)

class CreditCardFraudXGBoost:
    """
    Optimized XGBoost model for credit card fraud detection with enhanced visualizations
    """

    def __init__(self, output_dir='outputs', validation_split=0.2):
        """
        Initialize the fraud detection model.
        
        Args:
            output_dir: Directory to save model outputs
            validation_split: Proportion of training data to use for validation
        """
        self.output_dir = output_dir
        self.validation_split = validation_split
        self.model = None
        self.best_params = None
        self.feature_importances = None
        self.eval_results = {}
        self.best_threshold = 0.5
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def build_model(self, params=None):
        """
        Build an XGBoost model with specified parameters
        """
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_estimators': 100,
                'scale_pos_weight': 1,
                'use_label_encoder': False
            }
        return xgb.XGBClassifier(**params, random_state=42)

    def hyperparameter_tune(self, X_train, y_train, cv=3, n_iter=20):
        """
        Perform hyperparameter tuning using RandomizedSearchCV
        """
        print("\nPerforming hyperparameter tuning...")
        
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'max_depth': [3, 4, 5, 6, 7, 8, 9],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'n_estimators': [50, 100, 200, 300, 500]
        }
        
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42
        )
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=cv,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        start_time = time.time()
        random_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        self.best_params = random_search.best_params_
        self.model = random_search.best_estimator_
        
        print(f"\nHyperparameter tuning completed in {tuning_time:.2f} seconds")
        print(f"Best ROC-AUC: {random_search.best_score_:.4f}")
        print("Best parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        pd.DataFrame([self.best_params]).to_csv(
            os.path.join(self.output_dir, 'best_parameters.csv'), index=False
        )
        
        return self.best_params, self.model

    def train_model(self, X_train, y_train, X_test=None, y_test=None, tune_hyperparams=True):
        """
        Train the XGBoost model with optional hyperparameter tuning
        """
        print("\nTraining XGBoost model...")
        start_time = time.time()
        
        if tune_hyperparams:
            _, self.model = self.hyperparameter_tune(X_train, y_train, cv=3, n_iter=20)
        else:
            self.model = self.build_model()
            eval_set = [(X_train, y_train)]
            if X_test is not None and y_test is not None:
                eval_set.append((X_test, y_test))
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric=['auc', 'error', 'logloss'],
                early_stopping_rounds=20,
                verbose=True
            )
            self.eval_results = self.model.evals_result()
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        self.feature_importances = self.model.feature_importances_
        self.save_model('best_xgboost_fraud_model.pkl')
        
        if hasattr(self.model, 'evals_result'):
            self.plot_training_history()
        
        return self.model

    def plot_training_history(self):
        """Plot enhanced training metrics visualization"""
        if not self.eval_results:
            print("No training history available.")
            return

        plt.figure(figsize=(12, 8))
        for metric in self.eval_results['validation_0'].keys():
            plt.plot(self.eval_results['validation_0'][metric], label=f'Train {metric}')
            if 'validation_1' in self.eval_results:
                plt.plot(self.eval_results['validation_1'][metric], label=f'Validation {metric}')
        
        plt.title('Training History', fontsize=16)
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Metric Value', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if hasattr(self.model, 'best_iteration'):
            plt.axvline(x=self.model.best_iteration, color='k', linestyle='--', 
                        label=f'Early Stopping (iter {self.model.best_iteration})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=300)
        plt.close()

    def evaluate_model(self, X_test, y_test, thresholds=None):
        """Comprehensive model evaluation with threshold analysis"""
        print("\nEvaluating XGBoost model...")
        
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        results = []
        
        for thresh in thresholds:
            y_pred = (y_pred_proba > thresh).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            results.append({
                'threshold': thresh,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'false_positives': cm[0, 1],
                'false_negatives': cm[1, 0]
            })

        results_df = pd.DataFrame(results)
        results_df['f2_score'] = (5 * results_df['precision'] * results_df['recall']) / \
                                (4 * results_df['precision'] + results_df['recall'])
        
        # Plot threshold analysis
        self._plot_threshold_analysis(results_df)
        
        # Select best threshold (maximizing F2-score)
        best_idx = results_df['f2_score'].idxmax()
        self.best_threshold = results_df.loc[best_idx, 'threshold']
        best_metrics = results_df.loc[best_idx].to_dict()
        
        # Final evaluation at best threshold
        y_pred = (y_pred_proba > self.best_threshold).astype(int)
        
        # Generate curves
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Enhanced visualizations
        self._plot_roc_curve(fpr, tpr, roc_auc)
        self._plot_pr_curve(precision, recall, pr_auc, np.mean(y_test))
        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_probability_distribution(y_test, y_pred_proba)
        self.plot_feature_importance()
        
        # Print and return metrics
        print("\nOptimal Threshold:", self.best_threshold)
        print(classification_report(y_test, y_pred))
        
        return {
            'best_threshold': self.best_threshold,
            'accuracy': best_metrics['accuracy'],
            'precision': best_metrics['precision'],
            'recall': best_metrics['recall'],
            'f1_score': best_metrics['f1_score'],
            'f2_score': best_metrics['f2_score'],
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'false_positives': best_metrics['false_positives'],
            'false_negatives': best_metrics['false_negatives'],
            'threshold_analysis': results_df
        }

    def _plot_threshold_analysis(self, results_df):
        """Plot threshold analysis visualization"""
        plt.figure(figsize=(14, 10))
        
        # Precision-Recall-F1 plot
        plt.subplot(2, 2, 1)
        plt.plot(results_df['threshold'], results_df['precision'], label='Precision')
        plt.plot(results_df['threshold'], results_df['recall'], label='Recall')
        plt.plot(results_df['threshold'], results_df['f1_score'], label='F1 Score')
        plt.plot(results_df['threshold'], results_df['f2_score'], label='F2 Score')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision/Recall Tradeoff')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Misclassification plot
        plt.subplot(2, 2, 2)
        plt.plot(results_df['threshold'], results_df['false_positives'], label='False Positives')
        plt.plot(results_df['threshold'], results_df['false_negatives'], label='False Negatives')
        plt.xlabel('Threshold')
        plt.ylabel('Count')
        plt.title('Misclassification Trends')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Accuracy plot
        plt.subplot(2, 2, 3)
        plt.plot(results_df['threshold'], results_df['accuracy'])
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Threshold')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Best threshold marker
        plt.subplot(2, 2, 4)
        best_row = results_df.iloc[results_df['f2_score'].idxmax()]
        plt.text(0.1, 0.9, f"Best Threshold: {best_row['threshold']:.2f}", fontsize=12)
        plt.text(0.1, 0.7, f"Precision: {best_row['precision']:.4f}", fontsize=12)
        plt.text(0.1, 0.5, f"Recall: {best_row['recall']:.4f}", fontsize=12)
        plt.text(0.1, 0.3, f"F1: {best_row['f1_score']:.4f}", fontsize=12)
        plt.text(0.1, 0.1, f"F2: {best_row['f2_score']:.4f}", fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'threshold_analysis.png'), dpi=300)
        plt.close()

    def _plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot enhanced ROC curve"""
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
        """Plot enhanced Precision-Recall curve"""
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
        """Plot enhanced confusion matrix"""
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
        plt.xlabel('Predicted Probability', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('Probability Distribution by Class', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'probability_distribution.png'), dpi=300)
        plt.close()

    def plot_feature_importance(self):
        """Plot enhanced feature importance"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            print("No feature importances available.")
            return
            
        importances = self.model.feature_importances_
        features = pd.DataFrame({
            'Feature': range(len(importances)),
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=features.head(15), palette='viridis')
        plt.title('Top 15 Feature Importances', fontsize=16)
        plt.xlabel('Importance Score', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300)
        plt.close()
        
        # Save feature importances to CSV
        features.to_csv(os.path.join(self.output_dir, 'feature_importances.csv'), index=False)

    def save_model(self, filename):
        """Save the trained model"""
        if self.model is not None:
            path = os.path.join(self.output_dir, filename)
            joblib.dump(self.model, path)
            print(f"Model saved to {path}")

    def load_model(self, filename='best_xgboost_fraud_model.pkl'):
        """Load a trained model"""
        path = os.path.join(self.output_dir, filename)
        try:
            self.model = joblib.load(path)
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances = self.model.feature_importances_
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
            
        proba = self.model.predict_proba(X)[:, 1]
        return (proba > threshold).astype(int), proba

    def plot_cross_validation_performance(self, X, y, cv=5):
        """
        Evaluate and visualize model performance across different cross-validation folds
        
        Args:
            X: Features dataset 
            y: Target labels
            cv: Number of cross-validation folds
        """
        from sklearn.model_selection import cross_validate, KFold
        
        print("\nEvaluating cross-validation performance...")
        
        try:
            # Define scoring metrics
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
                'roc_auc': 'roc_auc'
            }
            
            # Create cross-validation folds
            kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
            
            # Use the same parameters as the trained model if available
            if self.model is not None and hasattr(self.model, 'get_params'):
                params = self.model.get_params()
            else:
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'use_label_encoder': False,
                    'random_state': 42
                }
            
            # Create model for cross-validation
            model = xgb.XGBClassifier(**params)
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, X, y, 
                cv=kfold, 
                scoring=scoring, 
                return_train_score=True,
                return_estimator=True
            )
            
            # Create a DataFrame to store results
            cv_df = pd.DataFrame()
            
            # Add each metric to the DataFrame
            for metric in scoring.keys():
                cv_df[f'train_{metric}'] = cv_results[f'train_{metric}']
                cv_df[f'test_{metric}'] = cv_results[f'test_{metric}']
            
            # Save detailed CV results
            cv_df.to_csv(os.path.join(self.output_dir, 'cross_validation_results.csv'), index=False)
            
            # Calculate mean and std for each metric
            cv_summary = pd.DataFrame({
                'Metric': [],
                'Train Mean': [],
                'Train Std': [],
                'Test Mean': [],
                'Test Std': [],
                'Gap': []
            })
            
            for metric in scoring.keys():
                train_mean = cv_df[f'train_{metric}'].mean()
                train_std = cv_df[f'train_{metric}'].std()
                test_mean = cv_df[f'test_{metric}'].mean()
                test_std = cv_df[f'test_{metric}'].std()
                gap = train_mean - test_mean
                
                cv_summary = pd.concat([cv_summary, pd.DataFrame({
                    'Metric': [metric],
                    'Train Mean': [train_mean],
                    'Train Std': [train_std],
                    'Test Mean': [test_mean],
                    'Test Std': [test_std],
                    'Gap': [gap]
                })], ignore_index=True)
            
            # Save summary statistics
            cv_summary.to_csv(os.path.join(self.output_dir, 'cross_validation_summary.csv'), index=False)
            
            # Create boxplot comparison of train vs test for each metric
            plt.figure(figsize=(14, 10))
            
            for i, metric in enumerate(scoring.keys()):
                plt.subplot(2, 3, i+1)
                
                # Create DataFrame for boxplot
                boxplot_data = pd.DataFrame({
                    'Train': cv_df[f'train_{metric}'],
                    'Test': cv_df[f'test_{metric}']
                })
                
                # Create boxplot
                boxplot = sns.boxplot(data=boxplot_data)
                
                # Add scatter points for individual fold values
                for j in range(len(cv_df)):
                    plt.scatter(0, cv_df[f'train_{metric}'][j], color='red', s=30)
                    plt.scatter(1, cv_df[f'test_{metric}'][j], color='blue', s=30)
                
                # Add lines connecting train and test for each fold
                for j in range(len(cv_df)):
                    plt.plot([0, 1], [cv_df[f'train_{metric}'][j], cv_df[f'test_{metric}'][j]], 
                             'k--', alpha=0.3)
                
                # Add mean values
                plt.axhline(y=cv_df[f'train_{metric}'].mean(), color='red', linestyle='-', alpha=0.7)
                plt.axhline(y=cv_df[f'test_{metric}'].mean(), color='blue', linestyle='-', alpha=0.7)
                
                plt.title(f'{metric.upper()} Score', fontsize=12)
                plt.xticks([0, 1], ['Train', 'Test'])
                plt.grid(True, linestyle='--', alpha=0.3)
                
                # Add text with mean values
                plt.text(0, cv_df[f'train_{metric}'].min() * 0.97, 
                        f"Mean: {cv_df[f'train_{metric}'].mean():.4f}", ha='center')
                plt.text(1, cv_df[f'test_{metric}'].min() * 0.97, 
                        f"Mean: {cv_df[f'test_{metric}'].mean():.4f}", ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'cv_performance_boxplots.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create radar chart for cross-validation performance
            plt.figure(figsize=(10, 10))
            
            # Prepare data for radar chart
            metrics = list(scoring.keys())
            train_means = cv_summary['Train Mean'].values
            test_means = cv_summary['Test Mean'].values
            
            # Number of metrics
            N = len(metrics)
            
            # Create angle for each metric
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Add the first metric again at the end to close the circle
            train_means = np.append(train_means, train_means[0])
            test_means = np.append(test_means, test_means[0])
            metrics += [metrics[0]]
            
            # Create radar plot
            ax = plt.subplot(111, polar=True)
            
            # Plot train and test scores
            ax.plot(angles, train_means, 'o-', linewidth=2, label='Train')
            ax.fill(angles, train_means, alpha=0.25)
            ax.plot(angles, test_means, 'o-', linewidth=2, label='Test')
            ax.fill(angles, test_means, alpha=0.25)
            
            # Set labels and title
            ax.set_thetagrids(np.array(angles[:-1]) * 180/np.pi, metrics[:-1])
            ax.set_title('Cross-Validation Performance Radar Chart', fontsize=16)
            ax.grid(True)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'cv_radar_chart.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Extract feature importances from all folds and visualize
            feature_importances = []
            for i, estimator in enumerate(cv_results['estimator']):
                importances = pd.DataFrame({
                    'feature': range(len(estimator.feature_importances_)),
                    'importance': estimator.feature_importances_,
                    'fold': i
                })
                feature_importances.append(importances)
            
            # Combine all folds
            all_importances = pd.concat(feature_importances, ignore_index=True)
            
            # Calculate average importance per feature
            avg_importance = all_importances.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
            avg_importance.sort_values('mean', ascending=False, inplace=True)
            
            # Save average feature importances
            avg_importance.to_csv(os.path.join(self.output_dir, 'cv_feature_importance.csv'), index=False)
            
            # Plot top 15 features with variability across folds
            plt.figure(figsize=(12, 10))
            top_features = avg_importance.head(15)
            
            # Create error bar plot
            plt.errorbar(
                x=top_features['mean'],
                y=top_features['feature'],
                xerr=top_features['std'],
                fmt='o',
                capsize=5,
                ecolor='red',
                markeredgecolor='black',
                linewidth=2
            )
            
            plt.title('Feature Importance Across CV Folds (Top 15)', fontsize=16)
            plt.xlabel('Importance Score (Mean ± Std)', fontsize=14)
            plt.ylabel('Feature', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'cv_feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Cross-validation performance evaluation completed")
            
        except Exception as e:
            print(f"Error in cross-validation performance evaluation: {e}")
            import traceback
            traceback.print_exc()

    def plot_learning_curves(self, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
        """
        Plot learning curves to show how model performance changes with training data size
        
        Args:
            X: Features dataset
            y: Target variable
            cv: Number of cross-validation folds
            train_sizes: Array of training set sizes to evaluate
        """
        from sklearn.model_selection import learning_curve
        
        print("\nGenerating learning curves...")
        
        try:
            # Define model parameters (use the same as the trained model if available)
            if self.model is not None and hasattr(self.model, 'get_params'):
                params = self.model.get_params()
            else:
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'use_label_encoder': False,
                    'random_state': 42
                }
            
            # Create model for learning curve evaluation
            model = xgb.XGBClassifier(**params)
            
            # Calculate learning curve
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=cv, train_sizes=train_sizes,
                scoring='roc_auc', n_jobs=-1, random_state=42
            )
            
            # Calculate mean and std for training and test scores
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # Plot learning curve
            plt.figure(figsize=(12, 8))
            plt.grid(True, linestyle='--', alpha=0.6)
            
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                             alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                             alpha=0.1, color="g")
            plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
            
            # Add annotations for final scores
            plt.annotate(f"Final train: {train_mean[-1]:.4f}", 
                        xy=(train_sizes[-1], train_mean[-1]),
                        xytext=(train_sizes[-1] * 0.8, train_mean[-1] * 0.95),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            
            plt.annotate(f"Final CV: {test_mean[-1]:.4f}", 
                        xy=(train_sizes[-1], test_mean[-1]),
                        xytext=(train_sizes[-1] * 0.8, test_mean[-1] * 1.05),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            
            # Calculate and plot the gap between training and CV scores
            gap = train_mean - test_mean
            ax2 = plt.gca().twinx()
            ax2.plot(train_sizes, gap, 'o--', color="blue", alpha=0.5, label="Train-CV Gap")
            ax2.set_ylabel("Gap (Train-CV)", color="blue", fontsize=14)
            ax2.tick_params(axis='y', labelcolor="blue")
            
            plt.title("Learning Curves (XGBoost)", fontsize=16)
            plt.xlabel("Training examples", fontsize=14)
            plt.ylabel("ROC AUC Score", fontsize=14)
            plt.legend(loc="best", fontsize=12)
            
            # Save the learning curve plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save learning curve data
            curve_data = pd.DataFrame({
                'train_size': train_sizes,
                'train_score_mean': train_mean,
                'train_score_std': train_std,
                'cv_score_mean': test_mean,
                'cv_score_std': test_std,
                'gap': gap
            })
            curve_data.to_csv(os.path.join(self.output_dir, 'learning_curves.csv'), index=False)
            
            print("Learning curves generated successfully")
            
        except Exception as e:
            print(f"Error generating learning curves: {e}")
            import traceback
            traceback.print_exc()

    def create_model_dashboard(self, X_test, y_test, y_pred_proba, metrics, output_file='model_dashboard.html'):
        """
        Create an interactive HTML dashboard summarizing model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            y_pred_proba: Predicted probabilities
            metrics: Dictionary containing model evaluation metrics
            output_file: Filename for the dashboard HTML
        """
        try:
            print("\nGenerating interactive model dashboard...")
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
            
            # Calculate needed metrics and curves
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # ROC Curve
            from sklearn.metrics import roc_curve, auc, precision_recall_curve
            fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # PR Curve
            precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            # Create interactive dashboard using plotly
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'ROC Curve', 'Precision-Recall Curve',
                    'Probability Distribution', 'Confusion Matrix',
                    'Feature Importance (Top 10)', 'Threshold Impact on Metrics'
                ),
                specs=[
                    [{"type": "xy"}, {"type": "xy"}],
                    [{"type": "xy"}, {"type": "xy"}],
                    [{"type": "xy"}, {"type": "xy"}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.08
            )
            
            # 1. ROC Curve (row 1, col 1)
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC = {roc_auc:.4f})',
                    line=dict(color='darkorange', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(color='navy', dash='dash')
                ),
                row=1, col=1
            )
            
            # 2. PR Curve (row 1, col 2)
            fig.add_trace(
                go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name=f'PR Curve (AUC = {pr_auc:.4f})',
                    line=dict(color='green', width=2)
                ),
                row=1, col=2
            )
            
            # Add baseline to PR curve
            baseline = sum(y_test) / len(y_test)
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[baseline, baseline],
                    mode='lines',
                    name=f'Baseline ({baseline:.4f})',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=2
            )
            
            # 3. Probability Distribution (row 2, col 1)
            # Get indices for each class
            fraud_idx = np.where(y_test == 1)[0]
            non_fraud_idx = np.where(y_test == 0)[0]
            
            fig.add_trace(
                go.Histogram(
                    x=y_pred_proba[non_fraud_idx],
                    name='Non-Fraud (Class 0)',
                    opacity=0.6,
                    marker_color='green',
                    nbinsx=30
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Histogram(
                    x=y_pred_proba[fraud_idx],
                    name='Fraud (Class 1)',
                    opacity=0.6,
                    marker_color='red',
                    nbinsx=30
                ),
                row=2, col=1
            )
            
            # Add threshold line
            fig.add_vline(
                x=0.5, line_dash="dash", line_color="black",
                annotation_text="Threshold: 0.5",
                annotation_position="top right",
                row=2, col=1
            )
            
            # 4. Confusion Matrix (row 2, col 2)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate percentages
            cm_percent = cm / np.sum(cm) * 100
            
            # Create text annotations
            annotations = []
            for i in range(2):
                for j in range(2):
                    annotations.append(
                        f"{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)"
                    )
            
            # Create confusion matrix heatmap
            cm_labels = ['Non-Fraud', 'Fraud']
            cm_fig = px.imshow(
                cm,
                x=cm_labels,
                y=cm_labels,
                color_continuous_scale='Blues',
                labels=dict(x="Predicted", y="Actual", color="Count"),
                text_auto=annotations
            )
            
            # Extract traces from cm_fig and add to main figure
            for trace in cm_fig['data']:
                fig.add_trace(trace, row=2, col=2)
            
            # 5. Feature Importance (row 3, col 1)
            if hasattr(self, 'feature_importances') and self.feature_importances is not None:
                # Create dataframe of feature importances
                importances = self.feature_importances
                feature_importance = pd.DataFrame({
                    'feature': [f"Feature_{i}" for i in range(len(importances))],
                    'importance': importances
                })
                feature_importance.sort_values('importance', ascending=False, inplace=True)
                
                # Plot top 10 features
                top_features = feature_importance.head(10)
                
                fig.add_trace(
                    go.Bar(
                        y=top_features['feature'],
                        x=top_features['importance'],
                        orientation='h',
                        marker=dict(color='dodgerblue')
                    ),
                    row=3, col=1
                )
            
            # 6. Threshold Impact on Metrics (row 3, col 2)
            thresholds = np.arange(0.01, 1.0, 0.01)
            precision_values = []
            recall_values = []
            f1_values = []
            
            for threshold in thresholds:
                y_pred_threshold = (y_pred_proba >= threshold).astype(int)
                precision_values.append(precision_score(y_test, y_pred_threshold))
                recall_values.append(recall_score(y_test, y_pred_threshold))
                f1_values.append(f1_score(y_test, y_pred_threshold))
            
            fig.add_trace(
                go.Scatter(
                    x=thresholds, y=precision_values,
                    mode='lines',
                    name='Precision',
                    line=dict(color='blue', width=2)
                ),
                row=3, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=thresholds, y=recall_values,
                    mode='lines',
                    name='Recall',
                    line=dict(color='green', width=2)
                ),
                row=3, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=thresholds, y=f1_values,
                    mode='lines',
                    name='F1 Score',
                    line=dict(color='red', width=2)
                ),
                row=3, col=2
            )
            
            # Add threshold line
            fig.add_vline(
                x=0.5, line_dash="dash", line_color="black",
                annotation_text="Default: 0.5",
                annotation_position="top right",
                row=3, col=2
            )
            
            # Find and mark the best F1 threshold
            best_f1_idx = np.argmax(f1_values)
            best_f1_threshold = thresholds[best_f1_idx]
            
            fig.add_vline(
                x=best_f1_threshold, line_dash="dot", line_color="purple",
                annotation_text=f"Best F1: {best_f1_threshold:.2f}",
                annotation_position="top left",
                row=3, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text="Credit Card Fraud Detection Model Dashboard",
                showlegend=True,
                height=1200,
                width=1600,
                template='plotly_white',
                legend=dict(orientation="h", y=-0.05),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Update subplot axes titles
            fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
            fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
            
            fig.update_xaxes(title_text="Recall", row=1, col=2)
            fig.update_yaxes(title_text="Precision", row=1, col=2)
            
            fig.update_xaxes(title_text="Predicted Probability", row=2, col=1)
            fig.update_yaxes(title_text="Count", row=2, col=1)
            
            fig.update_xaxes(title_text="Predicted", row=2, col=2)
            fig.update_yaxes(title_text="Actual", row=2, col=2)
            
            fig.update_xaxes(title_text="Importance", row=3, col=1)
            fig.update_yaxes(title_text="Feature", row=3, col=1)
            
            fig.update_xaxes(title_text="Threshold", row=3, col=2)
            fig.update_yaxes(title_text="Score", row=3, col=2)
            
            # Add model performance metrics as annotations
            metric_text = (
                f"<b>Model Performance Metrics:</b><br>"
                f"Accuracy: {metrics['accuracy']:.4f}<br>"
                f"Precision: {metrics['precision']:.4f}<br>"
                f"Recall: {metrics['recall']:.4f}<br>"
                f"F1 Score: {metrics['f1_score']:.4f}<br>"
                f"ROC-AUC: {metrics['roc_auc']:.4f}<br>"
                f"PR-AUC: {metrics['pr_auc']:.4f}"
            )
            
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.5, y=1.05,
                text=metric_text,
                showarrow=False,
                font=dict(size=14),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=10
            )
            
            # Save dashboard to HTML file
            dashboard_path = os.path.join(self.output_dir, output_file)
            fig.write_html(dashboard_path)
            print(f"Interactive model dashboard saved to {dashboard_path}")
            
            return dashboard_path
            
        except ImportError as e:
            print(f"Warning: Could not create dashboard due to missing dependencies: {e}")
            print("Install plotly to enable interactive dashboards")
            return None
        except Exception as e:
            print(f"Error creating model dashboard: {e}")
            import traceback
            traceback.print_exc()
            return None
