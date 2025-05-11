import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time
import os
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
#from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import NearMiss
#from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


class CreditCardFraudPreprocessor:
    """
    A class to preprocess credit card fraud data including:
    - Loading data from Kaggle or local file
    - Scaling features
    - Creating balanced datasets
    - Removing outliers
    - Splitting data for training and testing
    """
    
    def __init__(self, download_data=True, file_path=None, save_plots=False):
        """
        Initialize the preprocessor with configuration options.
        
        Parameters:
        -----------
        download_data : bool, default=True
            Whether to download the data from Kaggle using kagglehub
        file_path : str, default=None
            Path to the creditcard.csv file (used if download_data=False)
        save_plots : bool, default=False
            Whether to save visualization plots during preprocessing
        """
        self.download_data = download_data
        self.file_path = file_path
        self.save_plots = save_plots
        self.df = None
        self.balanced_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """
        Load the credit card fraud data from Kaggle or local file.
        
        Returns:
        --------
        pd.DataFrame
            The loaded dataframe
        """
        if self.download_data:
            print("Downloading credit card fraud dataset from Kaggle...")
            try:
                # Download the dataset
                path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
                print("Dataset downloaded to:", path)
                
                # List files in the directory
                files = os.listdir(path)
                print("Files in the directory:", files)
                
                # Find the CSV file
                csv_file = next((f for f in files if f.endswith('.csv')), None)
                if csv_file:
                    csv_path = os.path.join(path, csv_file)
                    print(f"Found CSV file: {csv_path}")
                else:
                    # If no CSV file is found, look for creditcard.csv specifically
                    csv_path = os.path.join(path, 'creditcard.csv')
                    if not os.path.exists(csv_path):
                        raise FileNotFoundError(f"Could not find CSV file in {path}")
                    
                # Load the data
                self.df = pd.read_csv(csv_path)
                print(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            except Exception as e:
                print(f"Error downloading or loading data: {e}")
                print("Falling back to file_path if provided...")
                if self.file_path:
                    self.df = pd.read_csv(self.file_path)
                else:
                    raise Exception("Could not download data and no file_path provided")
        else:
            if self.file_path is None:
                raise ValueError("file_path must be provided when download_data=False")
            self.df = pd.read_csv(self.file_path)
            
        return self.df
    
    def display_info(self):
        """
        Display basic information about the dataset.
        """
        if self.df is None:
            self.load_data()
            
        print('Dataset shape:', self.df.shape)
        print('No Frauds:', round(self.df['Class'].value_counts()[0]/len(self.df) * 100, 2), '% of the dataset')
        print('Frauds:', round(self.df['Class'].value_counts()[1]/len(self.df) * 100, 2), '% of the dataset')
        print('Missing values:', self.df.isnull().sum().max())
        
    def visualize_data(self):
        """
        Create and save plots of the dataset distribution.
        """
        if not self.save_plots:
            return
            
        if self.df is None:
            self.load_data()
        
        # Class distribution
        colors = ["#0101DF", "#DF0101"]
        plt.figure(figsize=(10,6))
        sns.countplot(x='Class', data=self.df, palette=colors)
        plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
        plt.savefig('class_distribution.png')
        plt.close()
        
        # Distribution of Amount and Time
        fig, ax = plt.subplots(1, 2, figsize=(18,4))
        amount_val = self.df['Amount'].values
        time_val = self.df['Time'].values
        sns.distplot(amount_val, ax=ax[0], color='r')
        ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
        ax[0].set_xlim([min(amount_val), max(amount_val)])
        sns.distplot(time_val, ax=ax[1], color='b')
        ax[1].set_title('Distribution of Transaction Time', fontsize=14)
        ax[1].set_xlim([min(time_val), max(time_val)])
        plt.savefig('amount_time_distributions.png')
        plt.close()
    
    def scale_features(self):
        """
        Scale the Amount and Time features using RobustScaler.
        
        Returns:
        --------
        pd.DataFrame
            The dataframe with scaled features
        """
        if self.df is None:
            self.load_data()
            
        rob_scaler = RobustScaler()
        self.df['scaled_amount'] = rob_scaler.fit_transform(self.df['Amount'].values.reshape(-1,1))
        self.df['scaled_time'] = rob_scaler.fit_transform(self.df['Time'].values.reshape(-1,1))
        self.df.drop(['Time','Amount'], axis=1, inplace=True)
        
        # Rearrange columns
        scaled_amount = self.df['scaled_amount']
        scaled_time = self.df['scaled_time']
        self.df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
        self.df.insert(0, 'scaled_amount', scaled_amount)
        self.df.insert(1, 'scaled_time', scaled_time)
        
        return self.df
    
    def create_balanced_dataset(self):
        """
        Create a balanced dataset by undersampling the majority class.
        
        Returns:
        --------
        pd.DataFrame
            Balanced dataframe with equal number of fraud and non-fraud cases
        """
        if self.df is None:
            self.scale_features()
            
        # Create balanced dataset
        self.df = self.df.sample(frac=1)  # Shuffle
        fraud_df = self.df.loc[self.df['Class'] == 1]
        non_fraud_df = self.df.loc[self.df['Class'] == 0][:len(fraud_df)]
        self.balanced_df = pd.concat([fraud_df, non_fraud_df])
        self.balanced_df = self.balanced_df.sample(frac=1, random_state=42)  # Shuffle again
        
        print(f"Created balanced dataset with {len(self.balanced_df)} samples")
        print('Class distribution:')
        print(self.balanced_df['Class'].value_counts())
        
        if self.save_plots:
            # Plot new class distribution
            colors = ["#0101DF", "#DF0101"]
            plt.figure(figsize=(10,6))
            sns.countplot(x='Class', data=self.balanced_df, palette=colors)
            plt.title('Equally Distributed Classes', fontsize=14)
            plt.savefig('equally_distributed_classes.png')
            plt.close()
            
            # Correlation matrices
            f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))
            corr = self.df.corr()
            sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
            ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)
            sub_sample_corr = self.balanced_df.corr()
            sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
            ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
            plt.savefig('Imbalanced_and_SubSample_Correlation_Matrix.png')
            plt.close()
            
            # Boxplots for correlations
            self._plot_correlations()
        
        return self.balanced_df
    
    def _plot_correlations(self):
        """Helper method to plot correlation boxplots"""
        if not self.save_plots or self.balanced_df is None:
            return
            
        colors = ["#0101DF", "#DF0101"]
        # Boxplots for negative correlations
        f, axes = plt.subplots(ncols=4, figsize=(20,4))
        sns.boxplot(x="Class", y="V17", data=self.balanced_df, palette=colors, ax=axes[0])
        axes[0].set_title('V17 vs Class Negative Correlation')
        sns.boxplot(x="Class", y="V14", data=self.balanced_df, palette=colors, ax=axes[1])
        axes[1].set_title('V14 vs Class Negative Correlation')
        sns.boxplot(x="Class", y="V12", data=self.balanced_df, palette=colors, ax=axes[2])
        axes[2].set_title('V12 vs Class Negative Correlation')
        sns.boxplot(x="Class", y="V10", data=self.balanced_df, palette=colors, ax=axes[3])
        axes[3].set_title('V10 vs Class Negative Correlation')
        plt.savefig('boxplots_negative_correlation.png')
        plt.close()
        
        # Boxplots for positive correlations
        f, axes = plt.subplots(ncols=4, figsize=(20,4))
        sns.boxplot(x="Class", y="V11", data=self.balanced_df, palette=colors, ax=axes[0])
        axes[0].set_title('V11 vs Class Positive Correlation')
        sns.boxplot(x="Class", y="V4", data=self.balanced_df, palette=colors, ax=axes[1])
        axes[1].set_title('V4 vs Class Positive Correlation')
        sns.boxplot(x="Class", y="V2", data=self.balanced_df, palette=colors, ax=axes[2])
        axes[2].set_title('V2 vs Class Positive Correlation')
        sns.boxplot(x="Class", y="V19", data=self.balanced_df, palette=colors, ax=axes[3])
        axes[3].set_title('V19 vs Class Positive Correlation')
        plt.savefig('boxplots_positive_correlation.png')
        plt.close()
        
        # Distribution plots
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        v14_fraud_dist = self.balanced_df['V14'].loc[self.balanced_df['Class'] == 1].values
        sns.distplot(v14_fraud_dist, ax=ax1, fit=norm, color='#FB8861')
        ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)
        v12_fraud_dist = self.balanced_df['V12'].loc[self.balanced_df['Class'] == 1].values
        sns.distplot(v12_fraud_dist, ax=ax2, fit=norm, color='#56F9BB')
        ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)
        v10_fraud_dist = self.balanced_df['V10'].loc[self.balanced_df['Class'] == 1].values
        sns.distplot(v10_fraud_dist, ax=ax3, fit=norm, color='#C5B3F9')
        ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)
        plt.savefig('Feature_Distribution_Analysis.png')
        plt.close()
    
    def remove_outliers(self):
        """
        Remove outliers from fraud cases in V14, V12, and V10 features.
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with outliers removed
        """
        if self.balanced_df is None:
            self.create_balanced_dataset()
        
        # Remove outliers for V14
        v14_fraud = self.balanced_df['V14'].loc[self.balanced_df['Class'] == 1].values
        q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
        v14_iqr = q75 - q25
        v14_cut_off = v14_iqr * 1.5
        v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
        self.balanced_df = self.balanced_df.drop(self.balanced_df[(self.balanced_df['V14'] > v14_upper) | 
                                            (self.balanced_df['V14'] < v14_lower)].index)
        
        # Remove outliers for V12
        v12_fraud = self.balanced_df['V12'].loc[self.balanced_df['Class'] == 1].values
        q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
        v12_iqr = q75 - q25
        v12_cut_off = v12_iqr * 1.5
        v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
        self.balanced_df = self.balanced_df.drop(self.balanced_df[(self.balanced_df['V12'] > v12_upper) | 
                                            (self.balanced_df['V12'] < v12_lower)].index)
        
        # Remove outliers for V10
        v10_fraud = self.balanced_df['V10'].loc[self.balanced_df['Class'] == 1].values
        q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
        v10_iqr = q75 - q25
        v10_cut_off = v10_iqr * 1.5
        v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
        self.balanced_df = self.balanced_df.drop(self.balanced_df[(self.balanced_df['V10'] > v10_upper) | 
                                            (self.balanced_df['V10'] < v10_lower)].index)
        
        print(f"Dataset after outlier removal: {len(self.balanced_df)} samples")
        
        if self.save_plots:
            # Boxplots after removing outliers
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
            colors = ['#B3F9C5', '#f9c5b3']
            # Feature V14
            sns.boxplot(x="Class", y="V14", data=self.balanced_df, ax=ax1, palette=colors)
            ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)
            # Feature 12
            sns.boxplot(x="Class", y="V12", data=self.balanced_df, ax=ax2, palette=colors)
            ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)
            # Feature V10
            sns.boxplot(x="Class", y="V10", data=self.balanced_df, ax=ax3, palette=colors)
            ax3.set_title("V10 Feature \n Reduction of outliers", fontsize=14)
            plt.savefig('Boxplots_after_removing_outliers.png')
            plt.close()
        
        return self.balanced_df
    
    def split_data(self):
        """
        Split the preprocessed data into training and testing sets.
        
        Returns:
        --------
        X_train, X_test, y_train, y_test : numpy.ndarray
            Train/test split of features and labels
        """
        if self.balanced_df is None:
            self.remove_outliers()
        
        # Prepare data for modeling
        X = self.balanced_df.drop('Class', axis=1)
        y = self.balanced_df['Class']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Convert to numpy arrays
        self.X_train = self.X_train.values
        self.X_test = self.X_test.values
        self.y_train = self.y_train.values
        self.y_test = self.y_test.values
        
        print("Data split completed.")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"y_test shape: {self.y_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def preprocess(self):
        """
        Run the full preprocessing pipeline.
        
        Returns:
        --------
        X_train, X_test, y_train, y_test : numpy.ndarray
            Preprocessed train/test split ready for model training
        """
        self.load_data()
        self.display_info()
        self.visualize_data()
        self.scale_features()
        self.create_balanced_dataset()
        self.remove_outliers()
        return self.split_data()


# Example usage:
#if __name__ == "__main__":
    # Create preprocessor instance
#    preprocessor = CreditCardFraudPreprocessor(download_data=True, save_plots=True)
    
    # Run the full preprocessing pipeline
#    X_train, X_test, y_train, y_test = preprocessor.preprocess()
    
#    print("Preprocessing complete and data is ready for modeling.")
