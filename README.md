# Credit-Card-Fraud-Detection-System
This project implements a deep learning approach for credit card fraud detection using a CNN model. The system includes data preprocessing, model training, and evaluation.

Project Structure :


credit-card-fraud-detection-System/
│
├── execute.py                 # Main execution script
│
├── preprocess_data/           # Data preprocessing modules
│   └── preprocess.py          # Contains CreditCardFraudPreprocessor class
│
├── model/                     # Model implementation
│   └── cnn_model.py           # Contains CreditCardFraudCNN class
│
├── data/                      # Directory for raw data (created by the script)
│
└── outputs/                   # Output directory (created by the script)
    ├── preprocessed_data/     # Stored preprocessed datasets
    │   ├── X_train.npy
    │   ├── X_test.npy
    │   ├── y_train.npy
    │   ├── y_test.npy
    │   └── data_info.txt
    │
    ├── cnn_model/             # CNN model outputs
    │   ├── best_cnn_fraud_model.h5
    │   ├── training_history.png
    │   └── confusion_matrix.png
    │
    └── plots/                 # Data visualization plots (if save_plots=True)



Requirements

Python 3.6+
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
TensorFlow/Keras
Kagglehub (for downloading the dataset)

Install the required packages using:

pip install numpy pandas matplotlib seaborn scikit-learn tensorflow kagglehub


Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle. The dataset will be automatically downloaded using the kagglehub library when running the preprocessing step. If you're running this for the first time, you may need to set up your Kaggle credentials:

Create a Kaggle account if you don't have one
Generate an API key from your Kaggle account settings
Save the API key as instructed by the kagglehub library

How to Run

Simply execute the main script:
python execute.py

The script will:

Create necessary directories if they don't exist
Download and preprocess the credit card fraud dataset (if not already preprocessed)
Train the CNN model (or load an existing trained model)
Evaluate the model and display performance metrics

Configuration Options
You can modify the following parameters in execute.py:
Preprocessing:
pythonX_train, X_test, y_train, y_test = preprocess_data(force_reprocess=False)

Set force_reprocess=True to re-download and reprocess the data, even if preprocessed data already exists

CNN Model Training:
pythonnn_model, nn_metrics = train_and_evaluate_cnn_model(
    X_train, X_test, y_train, y_test, force_retrain=False
)

Set force_retrain=True to retrain the model, even if a trained model already exists

Visualization:
In the preprocess_data function:
pythonpreprocessor = CreditCardFraudPreprocessor(
    download_data=True,
    save_plots=True
)

Set save_plots=False to skip generating and saving visualization plots during preprocessing

Classes and Functions
CreditCardFraudPreprocessor
This class handles all data preprocessing steps:

Downloading the dataset from Kaggle
Feature scaling
Creating a balanced dataset (equal number of fraud and non-fraud cases)
Removing outliers
Splitting data into training and testing sets

CreditCardFraudCNN
This class implements the CNN model:

Model architecture
Training with early stopping
Evaluation metrics calculation
Model saving and loading

Output Files

Preprocessed Data: Stored in outputs/preprocessed_data/ as NumPy arrays
Model Files: The best CNN model is saved at outputs/cnn_model/best_cnn_fraud_model.h5
Visualization: Training history, confusion matrices, and other plots are saved in their respective directories

Performance Metrics
The model evaluation provides the following metrics:

Accuracy
Precision
Recall
F1 Score
ROC AUC
PR AUC (Precision-Recall Area Under the Curve)

Notes

The first run will take longer as it downloads the dataset and performs preprocessing
Subsequent runs will use the saved preprocessed data and trained model (unless forced to reprocess/retrain)
Make sure you have sufficient disk space for the dataset and model files
