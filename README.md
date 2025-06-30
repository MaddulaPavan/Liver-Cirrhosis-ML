# Revolutionizing Liver Care: Predicting Liver Cirrhosis using Advanced Machine Learning Techniques

This project leverages advanced machine learning algorithms to predict liver cirrhosis, enabling early detection and improved patient outcomes. The application provides a web interface for healthcare professionals to input patient data and receive real-time predictions.

## Features
- Downloads and uses the latest liver cirrhosis dataset from Kaggle
- Cleans and preprocesses the data automatically
- Trains and optimizes seven different classification models
- Selects and saves the best-performing model
- Provides a web-based UI for user input and prediction

## Algorithms Used
- Logistic Regression
- Logistic Regression CV
- XGBoost Classifier
- Ridge Classifier
- K-Nearest Neighbors Classifier
- Random Forest Classifier
- Decision Tree Classifier

## Setup Instructions

1. **Clone the repository and navigate to the project directory:**
   ```
   git clone <repo-url>
   cd Cancer_SI
   ```

2. **Install the required dependencies:**
   ```
   pip install pandas scikit-learn xgboost flask kagglehub joblib
   ```

3. **Run the application:**
   ```
   python main.py
   ```

## Project Flow
1. User enters patient data via the web UI.
2. The backend loads the best-trained model and predicts the risk of cirrhosis.
3. The prediction is displayed instantly on the web page.

## Notes
- The dataset is automatically downloaded from Kaggle using KaggleHub.
- The model is retrained each time the script is run, ensuring the latest data and best parameters are used.
- For best results, ensure a stable internet connection for dataset download.
 
