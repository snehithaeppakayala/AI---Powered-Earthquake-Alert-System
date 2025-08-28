Project Title : AI-Powered Earthquake Alert System

##  Project Overview
This project predicts **earthquake alert levels** (Green, Yellow, Orange, Red) using **Machine Learning** techniques.  
We use features such as **latitude, longitude, depth, and magnitude** to classify earthquake severity levels.  

The project includes:
- Data preprocessing  
- Feature scaling  
- Handling imbalanced dataset using **SMOTE**  
- Training a **Random Forest Classifier**  
- Model evaluation with accuracy, classification report, and confusion matrix  
- Saving trained model using **Joblib**  

---

##  Features
- Predicts earthquake alert level (Green, Yellow, Orange, Red).  
- Balances dataset using SMOTE.  
- Visualizes confusion matrix.  
- Saves trained model, scaler, and label encoder for later deployment.  

---

##  Project Structure
AI-Powered Earthquake Alert System/
│-- earthquakes.csv # Dataset (from Kaggle)
│-- Alert.py # Main training script
│-- earthquake_model.pkl # Saved Random Forest model
│-- scaler.pkl # Saved StandardScaler
│-- label_encoder.pkl # Saved LabelEncoder
│-- README.md # Project documentation

---

## 🛠️ Installation
Install required dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn joblib

Usage

Place your dataset earthquakes.csv in the project folder.
Run the training script:

python Alert.py

The model, scaler, and label encoder will be saved as .pkl files.
You will also get evaluation metrics (accuracy, classification report, confusion matrix).

Example Output
Accuracy: 0.98

Classification Report:
              precision    recall  f1-score   support
       green       0.99      0.96      0.98
      orange       0.98      0.99      0.99
         red       0.98      1.00      0.99
      yellow      0.96      0.97      0.97

Confusion Martix Visualization

![](<Screenshot 2025-08-28 184839.png>)

Next Steps

Deploy model as a Flask Web App.
Add real-time user input for predictions.
Visualize ROC curves and alert distribution.