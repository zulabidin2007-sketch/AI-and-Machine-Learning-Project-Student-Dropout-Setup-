# Student Dropout Prediction System 🎓

## 📌 Project Overview
This project is an end-to-end Machine Learning pipeline and web application designed to predict whether a student is at risk of dropping out. It serves as an **educational early-warning system**, allowing teachers and administrators to identify at-risk students early and provide the necessary support to retain them.

## 🎯 Problem Statement
Student dropout is a major challenge for educational institutions. The goal of this project is to use machine learning to analyze student performance and socio-economic factors to predict dropout risks accurately.

## 📊 Dataset & Data Preprocessing
- **Dataset:** A custom dataset was created focusing on three main features: `Attendance_Percentage`, `Previous_Grades`, and `Financial_Issues`.
- **Data Preprocessing & Exploratory Analysis:** The data was cleaned and checked for null values. The categorical feature (`Financial_Issues`: Yes/No) was encoded into binary numerical values (1/0) for model compatibility.

## 🤖 Selected Model & Training Process
- **Model Selected:** `RandomForestClassifier`
- **Training Process:** The dataset was split into an 80% training set and a 20% testing set using `train_test_split`. The Random Forest model was trained on the training set to identify complex patterns between attendance, grades, and dropout rates.

## 📈 Evaluation Results & Prediction
- The model successfully learned to classify students into 'High Risk' and 'Low Risk' categories. 
- **Final Prediction:** The trained model was exported as a `model.pkl` file and integrated into a live web interface. Based on user inputs, it instantly outputs the predicted dropout risk.

## 🚀 Deployment (Live Application)
The project is deployed as a live interactive web application using **Streamlit**. 
- Users can input student data using sliders and dropdowns.
- The app loads the `model.pkl` file in the background and displays real-time predictions.

## 💻 How to Run Locally
1. Clone this repository.
2. Install the required libraries using:
   `pip install -r requirements.txt`
3. Run the Streamlit application:
   `streamlit run app.py`

---
*Developed as part of the Big Brains Machine Learning Project Phase 8.*
