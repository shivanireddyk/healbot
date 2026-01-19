🩺 HealBot – AI-Based Healthcare Chatbot

HealBot is a command-line healthcare chatbot that predicts possible diseases based on user-reported symptoms. It uses machine learning (Decision Tree Classifier) trained on medical symptom data to provide predictions, descriptions, severity analysis, and precautionary advice.

🚀 Key Features

Symptom-based disease prediction using Decision Tree Classifier.

Secondary prediction for improved accuracy

Severity-based health risk evaluation

Disease descriptions and precaution recommendations

Interactive, user-friendly command-line interface

Personalized input (name, age, gender)

🧠 Machine Learning Approach

Model: Decision Tree Classifier (Scikit-learn)

Encoding: Label Encoding for disease labels

Validation: Train–test split and cross-validation

Input: Binary symptom vector

Output: Predicted disease(s) with explanation

📂 Dataset Structure
Data/
│── Training.csv
│── Testing.csv

MasterData/
│── symptom_Description.csv
│── symptom_severity.csv
│── symptom_precaution.csv

🛠️ Technologies Used

Python

Pandas & NumPy

Scikit-learn

Regular Expressions

CSV Data Handling

▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/shivanireddyk/healbot.git
cd healbot

2️⃣ Install Dependencies
pip install pandas numpy scikit-learn pyttsx3

3️⃣ Run the Chatbot
python app.py

💬 How It Works

User enters personal details (name, age, gender)

User inputs symptoms (with pattern matching support)

Model predicts disease using a decision tree

Severity is calculated based on symptom duration

Disease description and precautions are displayed

⚠️ Disclaimer

HealBot is intended only for educational and informational purposes.
It is not a substitute for professional medical advice, diagnosis, or treatment.

📌 Future Enhancements

Web-based UI (Flask / Streamlit)

Voice interaction

Improved model accuracy using ensemble methods

Deployment on cloud platforms
