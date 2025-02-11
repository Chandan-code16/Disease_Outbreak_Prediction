# 🏥 Disease Outbreak Prediction Web App

![Project Banner](https://via.placeholder.com/1000x300?text=Disease+Outbreak+Prediction+App)

## 📌 Project Overview
This is a Machine Learning-powered Web App built with Python, Scikit-Learn, and Streamlit that predicts the likelihood of Diabetes, Heart Disease, and Parkinson’s Disease based on user inputs. The trained models analyze medical parameters to provide an instant health risk assessment.

🔗 Live Demo: [Click Here](https://your-app-name.streamlit.app)

## ✨ Features
✅ Predict Disease Risks (Diabetes, Heart Disease, Parkinson’s)  
✅ User-Friendly Web Interface (Built with Streamlit)  
✅ Machine Learning Models (RandomForestClassifier)  
✅ Real-Time Predictions Based on User Inputs  
✅ Deployed on Streamlit Cloud  

## 📂 Project Structure
Disease_Outbreak_Prediction/
│-- datasets/             # CSV files for training the models
│   │-- diabetes.csv
│   │-- heart.csv
│   │-- parkinsons.csv
│
│-- models/               # Trained Machine Learning Models
│   │-- diabetes_model.sav
│   │-- heart_model.sav
│   │-- parkinsons_model.sav
│
│-- train_models.py       # Python script to train ML models
│-- app.py                # Streamlit web app
│-- requirements.txt      # Dependencies for deployment
│-- README.md             # Project Documentation

## 🚀 Installation & Setup
Follow these steps to run the project locally:

### 1️⃣ Clone the Repository
git clone https://github.com/Chandan-code16/Disease_Outbreak_Prediction.git
cd Disease_Outbreak_Prediction

### 2️⃣ Create & Activate Virtual Environment
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on Mac/Linux
source venv/bin/activate

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Train Machine Learning Models
python train_models.py

### 5️⃣ Run the Streamlit Web App
streamlit run app.py
✅ Open your browser at http://localhost:8501/ to use the app!

## 🎯 How to Use
1️⃣ Select a disease from the sidebar (Diabetes, Heart Disease, Parkinson's).  
2️⃣ Enter the required medical parameters.  
3️⃣ Click Predict to get an instant result.  

## 📊 Technologies Used
- Python 🐍
- Scikit-Learn 🤖
- Streamlit 🎨
- Pandas & NumPy 📊
- Machine Learning (RandomForest) 🏥
- Git & GitHub 🖥

## 📢 Deployment
This project is deployed using Streamlit Cloud. To deploy your own version:
1️⃣ Push your code to GitHub.  
2️⃣ Go to [Streamlit Cloud](https://share.streamlit.io/).  
3️⃣ Connect your GitHub repository & deploy!  

## 💡 Future Improvements
🔹 Improve model accuracy with Hyperparameter Tuning.  
🔹 Add more diseases and models.  
🔹 Enhance UI with charts & visualizations.  

## 📬 Contact
If you have any questions or suggestions, feel free to reach out! 🚀  
📧 Email: er.chandanmishra03@gmail.com  
🔗 GitHub: [Chandan-code16](https://github.com/Chandan-code16)  
🔗 LinkedIn: [chandan-mishra-b2110a247](https://www.linkedin.com/in/chandan-mishra-b2110a247)  

---
🛠 Happy Coding! Keep Innovating! 🚀
