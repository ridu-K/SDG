# 🏠 House Price Prediction System

A machine learning–based web application built using Streamlit that predicts house prices based on property features, location, and proximity to key facilities. The system uses a Gradient Boosting Regressor trained on real estate data and provides both price prediction and feature importance analysis.

## 🚀 Features

📊 Train ML model directly from the UI

🧠 Gradient Boosting Regression for accurate predictions

📈 R² score display after training

🔍 Feature importance visualization

🏙️ Categorical encoding for Area and City

💾 Model persistence using Joblib

🌐 Interactive Streamlit web interface

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Joblib

📂 Project Structure
```
SDG/
├── house_price_app.py      # Main Streamlit application
├── trial.py                # (Optional / experimental script)
├── newfinal.csv            # Dataset (required)
├── house_price_model.joblib
├── label_encoders.joblib      
├── requirements.txt    # Python dependencies
└── README.md
```

## 📊 Dataset

The model is trained using a CSV file named:
```
newfinal.csv

```

### Required Columns

- Security
- Brokerage
- Built-up area
- Bathrooms
- Age of property
- School Time
- School Distance
- Hospital Time
- Hospital Distance
- Railway Time
- Railway Distance
- Area (categorical)
- City (categorical)
- Price (target variable)

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create a Virtual Environment (Recommended)
```
python -m venv venv
source venv/bin/activate     # Linux / macOS
venv\Scripts\activate        # Windows
```

### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### ▶️ Run the Application
```
streamlit run house_price_app.py
```

The app will open in your browser at:
```
http://localhost:8501
```

## 🧠 Model Training

- Click “Train Model” in the app
- The system:
  - Encodes categorical variables
  - Splits data into train/test sets
  - Trains a GradientBoostingRegressor
  - Saves the model and label encoders
  - Displays R² score

## 🔮 Price Prediction

- Enter property details
- Select Area and City
- Click Predict Price
- View:
  - 💰 Predicted house price
  - 📊 Feature importance chart
  - 🔑 Top factors influencing price

## 📈 Feature Importance

The app displays:
- Bar chart of feature importance
- Top 3 most influential features affecting house price

This improves model interpretability and user trust.

## 📦 Saved Models

After training, the following files are generated:
- house_price_model.joblib
- label_encoders.joblib

These are automatically loaded on subsequent runs.

## 🔮 Future Improvements

- Add data validation & preprocessing pipeline
- Support unseen categories
- Deploy on Streamlit Cloud / AWS
- Add price trend visualizations
- Hyperparameter tuning UI

## 👨‍💻 Author

### Kuppili Raja Satya Alpana

If you find this project useful, feel free to ⭐ the repository!
