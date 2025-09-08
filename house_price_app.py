import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Function to train the model
def train_model():
    # Load and prepare data
    data = pd.read_csv('newfinal.csv')
    
    # Encode categorical variables
    label_encoders = {}
    for column in ['Area', 'City']:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    
    # Split features and target
    X = data.drop(columns=['Price'])
    y = data['Price']
    
    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with best parameters from grid search
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Save model and encoders
    joblib.dump(model, 'house_price_model.joblib')
    joblib.dump(label_encoders, 'label_encoders.joblib')
    
    # Calculate and return accuracy metrics
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return model, label_encoders, r2

# Function to make predictions
def predict_price(model, label_encoders, input_data):
    # Encode categorical variables
    input_df = input_data.copy()
    for column in ['Area', 'City']:
        input_df[column] = label_encoders[column].transform(input_df[column])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    return prediction

# Streamlit interface
def main():
    st.title("House Price Prediction System")
    
    # Training section
    st.header("Model Training")
    if st.button("Train Model") or not os.path.exists('house_price_model.joblib'):
        with st.spinner("Training model... Please wait."):
            model, label_encoders, r2_score = train_model()
            st.success(f"Model trained successfully! R² Score: {r2_score:.4f}")
    else:
        model = joblib.load('house_price_model.joblib')
        label_encoders = joblib.load('label_encoders.joblib')
    
    # Prediction section
    st.header("Price Prediction")
    st.write("Enter the details below to predict house price")
    
    # Create input fields
    col1, col2 = st.columns(2)
    
    with col1:
        security = st.number_input("Security Deposit", min_value=0.0, format="%.2f")
        brokerage = st.number_input("Brokerage", min_value=0)
        built_up_area = st.number_input("Built-up Area (sq ft)", min_value=0.0, format="%.2f")
        bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, format="%.1f")
        age = st.number_input("Age of Property (years)", min_value=0.0, format="%.1f")
        school_time = st.number_input("School Time (minutes)", min_value=0)
        school_distance = st.number_input("School Distance (km)", min_value=0.0, format="%.2f")
    
    with col2:
        hospital_time = st.number_input("Hospital Time (minutes)", min_value=0)
        hospital_distance = st.number_input("Hospital Distance (km)", min_value=0.0, format="%.2f")
        railway_time = st.number_input("Railway Time (minutes)", min_value=0)
        railway_distance = st.number_input("Railway Distance (km)", min_value=0.0, format="%.2f")
        
        # Get unique values for Area and City from label encoders
        area_options = label_encoders['Area'].classes_
        city_options = label_encoders['City'].classes_
        
        area = st.selectbox("Area", area_options)
        city = st.selectbox("City", city_options)

    # Prediction button
    if st.button("Predict Price"):
        try:
            # Create input data frame
            input_data = pd.DataFrame({
                'Security': [security],
                'Brokerage': [brokerage],
                'Built-up area': [built_up_area],
                'Bathrooms': [bathrooms],
                'Age of property': [age],
                'School Time': [school_time],
                'School Distance': [school_distance],
                'Hospital Time': [hospital_time],
                'Hospital Distance': [hospital_distance],
                'Railway Time': [railway_time],
                'Railway Distance': [railway_distance],
                'Area': [area],
                'City': [city]
            })
            
            # Make prediction
            prediction = predict_price(model, label_encoders, input_data)
            
            # Display prediction
            st.success(f"Predicted House Price: ₹{prediction:,.2f}")
            
            # Display feature importance
            st.subheader("Feature Importance Analysis")
            feature_importance = pd.DataFrame({
                'Feature': input_data.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(feature_importance.set_index('Feature'))
            
            # Additional insights
            st.subheader("Key Insights")
            st.write("Top 3 factors affecting the price:")
            for i in range(3):
                feature = feature_importance.iloc[i]
                st.write(f"{i+1}. {feature['Feature']}: {feature['Importance']:.2%} importance")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please make sure all inputs are valid and try again.")

if __name__ == '__main__':
    main()