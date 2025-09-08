import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib
import os

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
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Save model and encoders
    joblib.dump(model, 'house_price_model.joblib')
    joblib.dump(label_encoders, 'label_encoders.joblib')
    
    return model, label_encoders

def main():
    st.set_page_config(page_title="House Price Predictor", layout="wide")
    
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            margin-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🏠 House Price Prediction System")
    st.write("Enter the house details below to get an estimated price")
    
    # Load or train model
    if not os.path.exists('house_price_model.joblib'):
        with st.spinner("Setting up the prediction model... Please wait."):
            model, label_encoders = train_model()
            st.success("Setup complete!")
    else:
        model = joblib.load('house_price_model.joblib')
        label_encoders = joblib.load('label_encoders.joblib')
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📍 Location Details")
        area = st.selectbox("Select Area", label_encoders['Area'].classes_)
        city = st.selectbox("Select City", label_encoders['City'].classes_)
        
    with col2:
        st.subheader("🏗️ Property Details")
        built_up_area = st.number_input("Built-up Area (sq ft)", 
                                      min_value=100.0, 
                                      max_value=10000.0, 
                                      value=1000.0,
                                      step=100.0)
        bathrooms = st.number_input("Number of Bathrooms", 
                                  min_value=1.0, 
                                  max_value=10.0, 
                                  value=2.0,
                                  step=0.5)
        age = st.number_input("Age of Property (years)", 
                            min_value=0.0, 
                            max_value=100.0, 
                            value=5.0,
                            step=1.0)
        
    with col3:
        st.subheader("💰 Financial Details")
        security = st.number_input("Security Deposit", 
                                 min_value=0.0, 
                                 max_value=1000000.0, 
                                 value=50000.0,
                                 step=5000.0)
        brokerage = st.number_input("Brokerage", 
                                  min_value=0, 
                                  max_value=100000, 
                                  value=5000,
                                  step=1000)

    # Create expandable section for additional details
    with st.expander("📍 Distance to Amenities (Optional)"):
        col4, col5, col6 = st.columns(3)
        
        with col4:
            school_time = st.number_input("Time to School (minutes)", 
                                        min_value=0, 
                                        max_value=120, 
                                        value=15)
            school_distance = st.number_input("Distance to School (km)", 
                                           min_value=0.0, 
                                           max_value=30.0, 
                                           value=2.0)
            
        with col5:
            hospital_time = st.number_input("Time to Hospital (minutes)", 
                                          min_value=0, 
                                          max_value=120, 
                                          value=20)
            hospital_distance = st.number_input("Distance to Hospital (km)", 
                                             min_value=0.0, 
                                             max_value=30.0, 
                                             value=3.0)
            
        with col6:
            railway_time = st.number_input("Time to Railway Station (minutes)", 
                                         min_value=0, 
                                         max_value=120, 
                                         value=25)
            railway_distance = st.number_input("Distance to Railway Station (km)", 
                                            min_value=0.0, 
                                            max_value=30.0, 
                                            value=4.0)

    # Predict button
    if st.button("🔍 Predict House Price"):
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
        
        # Encode categorical variables
        for column in ['Area', 'City']:
            input_data[column] = label_encoders[column].transform(input_data[column])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display prediction in a nice format
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; background-color: #f0f2f6; padding: 2rem; border-radius: 10px;'>
                <h2 style='color: #1f77b4;'>Estimated House Price</h2>
                <h1 style='color: #2ecc71; font-size: 3rem;'>₹{:,.2f}</h1>
            </div>
        """.format(prediction), unsafe_allow_html=True)
        
        # Display key factors
        st.markdown("### 📊 Key Factors Affecting the Price")
        feature_importance = pd.DataFrame({
            'Feature': input_data.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        top_features = feature_importance.head(3)
        for _, feature in top_features.iterrows():
            st.write(f"- {feature['Feature']}: {feature['Importance']:.2%} impact on price")

if __name__ == '__main__':
    main()