import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
from folium.plugins import MarkerCluster

# Initialize geocoder
geolocator = Nominatim(user_agent="my_house_price_app")

@st.cache_data
def geocode_location(location_string):
    """Geocode a location string with error handling and caching"""
    try:
        location = geolocator.geocode(location_string)
        if location:
            return location.latitude, location.longitude
        return None
    except GeocoderTimedOut:
        time.sleep(1)
        return geocode_location(location_string)
    except:
        return None

def create_map_with_locations(data):
    """Create a map with property locations"""
    # Start with India's center coordinates
    center_lat, center_lng = 20.5937, 78.9629
    
    # Create base map
    m = folium.Map(location=[center_lat, center_lng], zoom_start=5,
                   tiles="cartodbpositron")
    
    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Process each unique location
    unique_locations = data[['Area', 'City']].drop_duplicates()
    
    for _, row in unique_locations.iterrows():
        location_string = f"{row['Area']}, {row['City']}, India"
        coords = geocode_location(location_string)
        
        if coords:
            # Get properties for this location
            location_properties = data[
                (data['Area'] == row['Area']) & 
                (data['City'] == row['City'])
            ]
            
            # Calculate average price for this location
            avg_price = location_properties['Price'].mean()
            
            # Create popup content
            popup_content = f"""
            <div style='width: 200px'>
                <b>Location:</b> {row['Area']}, {row['City']}<br>
                <b>Average Price:</b> ₹{avg_price:,.2f}<br>
                <b>Properties:</b> {len(location_properties)}<br>
                <b>Price Range:</b><br>
                Min: ₹{location_properties['Price'].min():,.2f}<br>
                Max: ₹{location_properties['Price'].max():,.2f}
            </div>
            """
            
            # Add marker with popup
            folium.Marker(
                coords,
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(marker_cluster)
            
    return m

def load_or_train_model():
    """Load or train the model and return necessary components"""
    try:
        if os.path.exists('house_price_model.joblib'):
            model = joblib.load('house_price_model.joblib')
            label_encoders = joblib.load('label_encoders.joblib')
            data = pd.read_csv('newfinal.csv')
            return model, label_encoders, data
    except:
        pass
    
    # Train new model if loading fails
    data = pd.read_csv('newfinal.csv')
    
    label_encoders = {}
    encoded_data = data.copy()
    for column in ['Area', 'City']:
        label_encoders[column] = LabelEncoder()
        encoded_data[column] = label_encoders[column].fit_transform(data[column])
    
    X = encoded_data.drop(columns=['Price'])
    y = encoded_data['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'house_price_model.joblib')
    joblib.dump(label_encoders, 'label_encoders.joblib')
    
    return model, label_encoders, data

def main():
    st.set_page_config(page_title="House Price Predictor", layout="wide")
    
    st.title("🏠 House Price Prediction System")
    st.write("Interactive map-based property price prediction system")
    
    # Load or train model
    with st.spinner("Initializing the prediction model..."):
        model, label_encoders, data = load_or_train_model()
    
    # Create tabs
    tab1, tab2 = st.tabs(["📍 Interactive Map", "🏠 Property Details"])
    
    with tab1:
        st.subheader("Property Locations Map")
        st.write("Click on markers to see property details for each location")
        
        # Create and display map
        with st.spinner("Loading map..."):
            m = create_map_with_locations(data)
            st_folium(m, width=1000, height=500)
        
        # Location selection below map
        col1, col2 = st.columns(2)
        with col1:
            area = st.selectbox("Select Area", sorted(data['Area'].unique()))
        with col2:
            city = st.selectbox("Select City", sorted(data['City'].unique()))
        
        # Show properties for selected location
        filtered_data = data[
            (data['Area'] == area) & 
            (data['City'] == city)
        ]
        if not filtered_data.empty:
            st.write(f"### Available Properties in {area}, {city}")
            
            # Display summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Price", f"₹{filtered_data['Price'].mean():,.2f}")
            with col2:
                st.metric("Number of Properties", len(filtered_data))
            with col3:
                st.metric("Average Area", f"{filtered_data['Built-up area'].mean():,.1f} sq ft")
            
            # Display detailed property list
            st.dataframe(
                filtered_data[['Built-up area', 'Bathrooms', 'Price']]
                .assign(Price=lambda x: x['Price'].apply(lambda y: f"₹{y:,.2f}"))
                .style.set_properties(**{'background-color': '#f0f2f6'})
            )
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏗️ Property Details")
            built_up_area = st.number_input("Built-up Area (sq ft)", 
                                          min_value=100.0, max_value=10000.0, 
                                          value=1000.0, step=100.0)
            bathrooms = st.number_input("Number of Bathrooms", 
                                      min_value=1.0, max_value=10.0, 
                                      value=2.0, step=0.5)
            age = st.number_input("Age of Property (years)", 
                                min_value=0.0, max_value=100.0, 
                                value=5.0, step=1.0)
        
        with col2:
            st.subheader("💰 Financial Details")
            security = st.number_input("Security Deposit", 
                                     min_value=0.0, max_value=1000000.0, 
                                     value=50000.0, step=5000.0)
            brokerage = st.number_input("Brokerage", 
                                      min_value=0, max_value=100000, 
                                      value=5000, step=1000)
        
        with st.expander("📍 Distance to Amenities"):
            col3, col4, col5 = st.columns(3)
            
            with col3:
                school_time = st.number_input("Time to School (mins)", min_value=0, value=15)
                school_distance = st.number_input("Distance to School (km)", min_value=0.0, value=2.0)
            
            with col4:
                hospital_time = st.number_input("Time to Hospital (mins)", min_value=0, value=20)
                hospital_distance = st.number_input("Distance to Hospital (km)", min_value=0.0, value=3.0)
            
            with col5:
                railway_time = st.number_input("Time to Railway (mins)", min_value=0, value=25)
                railway_distance = st.number_input("Distance to Railway (km)", min_value=0.0, value=4.0)
        
        if st.button("🔍 Predict House Price", use_container_width=True):
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
            
            # Display prediction
            st.markdown("---")
            st.markdown(f"""
                <div style='text-align: center; background-color: #f0f2f6; padding: 2rem; border-radius: 10px;'>
                    <h2 style='color: #1f77b4;'>Estimated House Price</h2>
                    <h1 style='color: #2ecc71; font-size: 3rem;'>₹{prediction:,.2f}</h1>
                </div>
            """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()