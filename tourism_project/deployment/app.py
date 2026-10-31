import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Purchase Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the **Wellness Tourism Package** 
based on their demographic and interaction data. Enter the details below to get a prediction.
""")

# Input widgets categorized logically for user experience
st.subheader("Customer Demographic & Profile")
age = st.number_input("Age", 18, 100, 35)
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income (₹)", 5000.0, 150000.0, 20000.0, 500.0)

st.subheader("Travel Preferences & History")
number_of_person_visiting = st.slider("Number of Persons Visiting", 1, 6, 2)
preferred_property_star = st.selectbox("Preferred Property Star Rating", [3.0, 4.0, 5.0])
number_of_trips = st.number_input("Average Annual Trips", 1, 20, 3)
passport = st.selectbox("Holds a Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
number_of_children_visiting = st.slider("Number of Children Visiting (Age < 5)", 0, 3, 0)

st.subheader("Sales Interaction Details")
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
number_of_followups = st.slider("Number of Follow-ups", 1, 6, 3)
duration_of_pitch = st.number_input("Duration of Pitch (Minutes)", 5.0, 150.0, 15.0, 1.0)

# Construct input DataFrame matching training feature structure
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    result = "Likely to Purchase Wellness Package" if prediction == 1 else "Unlikely to Purchase"
    
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"**{result}** (Probability: {probability:.2%})")
    else:
        st.info(f"**{result}** (Probability: {probability:.2%})")
