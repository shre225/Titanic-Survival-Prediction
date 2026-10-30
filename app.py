import pickle
import pandas as pd
import streamlit as st

st.title(" Titanic Survival Predictor")
st.write("Enter the passenger details to predict their survival.")

# Load the trained model
with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
fare = st.number_input("Fare", min_value=0.0, value=30.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
ticket_group_size = st.number_input(
    "Ticket Group Size", min_value=1, value=1
)
has_cabin = st.selectbox("Cabin Information Available?", ["No", "Yes"])

# Feature engineering
if age <= 12:
    age_group = "Child"
elif age <= 18:
    age_group = "Teen"
elif age <= 35:
    age_group = "Young Adult"
elif age <= 60:
    age_group = "Adult"
else:
    age_group = "Senior"

if fare <= 7.91:
    fare_group = pd.Interval(-0.001, 7.91, closed="right")
elif fare <= 14.454:
    fare_group = pd.Interval(7.91, 14.454, closed="right")
elif fare <= 31.0:
    fare_group = pd.Interval(14.454, 31.0, closed="right")
else:
    fare_group = pd.Interval(31.0, 512.329, closed="right")

family_size = sibsp + parch + 1
is_alone = int(family_size == 1)

# Prepare input in the same feature format as the training data
passenger = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "AgeGroup": age_group,
    "TicketGroupSize": ticket_group_size,
    "FareGroup": fare_group,
    "Family_size": family_size,
    "IsAlone": is_alone,
    "Deck_analysis": "Unknown",
    "Title": title,
    "Embarked": embarked,
    "Has_Cabin": int(has_cabin == "Yes")
}])

# Make prediction
if st.button("Predict Survival"):
    try:
        prediction = model.predict(passenger)[0]

        if prediction == 1:
            st.success("The model predicts that the passenger survived! 🎉")
        else:
            st.error("The model predicts that the passenger did not survive.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(passenger)[0]
            st.write(
                f"Estimated survival probability: "
                f"{probabilities[1]:.2%}"
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that the input features and preprocessing match "
            "those used when training your saved model."
        )