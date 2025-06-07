import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Page configuration
st.set_page_config(page_title="Timelytics - Delivery Time Prediction", layout="wide")

# Title and description
st.title("Timelytics: Order to Delivery Time Prediction")
st.markdown("Enter order details to predict the expected delivery time.")


# Simulated trained model (for demonstration)
# In a real scenario, this would be a pre-trained model loaded from a file
@st.cache_resource
def load_model():
    # Check if model exists, otherwise create a dummy model
    model_file = "delivery_time_model.pkl"
    if os.path.exists(model_file):
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
    else:
        # Dummy model for demonstration
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        # Simulated training data
        X = pd.DataFrame({
            'product_category': ['Electronics', 'Clothing', 'Books', 'Furniture'] * 25,
            'customer_location': ['Urban', 'Suburban', 'Rural', 'Urban'] * 25,
            'shipping_method': ['Standard', 'Express', 'Standard', 'Express'] * 25,
            'order_weight': np.random.uniform(0.5, 10, 100),
            'distance_km': np.random.uniform(10, 1000, 100)
        })
        y = np.random.uniform(1, 10, 100)  # Simulated delivery times in days
        # Encode categorical variables
        le_category = LabelEncoder()
        le_location = LabelEncoder()
        le_shipping = LabelEncoder()
        X['product_category'] = le_category.fit_transform(X['product_category'])
        X['customer_location'] = le_location.fit_transform(X['customer_location'])
        X['shipping_method'] = le_shipping.fit_transform(X['shipping_method'])
        model.fit(X, y)
        # Save dummy encoders for later use
        with open('le_category.pkl', 'wb') as f:
            pickle.dump(le_category, f)
        with open('le_location.pkl', 'wb') as f:
            pickle.dump(le_location, f)
        with open('le_shipping.pkl', 'wb') as f:
            pickle.dump(le_shipping, f)
    return model


# Load encoders
@st.cache_resource
def load_encoders():
    with open('le_category.pkl', 'rb') as f:
        le_category = pickle.load(f)
    with open('le_location.pkl', 'rb') as f:
        le_location = pickle.load(f)
    with open('le_shipping.pkl', 'rb') as f:
        le_shipping = pickle.load(f)
    return le_category, le_location, le_shipping


# Load model and encoders
model = load_model()
le_category, le_location, le_shipping = load_encoders()

# Create input form
with st.form("order_form"):
    st.header("Enter Order Details")

    col1, col2 = st.columns(2)

    with col1:
        product_category = st.selectbox(
            "Product Category",
            options=['Electronics', 'Clothing', 'Books', 'Furniture'],
            help="Select the product category"
        )
        customer_location = st.selectbox(
            "Customer Location",
            options=['Urban', 'Suburban', 'Rural'],
            help="Select the customer location type"
        )
        shipping_method = st.selectbox(
            "Shipping Method",
            options=['Standard', 'Express'],
            help="Select the shipping method"
        )

    with col2:
        order_weight = st.number_input(
            "Order Weight (kg)",
            min_value=0.1,
            max_value=50.0,
            value=1.0,
            step=0.1,
            help="Enter the weight of the order in kilograms"
        )
        distance = st.number_input(
            "Distance to Customer (km)",
            min_value=1.0,
            max_value=2000.0,
            value=100.0,
            step=1.0,
            help="Enter the approximate distance to customer in kilometers"
        )

    submitted = st.form_submit_button("Predict Delivery Time")

# Prediction logic
if submitted:
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'product_category': [product_category],
            'customer_location': [customer_location],
            'shipping_method': [shipping_method],
            'order_weight': [order_weight],
            'distance_km': [distance]
        })

        # Encode categorical variables
        input_data['product_category'] = le_category.transform([product_category])[0]
        input_data['customer_location'] = le_location.transform([customer_location])[0]
        input_data['shipping_method'] = le_shipping.transform([shipping_method])[0]

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display results
        st.success("Prediction Successful!")
        st.subheader("Predicted Delivery Time")
        st.metric(
            label="Estimated Delivery Time",
            value=f"{prediction:.2f} days",
            delta=None
        )

        # Display input summary
        with st.expander("Input Summary"):
            st.write("**Product Category**: " + product_category)
            st.write("**Customer Location**: " + customer_location)
            st.write("**Shipping Method**: " + shipping_method)
            st.write(f"**Order Weight**: {order_weight:.2f} kg")
            st.write(f"**Distance**: {distance:.2f} km")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add some styling and footer
st.markdown("""
<style>
    .stForm {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("Built with Streamlit | Timelytics © 2025")