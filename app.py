import streamlit as st
from datasets import load_dataset
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download

# Load the dataset
dataset = load_dataset("louiecerv/customer_churn")["train"]

# Define repository details
repo_id = "louiecerv/churn_prediction_model"
filename = "churn_prediction_model.pkl"

# Download and cache the model automatically
model_path = hf_hub_download(repo_id=repo_id, filename=filename)

# Load the model
with open(model_path, "rb") as f:
    model = pickle.load(f)

def main():
    # Streamlit app
    st.title("Customer Churn Prediction App")

    about = """## 🤗 Hugging Face for Machine Learning

This app showcases the power of **Hugging Face** for ML applications:
- 📂 **Datasets**: Easily access and share datasets.
- 🧠 **Models**: Download and use pre-trained models or upload your own.
- 🌍 **Spaces**: Deploy your app effortlessly.

👉 **Explore Hugging Face** and build your own ML-powered projects!

---
🚀 *Developed with Streamlit & Hugging Face 🤗*

**Created by: Louie F. Cervantes, M.Eng. (Information Engineering) 
(c) 2025 West Visayas State University**
"""
    with st.expander("About thiss app"):
        st.markdown(about)
    
    # --- Dataset Exploration ---
    st.header("Dataset Exploration")
    st.write("Let's explore the customer churn dataset:")

    # Display dataset information
    st.subheader("Dataset Sample")
    st.write(dataset.to_pandas().head())

    # Split data
    # Convert dataset to Pandas DataFrame
    df = dataset.to_pandas()
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # Define X_train and y_train
    X_train = train_data.drop('churn', axis=1)
    y_train = train_data['churn']

    # Show dataset size
    st.write(f"**Dataset Size:** {len(dataset)} rows")

    # Visualize churn distribution
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='churn', data=dataset.to_pandas(), ax=ax)
    ax.set_xticks([0, 1])  # Set tick locations
    ax.set_xticklabels(["No Churn", "Churn"])
    st.pyplot(fig)

    # --- Feature Importance ---
    st.header("Feature Importance")
    st.write("Understanding which features contribute most to churn prediction:")

    # Get feature importances (coefficients)
    feature_importance = pd.DataFrame({'Feature': X_train.columns,
                                    'Importance': model.coef_[0]})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # Plot feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    st.pyplot(fig)

    # --- Churn Prediction ---
    st.header("Predict Customer Churn")
    st.write("Enter customer data to predict churn:")

    # Input features
    tenure = st.number_input("Tenure (months)", min_value=1, step=1)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=0.01)
    total_charges = st.number_input("Total Charges", min_value=0.0, step=0.01)
    churn = st.selectbox("Churn", ['Yes', 'No'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

    # Preprocess input features
    input_features = {
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_One year': int(contract == 'One year'),
        'contract_Two year': int(contract == 'Two year'),
        'internet_service_Fiber optic': int(internet_service == 'Fiber optic'),
        'internet_service_No': int(internet_service == 'No')
    }
    input_df = pd.DataFrame([input_features])

    # Ensure feature names match those used during training
    train_columns = X_train.columns
    input_df = input_df[train_columns]  # Reorder columns to match train data

    # Make prediction
    if st.button("Predict Churn"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of churn
        if prediction == 1:
            st.write("This customer is likely to **churn**.")
        else:
            st.write("This customer is likely to **stay**.")
        st.write(f"Churn Probability: {probability:.2f}")

    # --- Hugging Face Explanation ---
    st.header("Hugging Face for Machine Learning")
    st.write(
        """
        This app showcases the power of Hugging Face for building ML applications.
        - **Datasets:** Easily access and share datasets.
        - **Models:** Download and use pre-trained models or upload your own.
        - **Spaces:** Deploy your app with a simple interface.
        
        Explore Hugging Face and build your own amazing ML projects!
        """
    )

if __name__ == "__main__":
    main()