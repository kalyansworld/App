import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
st.title('Anomaly Detection')

try:
    data = pd.read_csv('creditcard_test.csv')
    st.write("Data Head:")
    st.write(data.head())
    data = data.apply(pd.to_numeric, errors='coerce')
    data.fillna(data.mean(), inplace=True)

    # Check for required columns
    if all(col in data.columns for col in ['Time', 'Amount']) and any(col.startswith('V') for col in data.columns):
        # Prepare the data for anomaly detection
        # Drop columns that are not needed for anomaly detection
        features = data.drop(columns=['Time'])
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        pca = PCA(n_components=2)  # Retain 95% of variance
        pca_features = pca.fit_transform(scaled_features)
        st.write("PCA features:",pca_features)
        # Initialize Isolation Forest model
        model = IsolationForest(contamination=0.01, random_state=42)  # Adjust contamination as needed
        data['Anomaly'] = model.fit_predict(pca_features)

        # Display the anomalies
        total_transactions = len(data)
        st.write("Total credit Transactions:",total_transactions)
        anomalies = data[data['Anomaly'] == -1]
        anomalies_no = len(anomalies)
        st.write(" No.of Anomaly Transactions:",anomalies_no)
        st.write("Anomaly Detection:")
        st.write(anomalies)

        # Visualize the anomalies
        fig, ax = plt.subplots(figsize=(10, 6))
        # Scatter plot for `Amount` to visualize anomalies
        sns.scatterplot(x='Time', y='Amount', hue='Anomaly', data=data, palette={-1: 'red', 1: 'blue'}, ax=ax)
        plt.title('Detected Anomalies in Transactions')
        st.pyplot(fig)

    else:
        st.error("The data is missing required columns or features.")

except Exception as e:
    st.error(f"An error occurred: {e}")
