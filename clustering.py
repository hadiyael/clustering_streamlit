import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Charger le modèle KMeans sauvegardé
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

# Interface Streamlit
st.title('Clustering avec KMeans')
st.write('Téléchargez un fichier CSV pour effectuer le clustering.')

# Téléchargement de fichier
uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Effectuer les prédictions de clusters
    predictions = kmeans_model.predict(data)
    data['Cluster'] = predictions
    
    # Afficher les données avec les clusters
    st.write(data.head())
    
    # Visualisation
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['Cluster'], cmap='viridis')
    plt.title('Clustering des données')
    st.pyplot(plt)
