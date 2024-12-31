import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import umap
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import skew

# Load the cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")  

# Feature selection (adjust based on your dataset)
features = ['Maternal mortality ratio', 'Life expectancy', 'GDP', 'Urban_population']
X = df[features]

# Make a copy of the original dataframe to preserve it
df_original = df.copy()

# Calculate skewness of the features
skewness = X.apply(skew)

# Apply a log transformation for features with high skewness
X_log_transformed = X.copy()
for feature in X.columns:
    if skewness[feature] > 1 or skewness[feature] < -1:  
        X_log_transformed[feature] = np.log1p(X[feature])

# Scaling the features 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log_transformed)  

# Apply UMAP to reduce the dimensionality of the dataset
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

# Apply KMeans clustering on the reduced data
kmeans = KMeans(n_clusters=5, random_state=42)
df_original['Cluster'] = kmeans.fit_predict(X_umap) 
df['Cluster'] = kmeans.fit_predict(X_umap)

# Streamlit Interface
st.set_page_config(page_title="Countries Clustering Visualization", layout="wide")

# Header with Image
st.markdown(
    """
    <style>
    .header-container {
        text-align: center;
    }
    .header-container img {
        max-width: 100%;
        height: auto;
    }
    </style>
    <div class="header-container">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQcnEoopczYX_eLHV5kWIGUiiow7pKpxLW-bQ&s" alt="App Header">
    </div>
    """, unsafe_allow_html=True
)

# Title of the App
st.title("🌍 Countries Clustering - 2023 Dataset")

# Short description and app explanation
st.write("""
This application visualizes the clustering results of various countries in 2023, based on features like **GDP**, **Life Expectancy**, **Maternal Mortality Ratio**, and **Urban Population**.
The goal is to identify patterns and group countries into clusters that share similar characteristics across these features.
The dataset is from [Kaggle - Global Country Information Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023).
""")

# Sidebar for additional information or interactions
st.sidebar.header("Clustering Parameters & Dataset Info")
st.sidebar.write("""
Here, we have applied **KMeans clustering** to categorize countries based on key socio-economic factors. UMAP is used to reduce the dimensionality of the dataset to make it easier to visualize in 2D space.
""")

# Dataset preview
st.write("### Dataset Preview with Clusters")
st.write("The table below shows the dataset with the assigned cluster labels.")
st.dataframe(df_original.head(), width=800)


# Cluster averages based on the original dataframe
st.write("### Cluster Summary")
cluster_summary = df_original.groupby('Cluster')[features].mean()
st.write(cluster_summary)

# 2D Cluster Visualization using UMAP (with transformed and scaled data)
st.write("### 2D Clustering Visualization (UMAP Projection)")
st.write("""
Below is the 2D visualization of the clustering based on UMAP projections, using all four features (GDP, Life Expectancy, Maternal Mortality Ratio, Urban Population).
""")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=df['Cluster'], palette="Set1", ax=ax)
ax.set_title('Countries Clustering (UMAP Projection)')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')

# Display the UMAP plot
st.pyplot(fig)

# 3D Clustering Visualization using Plotly
st.write("### 3D Clustering Visualization")
st.write("""
Explore the clusters in a 3D space using features like GDP, Life Expectancy, and Maternal Mortality Ratio. You can rotate the plot and zoom in to analyze the clusters in more detail.
""")
fig = px.scatter_3d(
    x=X_scaled[:, 0], y=X_scaled[:, 1], z=X_scaled[:, 2],
    color=df_original['Cluster'], 
    title='3D Clustering Visualization (Scaled Data)',
    labels={'x': 'Maternal mortality ratio', 'y': 'Life expectancy', 'z': 'GDP'}
)
st.plotly_chart(fig)

# Interactive 2D Plotly scatter plot
st.write("### Interactive 2D Plot (GDP vs Urban population)")
st.write("""
An interactive Plotly chart of DP vs Urban population, where each country's position is colored by its cluster.
Hover over the points to view country names and cluster information.
""")
fig = px.scatter(
    df_original,
    x=X_scaled[:, 3],  
    y=X_scaled[:, 2], 
    color='Cluster',
    hover_name='Country',
    title='Interactive Clustering Plot (Scaled Data)',
    labels={'x': 'Urban_population', 'y': 'GDP'}
)
st.plotly_chart(fig)


# Displaying the list of countries and their assigned clusters
st.write("### Countries and Their Assigned Clusters")
st.write("Below is a list of countries along with their assigned clusters.")
st.dataframe(df[['Country', 'Cluster']], width=800)



 # Adding download link for the clean dataset
st.sidebar.write("### Download the clean dataset")
st.sidebar.write("""
You can download the clean dataset used for clustering in this app.
""")

# Convert the dataframe to CSV for download
csv = df.to_csv(index=False)
st.sidebar.download_button(
    label="Download clean dataset",
    data=csv,
    file_name="cleaned_countries_2023.csv",
    mime="text/csv",
)


# User Data Upload Option
st.sidebar.header("Upload your own data")
st.sidebar.write("""
You can upload your own dataset and run clustering using the same pipeline. Make sure your dataset has the same features: **GDP**, **Life Expectancy**, **Maternal Mortality Ratio**, and **Urban Population**.
""")
# Upload data from the user
uploaded_file = st.sidebar.file_uploader("Choose a file (CSV format)", type=["csv"])

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)

    # Ensure the user data has the correct columns
    if set(['GDP', 'Life expectancy', 'Maternal mortality ratio', 'Urban_population']).issubset(user_df.columns):
        user_data_scaled = scaler.transform(user_df[['GDP', 'Life expectancy', 'Maternal mortality ratio', 'Urban_population']])
        user_data_umap = umap_model.transform(user_data_scaled)
        
        # Predict clusters for the user data
        user_clusters = kmeans.predict(user_data_umap)
        user_df['Cluster'] = user_clusters
        
        st.write("### Clustering Results for Your Data")
        st.dataframe(user_df[['Country', 'Cluster']], width=800)

        # Visualize the user data clusters in 2D
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=user_data_umap[:, 0], y=user_data_umap[:, 1], hue=user_df['Cluster'], palette="Set1", ax=ax)
        ax.set_title('User Data Clustering (UMAP Projection)')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        st.pyplot(fig)
    else:
        st.error("Your data is missing one or more required columns: GDP, Life Expectancy, Maternal Mortality Ratio, or Urban Population.")
        


# Footer
st.write("---")
st.write("© 2024 - MSDE Machine Learning Project")
