import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import hdbscan
from sklearn.manifold import TSNE
import plotly.express as px
from datetime import datetime
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
import joblib
import threading
import sys
from flask import Flask, request, jsonify
import streamlit as st
from streamlit_folium import folium_static

# Set style for plots
sns.set(style="whitegrid", palette="pastel")

# Load and preprocess data
def load_and_preprocess(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    columns_to_drop = [
        'INCIDENT_KEY', 'X_COORD_CD', 'Y_COORD_CD', 'Lon_Lat',
        'LOC_OF_OCCUR_DESC', 'JURISDICTION_CODE', 'LOC_CLASSFCTN_DESC',
        'STATISTICAL_MURDER_FLAG'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Convert date/time
    df['OCCUR_DATETIME'] = pd.to_datetime(
        df['OCCUR_DATE'] + ' ' + df['OCCUR_TIME'],
        format='%m/%d/%Y %H:%M:%S', errors='coerce'
    )

    # Drop rows with invalid dates
    df = df.dropna(subset=['OCCUR_DATETIME'])

    # Extract temporal features
    df['OCCUR_HOUR'] = df['OCCUR_DATETIME'].dt.hour
    df['OCCUR_DAYOFWEEK'] = df['OCCUR_DATETIME'].dt.day_name()
    df['OCCUR_MONTH'] = df['OCCUR_DATETIME'].dt.month_name()

    # Drop original date/time columns
    df = df.drop(columns=['OCCUR_DATE', 'OCCUR_TIME'])

    # Clean coordinates
    df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]
    df = df.dropna(subset=['Latitude', 'Longitude'])

    # Handle missing values
    df['PERP_RACE'] = df['PERP_RACE'].fillna('UNKNOWN')
    df['VIC_RACE'] = df['VIC_RACE'].fillna('UNKNOWN')
    df['LOCATION_DESC'] = df['LOCATION_DESC'].fillna('UNSPECIFIED')

    # Handle PERP_SEX column
    df['PERP_SEX'] = df['PERP_SEX'].replace({'(null)': np.nan})
    df['PERP_SEX'] = df['PERP_SEX'].fillna('U')  # U for Unknown
    gender_mapping = {'M': 'Male', 'F': 'Female', 'U': 'Unknown'}
    df['PERP_GENDER'] = df['PERP_SEX'].map(gender_mapping)

    # Create flag for known gender
    df['PERP_GENDER_KNOWN'] = df['PERP_SEX'].isin(['M', 'F'])

    # Handle PERP_AGE_GROUP
    df['PERP_AGE_GROUP'] = df['PERP_AGE_GROUP'].fillna('UNKNOWN')
    age_mapping = {
        '<18': 'Juvenile',
        '18-24': 'Young Adult',
        '25-44': 'Adult',
        '45-64': 'Middle-aged',
        '65+': 'Senior',
        'UNKNOWN': 'Unknown'
    }
    df['PERP_AGE_SIMPLIFIED'] = df['PERP_AGE_GROUP'].map(age_mapping)

    # Create flag for known age
    df['PERP_AGE_KNOWN'] = df['PERP_AGE_GROUP'] != 'UNKNOWN'

    # Victim age simplification
    df['VIC_AGE_SIMPLIFIED'] = df['VIC_AGE_GROUP'].apply(
        lambda x: 'MINOR' if '<18' in str(x) else 'ADULT' if '18-24' in str(x) or '25-44' in str(x) else 'SENIOR'
    )

    # Time of day categorization
    bins = [-1, 6, 12, 18, 24]
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    df['TIME_CATEGORY'] = pd.cut(df['OCCUR_HOUR'], bins=bins, labels=labels)

    # Create combined demographic feature
    df['PERP_DEMOGRAPHIC'] = df['PERP_AGE_SIMPLIFIED'] + ' ' + df['PERP_GENDER']

    # Create flag for known perpetrator
    df['PERP_IDENTIFIED'] = df['PERP_AGE_KNOWN'] & df['PERP_GENDER_KNOWN']

    print(f"Preprocessing complete. Final dataset shape: {df.shape}")
    return df

# Spatial clustering with model saving
def spatial_clustering(df, save_model=True):
    print("Performing spatial clustering...")
    # Prepare spatial data
    spatial_data = df[['Latitude', 'Longitude']].copy()

    # HDBSCAN clustering with prediction capability
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric='haversine',
        cluster_selection_method='eom',
        prediction_data=True  # Required for approximate_predict
    )

    # Convert to radians for haversine metric
    coords_rad = np.radians(spatial_data)
    df['SPATIAL_CLUSTER'] = clusterer.fit_predict(coords_rad)

    # Add cluster count information
    cluster_counts = df['SPATIAL_CLUSTER'].value_counts()
    df['CLUSTER_SIZE'] = df['SPATIAL_CLUSTER'].map(cluster_counts)

    print(f"Identified {cluster_counts[cluster_counts.index != -1].shape[0]} spatial clusters")
    
    # Save the clusterer model for future predictions
    if save_model:
        joblib.dump(clusterer, 'hdbscan_clusterer.joblib')
        print("Saved clustering model to hdbscan_clusterer.joblib")
    
    return df, clusterer

def evaluate_clustering(df):
    """
    Evaluate clustering quality using multiple metrics
    """
    print("Evaluating clustering quality...")

    # Prepare data - only clustered points (exclude noise)
    clustered = df[df['SPATIAL_CLUSTER'] != -1]

    if len(clustered) < 2:
        print("Not enough clustered points for evaluation")
        return None

    coords = clustered[['Latitude', 'Longitude']].values
    labels = clustered['SPATIAL_CLUSTER'].values

    # Calculate evaluation metrics
    metrics = {}

    # Silhouette Score (-1 to 1, higher is better)
    try:
        metrics['silhouette'] = silhouette_score(coords, labels, metric='haversine')
    except:
        metrics['silhouette'] = -1  # Error value

    # Calinski-Harabasz Index (higher is better)
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(coords, labels)
    except:
        metrics['calinski_harabasz'] = -1

    # Davies-Bouldin Index (lower is better)
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(coords, labels)
    except:
        metrics['davies_bouldin'] = float('inf')

    # Cluster separation index (custom metric)
    cluster_centers = clustered.groupby('SPATIAL_CLUSTER')[['Latitude', 'Longitude']].mean()
    min_distances = []
    for center in cluster_centers.values:
        distances = np.linalg.norm(cluster_centers.values - center, axis=1)
        min_distances.append(np.min(distances[distances > 0]))
    metrics['avg_min_cluster_distance'] = np.mean(min_distances)

    # Intra-cluster density
    intra_density = []
    for cluster_id in clustered['SPATIAL_CLUSTER'].unique():
        cluster_points = clustered[clustered['SPATIAL_CLUSTER'] == cluster_id][['Latitude', 'Longitude']].values
        center = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - center, axis=1)
        intra_density.append(np.mean(distances))
    metrics['avg_intra_cluster_density'] = np.mean(intra_density)

    print("Clustering evaluation complete")
    return metrics

# Add visualization for cluster evaluation
def visualize_cluster_quality(df, metrics):
    """
    Create visualizations to evaluate cluster quality
    """
    print("Visualizing cluster quality...")

    # Prepare data
    clustered = df[df['SPATIAL_CLUSTER'] != -1]

    # 1. Cluster size distribution
    cluster_sizes = clustered['SPATIAL_CLUSTER'].value_counts()

    # 2. T-SNE visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(clustered)-1))
    tsne_results = tsne.fit_transform(clustered[['Latitude', 'Longitude']])

    # Create figure
    fig, ax = plt.subplots(2, 2, figsize=(18, 16))

    # Cluster size distribution
    sns.barplot(x=cluster_sizes.values, y=cluster_sizes.index.astype(str),
                ax=ax[0, 0], palette='viridis')
    ax[0, 0].set_title('Cluster Size Distribution', fontsize=16)
    ax[0, 0].set_xlabel('Number of Points', fontsize=14)
    ax[0, 0].set_ylabel('Cluster ID', fontsize=14)

    # T-SNE visualization
    scatter = ax[0, 1].scatter(
        tsne_results[:, 0], tsne_results[:, 1],
        c=clustered['SPATIAL_CLUSTER'], cmap='tab20', alpha=0.6
    )
    ax[0, 1].set_title('t-SNE Cluster Visualization', fontsize=16)
    ax[0, 1].set_xlabel('t-SNE 1', fontsize=14)
    ax[0, 1].set_ylabel('t-SNE 2', fontsize=14)
    plt.colorbar(scatter, ax=ax[0, 1], label='Cluster ID')

    # Metric visualization
    metric_names = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    metric_values = [metrics.get(m, 0) for m in metric_names]
    metric_labels = ['Silhouette (↑)', 'Calinski-Harabasz (↑)', 'Davies-Bouldin (↓)']

    # Create radar chart for metrics
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False)
    values = np.array(metric_values)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    ax[1, 0] = plt.subplot(2, 2, 3, polar=True)
    ax[1, 0].plot(angles, values, 'o-', linewidth=2)
    ax[1, 0].fill(angles, values, alpha=0.25)
    ax[1, 0].set_xticks(angles[:-1])
    ax[1, 0].set_xticklabels(metric_labels)
    ax[1, 0].set_title('Cluster Quality Metrics', fontsize=16, pad=20)

    # Boxplot of intra-cluster distances
    intra_distances = []
    for cluster_id in clustered['SPATIAL_CLUSTER'].unique():
        cluster_points = clustered[clustered['SPATIAL_CLUSTER'] == cluster_id][['Latitude', 'Longitude']].values
        center = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - center, axis=1)
        intra_distances.append(distances)

    ax[1, 1].boxplot(intra_distances, vert=False)
    ax[1, 1].set_title('Intra-Cluster Distance Distribution', fontsize=16)
    ax[1, 1].set_xlabel('Distance from Cluster Center (degrees)', fontsize=14)
    ax[1, 1].set_ylabel('Cluster', fontsize=14)

    plt.tight_layout()
    return fig

# Temporal pattern analysis
def analyze_temporal(df):
    print("Analyzing temporal patterns...")
    # Hourly distribution
    hourly = df.groupby('OCCUR_HOUR').size().reset_index(name='COUNT')

    # Day of week distribution
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily = df.groupby('OCCUR_DAYOFWEEK').size().reindex(day_order).reset_index(name='COUNT')

    # Monthly distribution
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
    monthly = df.groupby('OCCUR_MONTH').size().reindex(month_order).reset_index(name='COUNT')

    return hourly, daily, monthly

# Visualization functions
def create_heatmap(df):
    print("Creating heatmap...")
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles='cartodbpositron')
    heat_data = [[row['Latitude'], row['Longitude']] for _, row in df.iterrows()]
    HeatMap(heat_data, radius=12, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
    return m

def create_cluster_map(df):
    print("Creating cluster map...")
    cluster_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles='cartodbpositron')

    # Add heatmap layer
    HeatMap(df[['Latitude', 'Longitude']].values, radius=10).add_to(cluster_map)

    # Add cluster markers
    for cluster_id in df['SPATIAL_CLUSTER'].unique():
        if cluster_id != -1:  # Skip noise points
            cluster_data = df[df['SPATIAL_CLUSTER'] == cluster_id]

            # Create popup with cluster info
            cluster_size = len(cluster_data)
            popup_text = f"<b>Cluster {cluster_id}</b><br>"
            popup_text += f"Incidents: {cluster_size}<br>"
            popup_text += f"Top Location: {cluster_data['LOCATION_DESC'].value_counts().index[0]}"

            # Get cluster center
            center = [cluster_data['Latitude'].mean(), cluster_data['Longitude'].mean()]

            # Add marker
            folium.CircleMarker(
                location=center,
                radius=8 + (cluster_size/50),
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                popup=popup_text
            ).add_to(cluster_map)

    return cluster_map

def plot_temporal_patterns(hourly, daily, monthly):
    print("Plotting temporal patterns...")
    fig, ax = plt.subplots(3, 1, figsize=(14, 18))

    # Hourly plot
    sns.barplot(x='OCCUR_HOUR', y='COUNT', data=hourly, ax=ax[0], color='royalblue')
    ax[0].set_title('Shooting Incidents by Hour of Day', fontsize=16)
    ax[0].set_xlabel('Hour of Day', fontsize=14)
    ax[0].set_ylabel('Incident Count', fontsize=14)
    ax[0].grid(True, linestyle='--', alpha=0.7)

    # Daily plot
    sns.barplot(x='OCCUR_DAYOFWEEK', y='COUNT', data=daily, ax=ax[1], color='mediumseagreen')
    ax[1].set_title('Shooting Incidents by Day of Week', fontsize=16)
    ax[1].set_xlabel('Day of Week', fontsize=14)
    ax[1].set_ylabel('Incident Count', fontsize=14)
    ax[1].grid(True, linestyle='--', alpha=0.7)

    # Monthly plot
    sns.barplot(x='OCCUR_MONTH', y='COUNT', data=monthly, ax=ax[2], color='tomato')
    ax[2].set_title('Shooting Incidents by Month', fontsize=16)
    ax[2].set_xlabel('Month', fontsize=14)
    ax[2].set_ylabel('Incident Count', fontsize=14)
    ax[2].grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig

def demographic_analysis(df):
    print("Performing demographic analysis...")
    fig, ax = plt.subplots(3, 2, figsize=(20, 20))

    # Perpetrator age distribution
    age_order = ['Juvenile', 'Young Adult', 'Adult', 'Middle-aged', 'Senior', 'Unknown']
    age_counts = df['PERP_AGE_SIMPLIFIED'].value_counts().reindex(age_order).reset_index()
    # Explicitly rename columns
    age_counts.columns = ['Age_Group', 'Count']
    sns.barplot(x='Age_Group', y='Count', data=age_counts, ax=ax[0, 0], palette='viridis')
    ax[0, 0].set_title('Perpetrator Age Distribution', fontsize=16)
    ax[0, 0].set_xlabel('Age Group', fontsize=14)
    ax[0, 0].set_ylabel('Count', fontsize=14)
    ax[0, 0].tick_params(axis='x', rotation=45)

    # Perpetrator gender distribution
    gender_counts = df['PERP_GENDER'].value_counts().reset_index()
    # Explicitly rename columns
    gender_counts.columns = ['Gender', 'Count']
    sns.barplot(x='Gender', y='Count', data=gender_counts, ax=ax[0, 1], palette='mako')
    ax[0, 1].set_title('Perpetrator Gender Distribution', fontsize=16)
    ax[0, 1].set_xlabel('Gender', fontsize=14)
    ax[0, 1].set_ylabel('Count', fontsize=14)

    # Victim race distribution
    vic_race = df['VIC_RACE'].value_counts().reset_index()
    # Explicitly rename columns
    vic_race.columns = ['Race', 'Count']
    sns.barplot(y='Race', x='Count', data=vic_race, ax=ax[1, 0], palette='rocket')
    ax[1, 0].set_title('Victim Race Distribution', fontsize=16)
    ax[1, 0].set_xlabel('Count', fontsize=14)
    ax[1, 0].set_ylabel('Race', fontsize=14)

    # Location types
    top_locations = df['LOCATION_DESC'].value_counts().head(10).reset_index()
    # Explicitly rename columns
    top_locations.columns = ['Location_Type', 'Count']
    sns.barplot(y='Location_Type', x='Count', data=top_locations, ax=ax[1, 1], palette='flare')
    ax[1, 1].set_title('Top 10 Location Types', fontsize=16)
    ax[1, 1].set_xlabel('Count', fontsize=14)
    ax[1, 1].set_ylabel('Location Type', fontsize=14)

    # Identification rate by borough
    borough_id = df.groupby('BORO')['PERP_IDENTIFIED'].mean().sort_values().reset_index()
    # Explicitly rename columns
    borough_id.columns = ['Borough', 'Identification_Rate']
    sns.barplot(y='Borough', x='Identification_Rate', data=borough_id, ax=ax[2, 0], palette='crest')
    ax[2, 0].set_title('Perpetrator Identification Rate by Borough', fontsize=16)
    ax[2, 0].set_xlabel('Identification Rate', fontsize=14)
    ax[2, 0].set_ylabel('Borough', fontsize=14)
    ax[2, 0].set_xlim(0, 1)

    # Identification rate by time of day
    time_id = df.groupby('TIME_CATEGORY')['PERP_IDENTIFIED'].mean().reset_index()
    # Explicitly rename columns
    time_id.columns = ['Time_Category', 'Identification_Rate']
    sns.barplot(x='Time_Category', y='Identification_Rate', data=time_id, ax=ax[2, 1], palette='magma')
    ax[2, 1].set_title('Identification Rate by Time of Day', fontsize=16)
    ax[2, 1].set_xlabel('Time of Day', fontsize=14)
    ax[2, 1].set_ylabel('Identification Rate', fontsize=14)
    ax[2, 1].set_ylim(0, 1)

    plt.tight_layout()
    return fig

# Perpetrator identification analysis
def analyze_identification(df):
    print("Analyzing identification patterns...")
    results = {}

    # Overall identification rates
    results['gender_id_rate'] = df['PERP_GENDER_KNOWN'].mean()
    results['age_id_rate'] = df['PERP_AGE_KNOWN'].mean()
    results['full_id_rate'] = df['PERP_IDENTIFIED'].mean()

    # Identification by precinct
    precinct_id = df.groupby('PRECINCT').agg(
        incident_count=('SPATIAL_CLUSTER', 'count'),
        gender_id_rate=('PERP_GENDER_KNOWN', 'mean'),
        full_id_rate=('PERP_IDENTIFIED', 'mean')
    ).reset_index().sort_values('full_id_rate')

    # Identification by borough
    borough_id = df.groupby('BORO').agg(
        incident_count=('SPATIAL_CLUSTER', 'count'),
        gender_id_rate=('PERP_GENDER_KNOWN', 'mean'),
        full_id_rate=('PERP_IDENTIFIED', 'mean')
    ).reset_index()

    # Identification by time of day
    time_id = df.groupby('OCCUR_HOUR')['PERP_IDENTIFIED'].mean().reset_index()

    # Identification by location
    location_id = df.groupby('LOCATION_DESC').agg(
        incident_count=('SPATIAL_CLUSTER', 'count'),
        id_rate=('PERP_IDENTIFIED', 'mean')
    ).reset_index().sort_values('id_rate')

    # Identification by cluster
    cluster_id = df[df['SPATIAL_CLUSTER'] != -1].groupby('SPATIAL_CLUSTER').agg(
        incident_count=('SPATIAL_CLUSTER', 'count'),
        id_rate=('PERP_IDENTIFIED', 'mean'),
        avg_lat=('Latitude', 'mean'),
        avg_lon=('Longitude', 'mean')
    ).reset_index()

    return {
        'results': results,
        'precinct_id': precinct_id,
        'borough_id': borough_id,
        'time_id': time_id,
        'location_id': location_id,
        'cluster_id': cluster_id
    }

# Visualization for identification analysis
def plot_identification_analysis(analysis_results):
    print("Visualizing identification analysis...")
    # Create figure
    fig, ax = plt.subplots(2, 2, figsize=(20, 16))

    # Borough identification rates
    borough_id = analysis_results['borough_id']
    sns.barplot(x='full_id_rate', y='BORO', data=borough_id,
                ax=ax[0, 0], palette='viridis')
    ax[0, 0].set_title('Perpetrator Identification Rate by Borough', fontsize=16)
    ax[0, 0].set_xlabel('Identification Rate', fontsize=14)
    ax[0, 0].set_ylabel('Borough', fontsize=14)
    ax[0, 0].set_xlim(0, 1)

    # Time of day identification rates
    time_id = analysis_results['time_id']
    sns.lineplot(x='OCCUR_HOUR', y='PERP_IDENTIFIED', data=time_id,
                 ax=ax[0, 1], color='royalblue', marker='o')
    ax[0, 1].set_title('Identification Rate by Hour of Day', fontsize=16)
    ax[0, 1].set_xlabel('Hour of Day', fontsize=14)
    ax[0, 1].set_ylabel('Identification Rate', fontsize=14)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].grid(True, linestyle='--', alpha=0.7)

    # Precinct identification rates (worst 10)
    worst_precincts = analysis_results['precinct_id'].head(10)
    sns.barplot(x='full_id_rate', y='PRECINCT', data=worst_precincts,
                ax=ax[1, 0], palette='rocket')
    ax[1, 0].set_title('Top 10 Precincts with Lowest Identification Rates', fontsize=16)
    ax[1, 0].set_xlabel('Identification Rate', fontsize=14)
    ax[1, 0].set_ylabel('Precinct', fontsize=14)

    # Location identification rates (worst 10 with sufficient incidents)
    location_id = analysis_results['location_id']
    location_id = location_id[location_id['incident_count'] > 10]  # Filter for meaningful locations
    worst_locations = location_id.head(10)
    sns.barplot(x='id_rate', y='LOCATION_DESC', data=worst_locations,
                ax=ax[1, 1], palette='mako')
    ax[1, 1].set_title('Top 10 Locations with Lowest Identification Rates', fontsize=16)
    ax[1, 1].set_xlabel('Identification Rate', fontsize=14)
    ax[1, 1].set_ylabel('Location Type', fontsize=14)

    plt.tight_layout()
    return fig

# Create identification cluster map
def create_identification_map(cluster_id):
    print("Creating identification cluster map...")
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles='cartodbpositron')

    # Add each cluster
    for _, row in cluster_id.iterrows():
        # Determine color based on identification rate
        if row['id_rate'] < 0.3:
            color = 'red'
        elif row['id_rate'] < 0.6:
            color = 'orange'
        else:
            color = 'green'

        # Create popup content
        popup_text = f"""
        <b>Cluster {row['SPATIAL_CLUSTER']}</b><br>
        Identification Rate: {row['id_rate']:.1%}<br>
        Incidents: {row['incident_count']}
        """

        # Add circle marker
        folium.CircleMarker(
            location=[row['avg_lat'], row['avg_lon']],
            radius=10 + (row['incident_count']/20),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed;
                bottom: 50px; left: 50px; width: 150px; height: 100px;
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white; padding:10px;">
      <p><b>Identification Rate</b></p>
      <p><span style="background:red; width:15px; height:15px; display:inline-block;"></span> &lt; 30%</p>
      <p><span style="background:orange; width:15px; height:15px; display:inline-block;"></span> 30-60%</p>
      <p><span style="background:green; width:15px; height:15px; display:inline-block;"></span> &gt; 60%</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

# Generate insights report
def generate_insights(df, analysis_results):
    print("Generating insights report...")
    insights = []
    results = analysis_results['results']
    borough_id = analysis_results['borough_id']
    precinct_id = analysis_results['precinct_id']
    cluster_id = analysis_results['cluster_id']

    insights.append("NYPD SHOOTING INCIDENT ANALYSIS REPORT")
    insights.append("=" * 50)
    insights.append(f"Total Incidents Analyzed: {len(df)}")
    insights.append(f"Geographical Clusters Identified: {cluster_id[cluster_id['incident_count'] > 15].shape[0]}")
    insights.append("")

    # Identification insights
    insights.append("PERPETRATOR IDENTIFICATION INSIGHTS:")
    insights.append(f"- Overall identification rate: {results['full_id_rate']:.1%}")
    insights.append(f"- Gender identification rate: {results['gender_id_rate']:.1%}")
    insights.append(f"- Age identification rate: {results['age_id_rate']:.1%}")

    # Borough insights
    borough_insights = ["- Borough identification rates:"]
    for _, row in borough_id.iterrows():
        borough_insights.append(f"  - {row['BORO']}: {row['full_id_rate']:.1%} ({row['incident_count']} incidents)")
    insights.append("\n".join(borough_insights))

    # Precinct insights
    worst_precinct = precinct_id.iloc[0]
    best_precinct = precinct_id.iloc[-1]
    insights.append(f"- Precinct with lowest identification: #{worst_precinct['PRECINCT']} ({worst_precinct['full_id_rate']:.1%})")
    insights.append(f"- Precinct with highest identification: #{best_precinct['PRECINCT']} ({best_precinct['full_id_rate']:.1%})")

    # Cluster insights
    cluster_insights = ["- High-risk clusters with low identification rates:"]
    low_id_clusters = cluster_id[(cluster_id['id_rate'] < 0.4) & (cluster_id['incident_count'] > 20)]
    if not low_id_clusters.empty:
        for _, row in low_id_clusters.iterrows():
            cluster_insights.append(f"  - Cluster {row['SPATIAL_CLUSTER']}: {row['id_rate']:.1%} ID rate ({row['incident_count']} incidents)")
    else:
        cluster_insights.append("  - No clusters meet high-risk criteria")
    insights.append("\n".join(cluster_insights))

    # Temporal insights
    peak_hour = df.groupby('OCCUR_HOUR').size().idxmax()
    id_at_peak = analysis_results['time_id'].loc[analysis_results['time_id']['OCCUR_HOUR'] == peak_hour, 'PERP_IDENTIFIED'].values[0]
    insights.append(f"- Peak incident hour: {peak_hour}:00 (ID rate: {id_at_peak:.1%})")

    # Location insights
    top_location = df['LOCATION_DESC'].value_counts().index[0]
    top_location_id = analysis_results['location_id'][analysis_results['location_id']['LOCATION_DESC'] == top_location]['id_rate'].values[0]
    insights.append(f"- Most common location: {top_location} (ID rate: {top_location_id:.1%})")

    # Recommendations
    insights.append("\nACTIONABLE RECOMMENDATIONS:")
    insights.append("1. Increase investigative resources in Bronx precincts with low identification rates")
    insights.append("2. Enhance witness protection programs in high-risk clusters with ID rates below 40%")
    insights.append("3. Deploy mobile surveillance units to locations with high incidents and low identification")
    insights.append("4. Implement community engagement programs in precincts with persistent low identification")
    insights.append("5. Focus forensic resources on peak incident hours (evening/night)")

    # Save insights
    with open('output/analysis_insights.txt', 'w') as f:
        f.write("\n".join(insights))

    return insights

# Cluster prediction function
def predict_cluster(latitude, longitude, clusterer=None):
    """Predict cluster for new coordinates"""
    if clusterer is None:
        try:
            clusterer = joblib.load('hdbscan_clusterer.joblib')
        except:
            raise ValueError("Clustering model not found. Run analysis first.")
    
    # Convert to radians
    point_rad = np.radians([[latitude, longitude]])
    
    # Get prediction
    cluster, prob = hdbscan.approximate_predict(clusterer, point_rad)
    
    return cluster[0], prob[0]

# Flask API setup
def run_flask_app():
    app = Flask(__name__)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        try:
            lat = float(data['latitude'])
            lon = float(data['longitude'])
            cluster, prob = predict_cluster(lat, lon)
            return jsonify({
                'cluster': int(cluster),
                'probability': float(prob),
                'latitude': lat,
                'longitude': lon
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
            
    @app.route('/')
    def index():
        return "NYPD Shooting Cluster Prediction API - POST /predict with {'latitude': <value>, 'longitude': <value>}"
    
    print("Starting Flask API server at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)

# Streamlit UI
def run_streamlit_ui():
    st.set_page_config(page_title="NYPD Shooting Analysis", layout="wide")
    
    st.title("NYPD Shooting Incident Cluster Prediction")
    st.markdown("""
    This tool predicts crime clusters based on historical NYPD shooting data.
    Enter coordinates to see which cluster the location belongs to.
    """)
    
    # Load cluster map if available
    cluster_map_exists = os.path.exists('output/cluster_map.html')
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Predict New Location")
        lat = st.number_input("Latitude", value=40.7128, format="%.6f", step=0.0001)
        lon = st.number_input("Longitude", value=-74.0060, format="%.6f", step=0.0001)
        
        if st.button("Predict Cluster", help="Predict cluster for the entered coordinates"):
            try:
                cluster, prob = predict_cluster(lat, lon)
                
                if cluster == -1:
                    st.warning("This location doesn't belong to any major crime cluster")
                else:
                    st.success(f"Predicted Cluster: **{cluster}** (Probability: {prob:.2%})")
                
                # Show on map
                st.subheader("Location on Map")
                m = folium.Map(location=[lat, lon], zoom_start=14)
                folium.Marker(
                    [lat, lon], 
                    popup=f"Predicted Cluster: {cluster}",
                    icon=folium.Icon(color='red' if cluster == -1 else 'blue')
                ).add_to(m)
                folium_static(m)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    with col2:
        st.subheader("Cluster Map")
        if cluster_map_exists:
            # Display the cluster map
            with open('output/cluster_map.html', 'r') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600)
        else:
            st.warning("Cluster map not available. Run the analysis first.")
    
    st.divider()
    st.subheader("Cluster Information")
    st.markdown("""
    - **Cluster -1**: Noise points (not part of any cluster)
    - **Other clusters**: Grouped crime hotspots identified by the algorithm
    - **Probability**: Confidence level of the cluster assignment
    """)
    
    st.subheader("Analysis Insights")
    if os.path.exists('output/analysis_insights.txt'):
        with open('output/analysis_insights.txt', 'r') as f:
            insights = f.read()
        st.text(insights)
    else:
        st.warning("No insights available. Run the analysis first.")

# Main analysis workflow
def main():
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Check for command-line arguments
    if '--api' in sys.argv:
        run_flask_app()
        return
    elif '--ui' in sys.argv:
        run_streamlit_ui()
        return
    
    # Load and preprocess data
    file_path = '/content/NYPD_Shooting_Incident_Data__Historic_.csv'
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return
    
    df = load_and_preprocess(file_path)
    
    # Spatial clustering (with model saving)
    df, clusterer = spatial_clustering(df)
    
    # Evaluate clustering
    clustering_metrics = evaluate_clustering(df)
    if clustering_metrics:
        cluster_quality_fig = visualize_cluster_quality(df, clustering_metrics)
        cluster_quality_fig.savefig('output/cluster_quality.png', dpi=300, bbox_inches='tight')

        # Add metrics to insights
        metrics_text = "\nCLUSTERING QUALITY METRICS:\n"
        metrics_text += f"- Silhouette Score: {clustering_metrics['silhouette']:.3f} (range: -1 to 1, higher better)\n"
        metrics_text += f"- Calinski-Harabasz Index: {clustering_metrics['calinski_harabasz']:.1f} (higher better)\n"
        metrics_text += f"- Davies-Bouldin Index: {clustering_metrics['davies_bouldin']:.3f} (lower better)\n"
        metrics_text += f"- Average Intra-Cluster Density: {clustering_metrics['avg_intra_cluster_density']:.6f} degrees\n"
        metrics_text += f"- Average Minimum Cluster Distance: {clustering_metrics['avg_min_cluster_distance']:.6f} degrees"

        with open('output/clustering_metrics.txt', 'w') as f:
            f.write(metrics_text)

    # Temporal analysis
    hourly, daily, monthly = analyze_temporal(df)

    # Identification analysis
    id_analysis = analyze_identification(df)

    # Visualizations
    heatmap = create_heatmap(df)
    cluster_map = create_cluster_map(df)
    temporal_fig = plot_temporal_patterns(hourly, daily, monthly)
    demo_fig = demographic_analysis(df)
    id_fig = plot_identification_analysis(id_analysis)
    id_map = create_identification_map(id_analysis['cluster_id'])

    # Generate insights
    insights = generate_insights(df, id_analysis)

    # Save outputs
    print("Saving outputs...")
    heatmap.save('output/heatmap.html')
    cluster_map.save('output/cluster_map.html')
    id_map.save('output/identification_map.html')
    temporal_fig.savefig('output/temporal_patterns.png', dpi=300, bbox_inches='tight')
    demo_fig.savefig('output/demographics.png', dpi=300, bbox_inches='tight')
    id_fig.savefig('output/identification_analysis.png', dpi=300, bbox_inches='tight')

    # Save data
    df.to_csv('output/processed_data.csv', index=False)

    print("Analysis complete!")
    print("\nKey Insights:")
    for insight in insights[:15]:  # Print first 15 insights
        print(f"- {insight}")
        
    print("\nNext steps:")
    print("1. Run the API server: python script.py --api")
    print("2. Launch the Streamlit UI: python script.py --ui")

if __name__ == "__main__":
    main()