import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns
import folium
from folium.plugins import (
    HeatMap,
    MarkerCluster
)
import hdbscan
from sklearn.manifold import TSNE
import plotly.express as px
from datetime import datetime
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
import os, sys
import joblib
from flask import (
    Flask,
    request,
    jsonify
)
import streamlit as ui 
from streamlit_folium import folium_static
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base.base_i import (
    LoadCsvDataForPreprocess,
    HDBSCAN
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)

sns.set(style="whitegrid", palette="pastel")

class LoadDataAndPreprocess(LoadCsvDataForPreprocess):

    def __init__(self, file_path: str):

        self.file_path = file_path

    def load_and_preprocess(self) -> pd.DataFrame:
        
        try:
            print("Loading Dataset.........")
            logger.info("Loading Dataset.........")
            df = pd.read_csv(self.file_path)

            # Drop unnecessary columns
            logger.info("Drop unnecessary columns.....")
            columns_to_drop = [
                'INCIDENT_KEY', 'X_COORD_CD', 'Y_COORD_CD', 'Lon_Lat',
                'LOC_OF_OCCUR_DESC', 'JURISDICTION_CODE', 'LOC_CLASSFCTN_DESC',
                'STATISTICAL_MURDER_FLAG'
            ]
            df = df.drop(columns=columns_to_drop,
                         errors='ignore')
            
             # Convert date/time
            logger.info("Convert date/time....")
            df['OCCUR_DATETIME'] = pd.to_datetime(
                df['OCCUR_DATE'] + ' ' + df['OCCUR_TIME'],
                format='%m/%d/%Y %H:%M:%S', errors='coerce'
            )

            # Drop rows with invalid dates
            logger.info("Drop rows with invalid dates...")
            df = df.dropna(subset=['OCCUR_DATETIME'])

            # Extract temporal features
            logger.info("Extract temporal features....")
            df['OCCUR_HOUR'] = df['OCCUR_DATETIME'].dt.hour
            df['OCCUR_DAYOFWEEK'] = df['OCCUR_DATETIME'].dt.day_name()
            df['OCCUR_MONTH'] = df['OCCUR_DATETIME'].dt.month_name()

            # Drop original data/time columns
            logger.info("Drop original data/time columns.....")
            df = df.drop(columns=['OCCUR_DATE', 'OCCUR_TIME'])

            # clean coordinates
            logger.info("clean coordinates...")
            df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]
            df = df.dropna(subset=['Latitude', 'Longitude'])

            #Handle PERP_SEX column
            logger.info("Handle PERP_SEX column.....")
            df['PERP_SEX'] = df['PERP_SEX'].replace({'(null)': np.nan})
            df['PERP_SEX'] = df['PERP_SEX'].fillna('U')  # U for Unknown
            gender_mapping = {'M': 'Male', 'F': 'Female', 'U': 'Unknown'}
            df['PERP_GENDER'] = df['PERP_SEX'].map(gender_mapping)
            
            df['PERP_GENDER_KNOWN'] = df['PERP_SEX'].isin(['M', 'F'])

            # Handle PERP_AGE_GROUP
            logger.info("Handle PERP_AGE_GROUP....")
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
            logger.info("Create flag for known age ........")
            df['PERP_AGE_KNOWN'] = df['PERP_AGE_GROUP'] != 'UNKNOWN'

            # Victim age simplification
            logger.info("Victim age simplification........")
            df['VIC_AGE_SIMPLIFIED'] = df['VIC_AGE_GROUP'].apply(
                lambda x: 'MINOR' if '<18' in str(x) else 'ADULT' if '18-24' in str(x) or '25-44' in str(x) else 'SENIOR'
            )

            # Time of day categorization
            logger.info("Time of day categorization........")
            bins = [-1, 6, 12, 18, 24]
            labels = ['Night', 'Morning', 'Afternoon', 'Evening']
            df['TIME_CATEGORY'] = pd.cut(df['OCCUR_HOUR'], bins=bins, labels=labels)

            # Create combined demographic feature
            logger.info("Create combined demographic feature......")
            df['PERP_DEMOGRAPHIC'] = df['PERP_AGE_SIMPLIFIED'] + ' ' + df['PERP_GENDER']

            # Create flag for known perpetrator
            logger.info("Create flag for known perpetrator")
            df['PERP_IDENTIFIED'] = df['PERP_AGE_KNOWN'] & df['PERP_GENDER_KNOWN']

            print(f"Preprocessing complete. Final dataset shape: {df.shape}")
            logger.info("(27302, 23)")
            logger.info(f"Preprocessing complete. Final dataset shape: {df.shape}")
            logger.info("This Operation Exited")
            return df

        except Exception as e:
            logger.error(f"Error Occured In : {e}")
            print(f"Error Occured In : {e}")
            raise 


class HDBSCANCluster(HDBSCAN):

    def __init__(
            self,
            min_cluster_size: int=15,
            min_samples: int=5,
            metric: str='haversine',
            cluster_selection_method: str='eom',
            prediction_data: bool=True
        ):

        try:
            self.min_cluster_size=min_cluster_size
            self.min_samples=min_samples
            self.metric=metric
            self.cluster_selection_method=cluster_selection_method
            self.prediction_data=prediction_data
        except Exception as e:
            logger.error(f"Error Occured In : {e}")
            print(f"Error Occured In : {e}")
            raise 

    def spatial_clustering(self, df: pd.DataFrame, 
                           save_model=True) -> tuple[pd.DataFrame, HDBSCAN]:

        try:
            logger.info("performing spratial clustering .....")

            spatial_data = df[['Latitude', 'Longitude']].copy()

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=self.metric,
                cluster_selection_method=self.cluster_selection_method,
                prediction_data=self.prediction_data
            )

            coords_rad = np.radians(spatial_data)
            df['SPATIAL_CLUSTER'] = clusterer.fit_predict(coords_rad)

            cluster_counts = df['SPATIAL_CLUSTER'].value_counts()
            df["CLUSTER_SIZE"] = df['SPATIAL_CLUSTER'].map(cluster_counts)
            print(f"Identified {cluster_counts[cluster_counts.index != -1].shape[0]} spatial clusters")
            if save_model:
                joblib.dump(clusterer, 'hdbscan_clusterer.joblib')
                logger.info("Saved clustering model to hdbscan_clusterer.joblib")
            return df, clusterer    
        except Exception as e:
            logger.error(f"Error Occured In : {e}")
            print(f"Error Occured In : {e}")
            raise   

    def evaluate_clusering(self, df: pd.DataFrame):

        try:
            logger.info("Evaluating clustering quality...")

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

            logger.info("Clustering evaluation complete")
            return metrics
        except Exception as e:
            logger.error(f"Error Occured In : {e}")
            print(f"Error Occured In : {e}")
            raise  

    def visualize_cluster_quality(self, df, metrics):

        try:
            print("Visualizing cluster quality...")

            # Prepare data
            clustered = df[df['SPATIAL_CLUSTER'] != -1]

            # 1. Cluster size distribution
            cluster_sizes = clustered['SPATIAL_CLUSTER'].value_counts()

            # 2. T-SNE visualization
            tsne = TSNE(
                n_components=2, 
                random_state=42, 
                perplexity=min(30, len(clustered)-1)
            )
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
        except Exception as e:
            logger.info(f"Error Occured In : {e}")
            raise    

    
class Pipeline:
    def __init__(self):
        pass

    def main(self):

        try:
            os.makedirs('output', exist_ok=True)
            file_path = r'C:\Users\Vishnu\Desktop\NYPD_SHOOTING_CLUSTERING\NYPD-Shooting-Incident-clustering\data\NYPD_Shooting_Incident_Data__Historic_.csv'
            if not os.path.exists(file_path):
                print(f"Error: Data file not found at {file_path}")
                return
            df=LoadDataAndPreprocess(
                file_path=file_path
            ).load_and_preprocess()
            df.to_csv("cleaned_and_cluster_label.csv", index=False)
            HdbscanAlgoritham=HDBSCANCluster(
                min_cluster_size=15,
                min_samples=5,
                metric='haversine',
                cluster_selection_method='eom',
                prediction_data=True
            )
            df, clusterer = HdbscanAlgoritham.spatial_clustering(
                df=df,
                save_model=True
            )
            clustering_metrics=HdbscanAlgoritham.evaluate_clusering(df=df)
            if clustering_metrics:
               cluster_quality_fig = HdbscanAlgoritham.visualize_cluster_quality(df, clustering_metrics)
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

        except Exception as e:
            raise