import folium.map
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
    LoadCsvDataForPreprocessEstimatorMixin,
    HDBSCANEstimatorMixin,
    AnalyzeVisualizationBaseEstimatorMixin
)
import warnings 
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)

sns.set(style="whitegrid", palette="pastel")

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise e
    

class LoadDataAndPreprocess(LoadCsvDataForPreprocessEstimatorMixin):

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


class HDBSCANCluster(HDBSCANEstimatorMixin):

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
                           save_model=True) -> tuple[pd.DataFrame, hdbscan.HDBSCAN]:

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
                save_object("models/hdbscan_clusterer.joblib", clusterer)
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


class AnalyzeVisualization(AnalyzeVisualizationBaseEstimatorMixin):

    def __init__(self):

        pass   

    def analyze_temporal(self, 
                         df: pd.DataFrame) ->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        try:
            logger.info("Analyzing temporal patterns....")

            #Hourly distributhon
            hourly = df.groupby('OCCUR_HOUR').size().reset_index(name='COUNT')

            # Day of week distribution
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily = df.groupby('OCCUR_DAYOFWEEK').size().reindex(day_order).reset_index(name='COUNT'   )

            # Monthly distribution
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly = df.groupby('OCCUR_MONTH').size().reindex(month_order).reset_index(name='COUNT')
            logger.info("analyze_temporal Operation Exited...")
            return hourly, daily, monthly
        except Exception as e:
            logger.error(f"Error Occured In : {e}")
            print(f"Error Occured In : {e}")
            raise  

    def create_heatmap(self, df: pd.DataFrame) -> folium.map:

        try:
            logger.info("Creating heatmap...")
            m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles='cartodbpositron')
            heat_data = [[row['Latitude'], row['Longitude']] for _, row in df.iterrows()]
            HeatMap(heat_data, radius=12, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
            logger.info("create_map Operation Exited...")
            return m
        except Exception as e:
            logger.error(f"Error Occured In : {e}")
            raise   

    def create_cluster_map(self, df: pd.DataFrame) -> folium.map:

        try:
            print("Creating cluster map...")
            cluster_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles='cartodbpositron')

            # Add heatmap layer
            HeatMap(df[['Latitude', 'Longitude']].values, radius=10).add_to(cluster_map)

            # Add cluster markers
            for cluster_id in df['SPATIAL_CLUSTER'].unique():
                if cluster_id == -1:  # Skip noise points
                    continue
                    
                cluster_data = df[df['SPATIAL_CLUSTER'] == cluster_id]
                
                # Skip empty clusters
                if cluster_data.empty:
                    continue
                    
                cluster_size = len(cluster_data)
                
                # Handle location description safely
                try:
                    # Check if we have location data
                    if cluster_data['LOCATION_DESC'].notna().any():
                        top_location = cluster_data['LOCATION_DESC'].value_counts().index[0]
                    else:
                        top_location = "UNKNOWN"
                except IndexError:
                    top_location = "UNKNOWN"
                except Exception:
                    top_location = "UNKNOWN"
                
                # Create popup with cluster info
                popup_text = f"<b>Cluster {cluster_id}</b><br>"
                popup_text += f"Incidents: {cluster_size}<br>"
                popup_text += f"Top Location: {top_location}"

                # Get cluster center - handle potential NaN values
                try:
                    center = [
                        cluster_data['Latitude'].mean(),
                        cluster_data['Longitude'].mean()
                    ]
                    # Validate coordinates
                    if np.isnan(center[0]) or np.isnan(center[1]):
                        center = [40.7128, -74.0060]  # Default NYC center
                except Exception:
                    center = [40.7128, -74.0060]

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
            
            print("Cluster map created successfully")
            return cluster_map
        except Exception as e:
            logger.info(f"Error Occured In : {e}")
            raise  

    def plot_temporal_patterns(
            self, 
            hourly, 
            daily, 
            monthly
    ) -> plt.Figure:
        
        try:
            logger.info("Plotting temporal patterns...")
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
            logger.info("plot_temporal_patterns Operation Exited...")
            return fig
        except Exception as e:
            logger.info(f"Error Occured In : {e}")
            raise

    def demographic_analysis(self, df: pd.DataFrame) -> plt.Figure:

        try:
            logger.info("Performing demographic analysis...")
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
            logger.info("demographic_analysis Operation Exited...")
            return fig
        except Exception as e:
            logger.error(f"Error Occured In : {e}")
            raise  

    def analyze_identification(self, 
                               df: pd.DataFrame) -> dict[str, pd.DataFrame | dict]: 

        try:
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
            logger.info("analyze_identification Operation Exited...")
            return {
                'results': results,
                'precinct_id': precinct_id,
                'borough_id': borough_id,
                'time_id': time_id,
                'location_id': location_id,
                'cluster_id': cluster_id
            }
        except Exception as e:
            logger.error(f"Error Occured In : {e}")
            raise  

    def plot_identification_analysis(self, 
                                     analysis_results: dict[str, pd.DataFrame | dict]) -> plt.Figure:
        
        try:
            logger.info("Visualizing identification analysis...")
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
            logger.info("plot_identification_analysis Operation Exited...")
            return fig 
        except Exception as e:
            logger.error(f"Error Occured In : {e}")    
            raise 

    def create_identification_map(self, cluster_id: pd.DataFrame) -> folium.map:

        try:
            logger.info("Creating identification cluster map...")
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
            logger.info("create_identification_map Operation Exited...")
            return m
        except Exception as e:
            logger.error(f"Error Occured In : {e}")   
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
            #df.to_csv("cleaned_and_cluster_label.csv", index=False)
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
            
            AnalyzeVisualizationoperation=AnalyzeVisualization()
            # Temporal analysis
            hourly, daily, monthly = AnalyzeVisualizationoperation.analyze_temporal(df)

            # Identification analysis
            id_analysis = AnalyzeVisualizationoperation.analyze_identification(df)

            # Visualizations
            heatmap = AnalyzeVisualizationoperation.create_heatmap(df)
            cluster_map = AnalyzeVisualizationoperation.create_cluster_map(df)
            temporal_fig = AnalyzeVisualizationoperation.plot_temporal_patterns(hourly, daily, monthly)
            demo_fig = AnalyzeVisualizationoperation.demographic_analysis(df)
            id_fig = AnalyzeVisualizationoperation.plot_identification_analysis(id_analysis)
            id_map = AnalyzeVisualizationoperation.create_identification_map(id_analysis['cluster_id'])

            # Generate insights
            #insights = generate_insights(df, id_analysis)

            # Save outputs
            logger.info("Saving outputs...")
            heatmap.save('output/heatmap.html')
            cluster_map.save('output/cluster_map.html')
            id_map.save('output/identification_map.html')
            temporal_fig.savefig('output/temporal_patterns.png', dpi=300, bbox_inches='tight')
            demo_fig.savefig('output/demographics.png', dpi=300, bbox_inches='tight')
            id_fig.savefig('output/identification_analysis.png', dpi=300, bbox_inches='tight')

            # Save data
            df.to_csv('output/processed_data.csv', index=False)

            logger.info("Analysis complete!")       

        except Exception as e:
            raise


