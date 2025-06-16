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
from base.base_i import LoadCsvDataForPreprocess

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


if __name__ == "__main__":
    df=LoadDataAndPreprocess(
        file_path=r"C:\Users\Vishnu\Desktop\NYPD_SHOOTING_CLUSTERING\NYPD-Shooting-Incident-clustering\data\NYPD_Shooting_Incident_Data__Historic_.csv"
    ).load_and_preprocess()