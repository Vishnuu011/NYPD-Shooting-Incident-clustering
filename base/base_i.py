from abc import ABC, abstractmethod
import folium.map
import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
import folium

class LoadCsvDataForPreprocessEstimatorMixin(ABC):
    
    def __init__(self, file_path: str):
        pass

    @abstractmethod
    def load_and_preprocess(self) -> pd.DataFrame:
        pass

class HDBSCANEstimatorMixin(ABC):

    def __init__(self):
        pass

    def spatial_clustering(df: pd.DataFrame, save_model=True):
        pass

    def evaluate_clusering(df: pd.DataFrame) -> dict:

        pass
    def visualize_cluster_quality(df, metrics) -> plt.Figure:
        pass





class AnalyzeVisualizationBaseEstimatorMixin(ABC):

    def __init__(self):
        
        pass

    def analyze_temporal(self, df: pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass

    def create_heatmap(self, df:pd.DataFrame) -> folium.Map:
        pass

    def create_cluster_map(self, df: pd.DataFrame) -> folium.Map:
        pass

    def plot_temporal_patterns(self, hourly: pd.DataFrame, 
                               daily: pd.DataFrame, 
                               monthly: pd.DataFrame) -> plt.Figure:
        pass

    def demographic_analysis(self, df: pd.DataFrame) -> plt.Figure:
        pass

    def analyze_identification(self, df: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
        pass

    def plot_identification_analysis(self, 
                                     analysis_results: dict[str, pd.DataFrame | dict]) -> plt.Figure:
        pass

    def create_identification_map(self, cluster_id: pd.DataFrame) -> folium.map:
        pass
