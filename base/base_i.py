from abc import ABC, abstractmethod
import folium.map
import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, List
import folium
import hdbscan

class LoadCsvDataForPreprocessEstimatorMixin(ABC):
    
    def __init__(self, file_path: str):
        pass

    @abstractmethod
    def load_and_preprocess(self) -> pd.DataFrame:
        pass


class HDBSCANEstimatorMixin(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def spatial_clustering(self, df: pd.DataFrame, save_model=True) -> tuple[pd.DataFrame, hdbscan.HDBSCAN]:
        pass

    @abstractmethod
    def evaluate_clustering(self, df: pd.DataFrame) -> dict:
        pass

    @abstractmethod
    def visualize_cluster_quality(self, df: pd.DataFrame, metrics: dict) -> plt.Figure:
        pass




class AnalyzeVisualizationBaseEstimatorMixin(ABC):

    def __init__(self):
        
        pass
    
    @abstractmethod
    def analyze_temporal(self, df: pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass
    
    @abstractmethod
    def create_heatmap(self, df:pd.DataFrame) -> folium.Map:
        pass

    def create_cluster_map(self, df: pd.DataFrame) -> folium.Map:
        pass
    
    @abstractmethod
    def plot_temporal_patterns(self, hourly: pd.DataFrame, 
                               daily: pd.DataFrame, 
                               monthly: pd.DataFrame) -> plt.Figure:
        pass
    
    @abstractmethod
    def demographic_analysis(self, df: pd.DataFrame) -> plt.Figure:
        pass
    
    @abstractmethod
    def analyze_identification(self, df: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
        pass
    
    @abstractmethod
    def plot_identification_analysis(self, 
                                     analysis_results: dict[str, pd.DataFrame | dict]) -> plt.Figure:
        pass
    
    @abstractmethod
    def create_identification_map(self, cluster_id: pd.DataFrame) -> folium.map:
        pass




class InsightsGeneraterMixin(ABC):

    def __init__(self):

        pass
    
    @abstractmethod
    def generate_insights(self, 
                          df: pd.DataFrame,
                          analysis_results: dict[str, pd.DataFrame | dict]
                          ) -> list:
        pass

class BasePredictMixin(ABC):
    @abstractmethod
    def predict_cluster(self, latitude: float, longitude: float, clusterer=None) -> tuple[int, float]:
        pass

class ApplicationRunMixin(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def run_flask_app() -> None:
        pass

    @abstractmethod
    def run_streamlit_ui() -> None:
        pass