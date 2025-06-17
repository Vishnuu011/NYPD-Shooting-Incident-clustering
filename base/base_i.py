from abc import ABC, abstractmethod
import pandas as  pd

class LoadCsvDataForPreprocess(ABC):
    
    def __init__(self, file_path: str):
        pass

    @abstractmethod
    def load_and_preprocess(self) -> pd.DataFrame:
        pass

class HDBSCAN(ABC):

    def __init__(self):
        pass

    def spatial_clustering(df: pd.DataFrame, save_model=True):
        pass

    def evaluate_clusering(df: pd.DataFrame) -> dict:

        pass
    def visualize_cluster_quality(df, metrics):
        pass