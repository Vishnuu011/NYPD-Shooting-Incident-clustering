from abc import ABC, abstractmethod
import pandas as  pd

class LoadCsvDataForPreprocess(ABC):
    
    def __init__(self, file_path: str):
        pass

    @abstractmethod
    def load_and_preprocess(self) -> pd.DataFrame:
        pass
