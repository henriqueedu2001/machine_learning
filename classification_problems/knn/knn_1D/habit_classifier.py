import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class HabitClassifier:    
    def __init__(self, train_dataset, test_dataset):
        """_summary_

        Args:
            train_dataset (_type_): _description_
            test_dataset (_type_): _description_
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
    def get_dataset_info(self):
        print(self.train_dataset.shape)
        print(self.test_dataset.shape)
                
        
class DatasetHandler:
    """_summary_
    """
    
    def get_dataframe(relative_path: str) -> pd.DataFrame:
        """Fornece o dataframe do pandas, ao se informar o caminho do dataset .csv

        Args:
            relative_path (str): nome do arquivo .csv; informar 'nome_do_aquivo.csv'

        Returns:
            pd.DataFrame: dataframe do pandas
        """
        df_path = DatasetHandler.get_path(relative_path)
        df = pd.read_csv(df_path)
        
        return df
    
    
    def train_test_split(dataframe: pd.DataFrame, train_frac: float) -> (pd.DataFrame, pd.DataFrame):
        """Divide o dataset original em um dataset de treino e outro de teste

        Args:
            dataframe (pd.DataFrame): dataset original
            train_frac (float): fração do dataset original que será incluída no dataset de treino

        Returns:
            (pd.DataFrame, pd.DataFrame): tupla com dataset de treino e dataset de teste, nessa ordem
        """
        train_dataset = dataframe.sample(frac=train_frac)
        test_dataset = dataframe.drop(train_dataset.index)
        
        return train_dataset, test_dataset
        
        
    def get_path(relative_path: str) -> os.path:
        """Obtém o caminho do dataset .csv, a partir do caminho relativo

        Args:
            relative_path (str): caminho relativo do arquivo .csv do dataset

        Returns:
            (os.path): caminho completo até o dataset
        """
        absolute_path = os.path.dirname(__file__)
        full_path = os.path.join(absolute_path, relative_path)
        
        return full_path
    
def test():
    df = DatasetHandler.get_dataframe('passaros_noturnos.csv')
    train_df, test_df = DatasetHandler.train_test_split(df, 0.7)
    
    habit_classifier = HabitClassifier(train_df, test_df)
    habit_classifier.get_dataset_info()
    
    
test()