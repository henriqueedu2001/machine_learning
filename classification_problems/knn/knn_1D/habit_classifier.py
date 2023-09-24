import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class HabitClassifier:
    """Classificador binário de hábitos noturnos dos pássaros
    """
    
    def __init__(self, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, k: int):
        """Classe 

        Args:
            train_dataset (_type_): _description_
            test_dataset (_type_): _description_
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.k = k
    
    
    def predict(self, bird_activity_time):
        classification_series = []
        
        for i in self.train_dataset.index:
            sample_activity_time = self.train_dataset['activity_time'][i]
            sample_habit = self.train_dataset['habit'][i]
            
            habit_time_dist = self.time_distance(bird_activity_time, sample_activity_time)
            
            new_classification_row = {
                'habit_time_distance': habit_time_dist,
                'sample_habit': sample_habit
            }
            
            classification_series.append(new_classification_row)
        
        classification_df = pd.DataFrame(classification_series)
        k_nearest = classification_df.nsmallest(self.k, 'habit_time_distance')
        most_frequent_habit = k_nearest['sample_habit'].mode()[0]
        
        return most_frequent_habit
        
    def get_k_nearest_samples(self, classification_dataframe: pd.DataFrame, column_name: str):
        return classification_dataframe.nsmallest(self.k, column_name)

    def get_predominant_habit(self, classification_dataframe: pd.DataFrame):
        classification_dataframe

    def time_distance(self, time_a: float, time_b: float):
        return np.absolute(time_a - time_b)
    
    
    def get_dataset_info(self):
        print(self.train_dataset.iloc[:]['activity_time'])
                
        
class DatasetHandler:
    """Ferramenta útil para lidar com datasets
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
    k_parameter = 5
    
    habit_classifier = HabitClassifier(train_df, test_df, k_parameter)
    h = habit_classifier.predict(14.0)
    print(h)
    
test()