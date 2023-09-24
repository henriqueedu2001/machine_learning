import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class HabitClassifier:
    """Classificador binário de hábitos noturnos dos pássaros
    """
    
    def __init__(self, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, k: int):
        """Construtor do classificador de habitos noturnos

        Args:
            train_dataset (_type_): dataset de treino
            test_dataset (_type_): dataset de teste
            k (int): parâmetro k do algoritmo knn
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.k = k
    
    
    def predict(self, bird_activity_time: float) -> str:
        """prediz a classe a que pertence um pássaro, a partir de seu horário de pico de atividade

        Args:
            bird_activity_time (float): horário de pico de atividade do pássaro

        Returns:
            str: hábito do pássaro (classes 'diurne' e 'nocturne')
        """
        classification_series = []
        
        for i in self.train_dataset.index:
            # horário de pico de atividade e hábito de cada pássaro do dataset de treino
            sample_activity_time = self.train_dataset['activity_time'][i]
            sample_habit = self.train_dataset['habit'][i]
            
            # distância temporal
            habit_time_dist = self.time_distance(bird_activity_time, sample_activity_time)
            
            new_classification_row = {
                'habit_time_distance': habit_time_dist,
                'sample_habit': sample_habit
            }
            
            classification_series.append(new_classification_row)
        
        # procura pelas k pássaros com hábito mais próximo do pássaro de estudo 
        classification_df = pd.DataFrame(classification_series)
        k_nearest = classification_df.nsmallest(self.k, 'habit_time_distance')
        
        # hábito mais frequente, dentre as k instâncias encontradas
        most_frequent_habit = k_nearest['sample_habit'].mode()[0]
        
        return most_frequent_habit


    def time_distance(self, time_a: float, time_b: float) -> float:
        """Calcula a distância entre os instantes de tempo, com a métrica dist(a, b) = |a - b|

        Args:
            time_a (float): instante de tempo time_a
            time_b (float): instante de tempo time_b

        Returns:
            float: distância temporal entre os instantes de tempo time_a e time_b
        """
        return np.absolute(time_a - time_b)
    
    
    def get_test_metrics(self):
        """obtém métricas de desempenho do modelo

        Returns:
            Dict: dicionário com os dados: 
                true_positives
                true_negatives
                false_positives
                false_negatives
                precision
                recall
                accuracy
                true_positive_rate
                true_negative_rate
        """
        # verdadeiro positivo, verdadeiro negativo, falso positivo e falso negativo (quantidades)
        TP, TN, FP, FN = 0, 0, 0, 0
        
        for i in self.test_dataset.index:
            # horário de pico de atividade e hábito de cada pássaro do dataset de teste
            sample_activity_time = self.test_dataset['activity_time'][i]
            sample_habit = self.test_dataset['habit'][i]
            
            # previsão do classificador
            predicted_habit = self.predict(sample_activity_time)
            
            if sample_habit == 'nocturne' and predicted_habit == 'nocturne':
                TP = TP + 1
            elif sample_habit == 'diurne' and predicted_habit == 'diurne':
                TN = TN + 1
            elif sample_habit == 'diurne' and predicted_habit == 'nocturne':
                FP = FP + 1
            elif sample_habit == 'nocturne' and predicted_habit == 'diurne':
                FN = FN + 1
                
        return {
            'true_positives': TP,
            'true_negatives': TN,
            'false_positives': FP,
            'false_negatives': FN,
            'precision': TP/(TP + FP),
            'recall': TP/(TP + FN),
            'accuracy': (TP + TN)/(TP + TN + FP + FN),
            'true_positive_rate': TP/(TP + FP),
            'true_negative_rate': TN/(TN + FN),
        }
                
        
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
    performance_metrics = habit_classifier.get_test_metrics()
    print(performance_metrics)
    
test()