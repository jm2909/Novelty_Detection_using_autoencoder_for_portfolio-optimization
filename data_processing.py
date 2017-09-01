import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler,StandardScaler
class Datareading():
    def __init__(self,dataframe):
        self.df = dataframe
    def dataprocessed(self):
        dff = pd.read_csv(self.df)
        Independent_data = dff.drop(["Rating"], 1)
        Response = dff["Rating"]
        return Independent_data,Response

def __Processing__(df,process = 'min-max'):
    if process == 'min-max':
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(df)
    else:
        process == 'standard'
        scaler = StandardScaler()
        dataset = scaler.fit_transform(df)
    return scaler,dataset