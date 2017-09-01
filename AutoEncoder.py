
import numpy as np
import pandas as pd
from _ExpectationMaximization_ import SingleseriesGaussianMixture
from keras.models import Sequential,Model
from keras.layers import Dense
import data_processing

dataread = data_processing.Datareading(dataframe = 'dataset.csv')
Independent_data,Response = dataread.dataprocessed()

scaler,Xind  = data_processing.__Processing__(df = Independent_data,process='min-max')
batch_size = 17
nb_epochs  =10
hidden_dim1 = 10
hidden_dim2 = 5

model = Sequential()
model.add(Dense(hidden_dim1,input_dim = Xind.shape[1],activation = 'relu',name= 'Encoder1'))
model.add(Dense(hidden_dim2,activation = 'relu',name= 'Encoder2'))
model.add(Dense(Xind.shape[1], activation= 'sigmoid',name = 'Decoder'))
model.compile(loss='mean_squared_error', optimizer='adadelta')
model.fit(Xind,Xind,epochs=10,batch_size=17)


predict = model.predict(Xind, batch_size=17)
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('Encoder1').output)
OUTput = intermediate_layer_model.predict(Xind)

n_cluster = 8
Anomaly = predict - Xind
em = SingleseriesGaussianMixture(np.mean(Anomaly,1),n_cluster)
mu_array, sigma_array, sum_array, count, sumsq,labels = em.__EM__()
rate= []
c= []
for  i in range(0,n_cluster):

    if len(Response.loc[labels.icol(0) == i].astype(object).value_counts().index) == 1 and (Response.loc[labels.icol(0) == i].astype(object).value_counts().index[0] == 0):
        zero = Response.loc[labels.icol(0) == i].astype(object).value_counts()[0]
        one =1
    elif len(Response.loc[labels.icol(0) == i].astype(object).value_counts().index) == 1 and (Response.loc[labels.icol(0) == i].astype(object).value_counts().index[0] == 1):
        one = Response.loc[labels.icol(0) == i].astype(object).value_counts()[0]
        zero =1
    else:
        zero = Response.loc[labels.icol(0) == i].astype(object).value_counts()[0]
        one = Response.loc[labels.icol(0) == i].astype(object).value_counts()[1]
    ratio = zero/one
    rate.append(ratio)
    c.append(zero+one)



Report_Reconstruction_Error=pd.DataFrame(np.transpose(np.vstack((mu_array,sigma_array,np.array(rate),np.array(c)))),columns  = ['Mean','Sigma','Odds','Count'])
Report_Reconstruction_Error =Report_Reconstruction_Error.sort(['Mean'])

print(Report_Reconstruction_Error)