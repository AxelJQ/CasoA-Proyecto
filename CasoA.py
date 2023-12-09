import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class Perceptron:
    def __init__(self):

        '''definir la variable para cargar el dataset'''
        self.df = pd.read_csv('diabetes.csv')
        self.x = self.df.iloc[:,0:8]
        self.y = self.df.iloc[:,8]

        '''definir el model de red neuronal'''
        self.model = 0

        '''metricas accuracy'''
        self.accuracy = 0

        '''hacer predicciones'''
        self.prediccion = 0

    def visualizacion_datos(self):

        print(self.df.head())

        print('\n\nImprimiendo datos de entrada')
        print(self.x)

        print('\n\nImprimiendo datos de salida')
        print(self.y)

    def perceptron(self):

        self.model = Sequential()
        self.model.add(Dense(12, input_dim = 8, activation = 'relu'))
        self.model.add(Dense(8, activation = 'relu'))
        self.model.add(Dense(1, activation = 'sigmoid'))

    def compilarmodelo(self):
        
        self.model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    def ajustarfit(self):

        self.model.fit(self.x, self.y, epochs = 100, batch_size = 16)

    def evaluarModel(self):

        self.accuracy = self.model.evaluate(self.x, self.y)
        print(f'Imprimiendo el accuracy {self.accuracy * 100}')

    def predicciones(self):

        self.prediccion = self.model.predict(self.x)

        for i in range(20):

            if self.prediccion[i] >= 0.5:
                print(f'El valor de la prediccion es {1} y el valor esperado es {self.y[i]}')
            else:
                print(f'El valor de la prediccion es {0} y el valor esperado es {self.y[i]}')

p = Perceptron()
p.visualizacion_datos()
p.perceptron()
p.compilarmodelo()
p.ajustarfit()
p.evaluarModel()
p.predicciones()