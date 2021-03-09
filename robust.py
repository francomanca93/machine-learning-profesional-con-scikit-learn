from math import gamma
import pandas as pd
from scipy.sparse.construct import random

from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor
)
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

# Para eliminar algunos warnings. Si los queremos ver comentamos estas lineas.
import warnings
warnings.simplefilter("ignore")

if __name__ == "__main__":
    dataset = pd.read_csv('./datasets/felicidad_corrupt.csv')
    
    X = dataset.drop(['country', 'score'], axis=1)
    y = dataset[['score']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)
    
    estimadores = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(), # Meta estimador
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    for name, estimador in estimadores.items():
        # Entrenamos
        estimador.fit(X_train, y_train)
        # Predecimos
        predictions = estimador.predict(X_test)
        # Medimos
        print('='*64)
        print(name)
        meanSquaredError = mean_squared_error(y_test, predictions)
        print('MSE: ', meanSquaredError)

        # Graficamos
        plt.ylabel('Predicted Score')
        plt.xlabel('Real Score')
        plt.title(f'Predicted VS Real - {name}')
        plt.scatter(y_test, predictions, label=f'{name}. MSE: {meanSquaredError}')
        plt.plot(predictions, predictions, 'r--')
        plt.legend()
        plt.show()