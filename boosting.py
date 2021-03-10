import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

if __name__== "__main__":
    df_heart = pd.read_csv('./datasets/heart.csv')
    print(df_heart['target'].describe())
    
    X = df_heart.drop(['target'], axis=1)
    y = df_heart['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3)
    
    # Con cross validation podemos optimizar la cantidad de estimadores que deberiamos utilizar
    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
    boost_pred = boost.predict(X_test)
    print('='*64)
    print('GradientBoostingClassifier accuracy :', accuracy_score(boost_pred, y_test))

    # Graficando
    estimators = range(10, 200, 10)
    total_accuracy = []
    for i in estimators:
        boost = GradientBoostingClassifier(n_estimators=i).fit(X_train, y_train)
        boost_pred = boost.predict(X_test)

        total_accuracy.append(accuracy_score(y_test, boost_pred))
    
    plt.plot(estimators, total_accuracy)
    plt.xlabel('Estimators')
    plt.ylabel('Accuracy')
    plt.title('GradientBoostingClassifier accuracies in function of estimators')
    plt.show()

    print(np.array(total_accuracy).max())