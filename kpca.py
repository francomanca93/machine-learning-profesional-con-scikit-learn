import pandas as pd
from scipy.sparse.construct import random
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df_heart = pd.read_csv('./datasets/heart.csv')
    
    # Guardamos nuestro dataset sin la columna de target
    df_features = df_heart.drop(['target'], axis=1)
    # Este será nuestro dataset, pero sin la columna
    df_target = df_heart['target']
    
    # Normalizamos los datos
    df_features = StandardScaler().fit_transform(df_features)
    
    # Partimos el conjunto de entrenamiento.
    # Para añadir replicabilidad usamos el random state
    X_train, X_test, y_train, y_test = train_test_split(df_features,
                                                        df_target,
                                                        test_size=0.3,
                                                        random_state=42)
    # Configuracion de la regresión logística
    logistic = LogisticRegression(solver='lbfgs')

    # KPCA
    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(X_train)
    
    # Configuramos los datos de entrenamiento con PCA
    df_train = kpca.transform(X_train)
    df_test = kpca.transform(X_test)

    # Entrenando algoritmo de KPCA
    # Entrenamos la regresion logistica con datos del PCA
    logistic.fit(df_train, y_train)
    # Calculamos nuestra exactitud de nuestra predicción
    print('Score/Accuracy KPCA: ', logistic.score(df_test, y_test))