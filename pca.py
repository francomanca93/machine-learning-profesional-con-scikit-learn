import pandas as pd
from scipy.sparse.construct import random
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df_heart = pd.read_csv('./datasets/heart.csv')
    
    print(df_heart.head(5))

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
    
    print(X_train.shape)
    print(y_train.shape)
    
    # Configuracion de la regresión logística
    logistic = LogisticRegression(solver='lbfgs')

        
    # PCA
    # Llamamos y configuramos nuestro algoritmo PCA
    # El número de componentes es opcional
    # Si no le pasamos el número de componentes lo asignará de esta forma:
    # a: n_components = min(n_muestras, n_features)
    pca = PCA(n_components=3)
    # Entrenando algoritmo de PCA
    pca.fit(X_train)

    # Configuramos los datos de entrenamiento con PCA
    df_train = pca.transform(X_train)
    df_test = pca.transform(X_test)
    # Entrenamos la regresion logistica con datos del PCA
    logistic.fit(df_train, y_train)
    # Calculamos nuestra exactitud de nuestra predicción
    print('Score/Accuracy PCA: ', logistic.score(df_test, y_test))
    
    # IPCA
    # Haremos una comparación con incremental PCA, haremos lo mismo para el IPCA.
    # El parámetro batch se usa para crear pequeños bloques,
    # de esta forma podemos ir entrenandolos poco a poco y combinarlos en el resultado final
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    # Entrenando algoritmo de IPCA
    ipca.fit(X_train)

    # Configuramos los datos de entrenamiento con IPCA
    df_train = ipca.transform(X_train)
    df_test = ipca.transform(X_test)
    # Entrenamos la regresion logistica con datos del IPCA
    logistic.fit(df_train, y_train)
    print('Score/Accuracy IPCA: ', logistic.score(df_test, y_test))
    