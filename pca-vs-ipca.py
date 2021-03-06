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

    pca_data = {'accuracy': [],
                'n_components': []}
    ipca_data = {'accuracy': [],
                'n_components': []}
    # PCA
    for n in range(2, 10):
        pca = PCA(n_components=n)
        pca.fit(X_train)
        df_train = pca.transform(X_train)
        df_test = pca.transform(X_test)
        logistic.fit(df_train, y_train)
        acccuracy = logistic.score(df_test, y_test)
        
        pca_data['accuracy'].append(acccuracy)
        pca_data['n_components'].append(n)
    
    # IPC
    for n in range(2, 10):
        ipca = IncrementalPCA(n_components=n, batch_size=10)
        ipca.fit(X_train)
        df_train = ipca.transform(X_train)
        df_test = ipca.transform(X_test)
        logistic.fit(df_train, y_train)
        acccuracy = logistic.score(df_test, y_test)
        
        ipca_data['accuracy'].append(acccuracy)
        ipca_data['n_components'].append(n)
    
    
    plt.plot(pca_data['n_components'], pca_data['accuracy'], label='PCA')
    plt.plot(ipca_data['n_components'], ipca_data['accuracy'], label='IPCA')
    plt.title('N Components vs Accuracy - PCA vs IPCA')
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy of Logistic-Regression')
    plt.legend()
    plt.show()
