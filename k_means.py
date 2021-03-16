import pandas as pd
from seaborn import palettes

# MiniBatchMeans es casi igual al algoritmo de K_Means pero consume menos recursos.
from sklearn.cluster import MiniBatchKMeans

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    
    dataset = pd.read_csv('./datasets/candy.csv')
    
    # print(dataset.head())
    
    # al ser aprendizaje no supervisado, no separamos nuestro datasets en partes 
    # eliminaremos los nombres de los caramelos, ya que eso no nos sirve para el algoritmo
    X = dataset.drop('competitorname', axis=1)
    
    kmeans = MiniBatchKMeans(n_clusters=4,
                             batch_size=8).fit(X)
    print('Total de centros: ', len(kmeans.cluster_centers_))
    print('*'*64)
    print(kmeans.predict(X))
    
    dataset['group'] = kmeans.predict(X)
    
    print(dataset)
    
    sns.pairplot(dataset[['sugarpercent','pricepercent','winpercent', 'group']],
                 hue = 'group',
                 palette='colorblind')

    plt.show()