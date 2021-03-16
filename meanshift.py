import pandas as pd

from sklearn.cluster import MeanShift

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    
    dataset = pd.read_csv('./datasets/candy.csv')
    
    print(dataset.head())
    
    X = dataset.drop('competitorname', axis=1)
    
    meanshift = MeanShift(bandwidth=None).fit(X)
    
    print(meanshift.labels_)
    
    # Aquí el algoritmo nos devolvio 3 clusters, porque le pareció que esa era la cantidad 
    # correcta teniendo en cuenta como se distrubuye la densidad de nuestros datos 
    print(max(meanshift.labels_))
    print('='*64)
    
    # Imprimamos la ubicación de los centros que puso sobre nuestros datos.
    # Hay que recordar que estos algoritmos crean un centro y
    # a partir de ahí se ajuztan a todos los datos que lo rodean  
    print(meanshift.cluster_centers_)
    
    dataset['meanshift'] = meanshift.labels_
    
    print('='*64)
    print(dataset.head())
    
    sns.pairplot(dataset[['sugarpercent','pricepercent','winpercent', 'meanshift']],
             hue = 'meanshift',
             palette='colorblind')

    plt.show()