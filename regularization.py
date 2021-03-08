import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = pd.read_csv('./datasets/whr2017.csv')
    print(dataset.describe())

    # Vamos a elegir los features que vamos a usar
    X = dataset[['gdp', 'family', 'lifexp', 'freedom',
                 'corruption', 'generosity', 'dystopia']]
    
    # Definimos nuestro objetivo, que sera nuestro data set, pero solo en la columna score 
    y = dataset[['score']]
    
    print(X.shape)
    print(y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.25)

    # Aquí definimos nuestros regresores uno por 1 y llamamos el fit o ajuste 
    modelLinear = LinearRegression().fit(X_train, y_train)
    # Vamos calcular la prediccion que nos arroja con la funcion predict con la regresion lineal 
    # y le vamos a mandar el test 
    y_predict_linear = modelLinear.predict(X_test)

    # Alpha, que es valor labda y entre mas valor tenga alpha mas penalizacion 

    # Lasso
    # Seleccionamos, configuramos y entrenamos Lasso
    modelLasso = Lasso(alpha=0.2).fit(X_train, y_train)
    # Hacemos una prediccion para ver si es mejor o peor de lo que teniamos en el modelo lineal sobre
    # exactamente los mismos datos que teníamos anteriormente 
    y_predict_lasso = modelLasso.predict(X_test)

    # Ridge
    # Seleccionamos, configuramos y entrenamos Ridge
    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    # Calculamos el valor predicho para nuestra regresión ridge 
    y_predict_ridge = modelRidge.predict(X_test)

    # Loss Function - Funciones de perdida para cada modelo

    # Calculamos la perdida para cada uno de los modelos que entrenamos, empezaremos con nuestro modelo 
    # lineal, con el error medio cuadratico y lo vamos a aplicar con los datos de prueba con la prediccion 
    # que hicimos 
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    # Mostramos la perdida lineal con la variable que acabamos de calcular
    print("Linear loss: %.10f" % float(linear_loss))
    
    # Mostramos nuestra perdida Lasso, con la variable lasso loss 
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso Loss: %.10f" % float(lasso_loss))

    # Mostramos nuestra perdida de Ridge con la variable ridge loss 
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge loss: %.10f" % float(ridge_loss))
    
    print('='*32)
    print('Coef LASSO')
    print(modelLasso.coef_)
    
    print('='*32)
    print('Coef RIDGE')
    print(modelRidge.coef_)
