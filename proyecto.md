<div align="center">
    <h1>Curso Profesional de Machine Learning con Scikit-Learn</h1>
    <img src="https://imgur.com/cE2aEMF.png" width="">
</div>

## Tabla de contenidos
- [1. Aprender los conceptos clave](#1-aprender-los-conceptos-clave)
  - [Visión general del curso](#visión-general-del-curso)
  - [¿Cómo aprenden las máquinas?](#cómo-aprenden-las-máquinas)
    - [Alternativas al Machine Learning dentro de la Inteligencia Artificial](#alternativas-al-machine-learning-dentro-de-la-inteligencia-artificial)
  - [Problemas que podemos resolver con Scikit-learn](#problemas-que-podemos-resolver-con-scikit-learn)
  - [Las matemáticas que vamos a necesitar](#las-matemáticas-que-vamos-a-necesitar)
- [2. Iniciar un proyecto con sklearn](#2-iniciar-un-proyecto-con-sklearn)
  - [Configuración de nuestro entorno Python](#configuración-de-nuestro-entorno-python)
  - [Instalación de librerías en Python](#instalación-de-librerías-en-python)
  - [Datasets que usaremos en el curso](#datasets-que-usaremos-en-el-curso)
- [3. Optimización de features](#3-optimización-de-features)
  - [¿Cómo afectan nuestros features a los modelos de Machine Learning?](#cómo-afectan-nuestros-features-a-los-modelos-de-machine-learning)
  - [Introducción al algoritmo PCA (Principal Component Analysis)](#introducción-al-algoritmo-pca-principal-component-analysis)
  - [Preparación de datos para PCA e IPCA](#preparación-de-datos-para-pca-e-ipca)
  - [Implementación del algoritmo PCA e IPCA](#implementación-del-algoritmo-pca-e-ipca)
  - [Kernels y KPCA](#kernels-y-kpca)
  - [Regularización](#regularización)
    - [Tipos de regularización](#tipos-de-regularización)
  - [Implementación de Lasso y Ridge](#implementación-de-lasso-y-ridge)
  - [Explicación resultado de la implementación](#explicación-resultado-de-la-implementación)
- [4. Regresiones robustas](#4-regresiones-robustas)
  - [Valores atípicos](#valores-atípicos)
    - [¿Por qué son problemáticos?](#por-qué-son-problemáticos)
    - [¿Cómo identificarlos?](#cómo-identificarlos)
      - [Analíticamente, con métodos estadísticos](#analíticamente-con-métodos-estadísticos)
      - [Graficamente](#graficamente)
  - [Regresiones Robustas en Scikit-learn](#regresiones-robustas-en-scikit-learn)
    - [Tipos | Más utilizadas](#tipos--más-utilizadas)
  - [Preparación de datos para la regresión robusta](#preparación-de-datos-para-la-regresión-robusta)
  - [Implementación regresión robusta](#implementación-regresión-robusta)
- [5. Métodos de ensamble aplicados a clasificación](#5-métodos-de-ensamble-aplicados-a-clasificación)
  - [¿Qué son los métodos de ensamble?](#qué-son-los-métodos-de-ensamble)
    - [Bagging](#bagging)
    - [Boosting](#boosting)
  - [Preparación de datos para implementar métodos de ensamble](#preparación-de-datos-para-implementar-métodos-de-ensamble)
  - [Implementación de Bagging](#implementación-de-bagging)
  - [Implementación de Boosting](#implementación-de-boosting)
- [6. Clustering](#6-clustering)
  - [Estrategias de Clustering](#estrategias-de-clustering)
    - [Casos de aplicación de clustering](#casos-de-aplicación-de-clustering)
  - [Implementación de Batch K-Means](#implementación-de-batch-k-means)
  - [Implementactión de Mean-Shift](#implementactión-de-mean-shift)
- [7. Optimización paramétrica](#7-optimización-paramétrica)
  - [Validación de nuestros modelos. Hold Out y Cross Validation](#validación-de-nuestros-modelos-hold-out-y-cross-validation)
    - [Tipos de validación](#tipos-de-validación)
      - [Hold-Out](#hold-out)
      - [K-Folds](#k-folds)
      - [LOOCV](#loocv)
  - [Implementación de K-Folds Cross Validation](#implementación-de-k-folds-cross-validation)
  - [Optimización de hiperparametros | Hyperparameter Optimization](#optimización-de-hiperparametros--hyperparameter-optimization)
    - [Optimización manual](#optimización-manual)
    - [Optimizacion por grilla de parámetros | GridSearchCV](#optimizacion-por-grilla-de-parámetros--gridsearchcv)
    - [Optimizacion por búsqueda aleatorizada | RandomizedSearchCV](#optimizacion-por-búsqueda-aleatorizada--randomizedsearchcv)
    - [GridSearchCV vs RandomizedSearchCV](#gridsearchcv-vs-randomizedsearchcv)
  - [Implementación de Randomized](#implementación-de-randomized)
  - [BONUS: Auto Machine Learning](#bonus-auto-machine-learning)
    - [auto-sklearn](#auto-sklearn)
- [8. Salida a producción](#8-salida-a-producción)
  - [Revisión de nuestra arquitectura de código](#revisión-de-nuestra-arquitectura-de-código)
  - [Importar y exportar modelos con Sklearn](#importar-y-exportar-modelos-con-sklearn)
  - [Creación de una API con Flask para el modelo](#creación-de-una-api-con-flask-para-el-modelo)
- [Conclusiones](#conclusiones)
  - [Manejo de features](#manejo-de-features)
  - [Algoritmos de ML](#algoritmos-de-ml)
  - [Validacion y optimizacion de hiperparametros](#validacion-y-optimizacion-de-hiperparametros)
  - [Como exponer un modelo en produccion](#como-exponer-un-modelo-en-produccion)

# 1. Aprender los conceptos clave

## Visión general del curso

Se hará un proyecto de Implementación de regresión robusta. En el usaremos el dataset del índice de felicidad y aprenderemos a limpiar y transformar datos atípicos para procesarlos por medio de un regresión robusta.

Buscaremos trabajar todo el proceso de Machine Learning de principio a final de una manera profesional. Utilizaremos herramientas para generar procesos orientados a la industria.

La herramienta principal será [Scikit Learn](https://scikit-learn.org/stable/)

¿Por qué usar Scikit-learn?

- Curva de aprendizaje suave.
- Es una biblioteca muy versátil.
- Comunidad de soporte.
- Uso en producción.
- Integración con librerías externas.

Módulos de Scikit-learn

1. Clasificación.
   - Nos ofrece los algoritmos de clasificación mas utilizados en la industria.
2. Regresión.
   - Nos ofrece los algoritmos de regresión mas utilizados en la industria.
3. Clustering.
   - Nos ofrece los algoritmos de clustering mas utilizados en la industria.
4. Preprocesamiento.
   - Podemos tomar los datos en crudo, normalizarlos, extraer feature, transformarlos, hacer encoding, para poder trabajarlo mas facil con los modelos.
5. Reducción de dimensionalidad.
   - Contiene este modulo tomar datos y transformarlos de tal manera que podamos extraer la informacion mas relevante para que una vez lleguen a nuestros modelos, estos trabajen los mejor posible.
6. Selección del modelo.
   - Scikit Learno nos ayuda a seleccionar el mejor modelo para nuestros datos, podemos automatizar el proceso de optimizacion para que podamos garantizar que nuestro proyecto al final tenga el mejor enfoque posible.

Preguntas que nos haremos y resolveremos en el curso:
- ¿Como nos ayuda Scikit Learn en el preprocesamiento de datos?
- ¿Que modelos podemos utilizar para resolver problemas especificos?
- ¿Cual es el procedimiento a seguir para poder optmizar los modelos?

## ¿Cómo aprenden las máquinas?

Las maquinas aprenden de datos. Desde el punto de vista de los datos, podemos aplicar tres técnicas según la naturaleza y disponibilidad de los mismos

- Aprendizaje supervisado (Algoritmos **por observación**):
  - Si de los datos se puede extraer con anticipación información precisa del resultado que esperamos.
- Aprendizaje por refuerzo (Algoritmos por **prueba y error**):
  - Si de los datos no se puede sacar exactamente la información que queremos predecir, pero si podemos dejar que el modelo tome decisiones y evalue si estas decisiones son buenas o malas.
- Aprendizaje no supervisado (Algoritmos **por descubrimiento**):
  - Cuando no se tiene ninguna información adicional sobre lo que esperamos, sino que los datos por sí solos nos van a revelar información relevante sobre su propia naturaleza y estructura.

### Alternativas al Machine Learning dentro de la Inteligencia Artificial

- **Algoritmos evolutivos**
  - Son una serie de algoritmos heuristicos, en donde, en tu espacio de soluciones se explora las mejores candidatos, según se optimice cierta función de costo. Por ejemplo, se usa en la industría automotriz o de diseño aeroespacial para encontrar el mejor diseño que minimice, por ejemplo, la resistencia al aire.

- **Lógica Difusa**
  - Es una generalización de la lógica clásica, pero en lugar de tener solo dos condiciones (verdadero, falso) [principio de tercero excluido], se tienen condiciones de verdad continuas. Por ejemplo, si 1 representa verdadero, y 0 representa falso, en la lógica difusa, el grado de verdad ahora puede tomar valores en el intervalo continuo de [0, 1]. Este enfoque tenía mucho auge en sistemas de control y robotica, antes del auge de las redes neuronales.

- **Agentes y Sistemas expertos**
  - Para sistemas cuyas propiedades se puede describir por la interacción de agentes, se utiliza este enfoque para encontrar o describir comportamientos en el colectivo. (Por ejemplo, los mercados financieros compuestos por agentes economicos, vendedores y compradores, etc). La física estadística y los sistemas complejos también se ayudan de este enfoque.

## Problemas que podemos resolver con Scikit-learn

Algunas limitaciones de Scikit-learn

1. No es una herramienta de Computer Vision.
   - Se necesita complementar con una herramienta adicional tales como [OpenCV](https://opencv.org/) o [TorchVision](https://pytorch.org/docs/stable/torchvision/index.html) que forma parte del proyecto de Pytorch.
2. No se puede correr en GPUs.
3. No es una herramienta de estadística avanzada.
4. No es muy flexible en temas de Deep Learning.

Qué problemas podemos abordar con Scikit-learn?

- **Clasificaciones**: Necesitamos etiquetar nuestros datos para que encajen en alguna de ciertas categorías previamente definidas.
  - Ejemplos:
    - ¿Es cáncer o no es cáncer?
    - ¿La imagen pertenece a un Ave, Perro o Gato?
    - ¿A qué segmento de clientes pertenece determinado usuario?

- **Regresión**: Cuando necesitamos modelar el comportamiento de una variable continua, dadas otras variables correlaciones
  - Ejemplos:
    - Predecir el precio del dólar para el mes siguiente.
    - El total de calorías de una comida dados sus ingredientes.
    - La ubicación más probable de determinado objeto dentro de una imagen.

- **Clustering**: Queremos descubrir subconjuntos de datos similares dentro del dataset. Queremos encontrar valores que se salen del comportamiento global.
  - Ejemplo:
    - Identificar productos similares para un sistema de recomendación.
    - Descubrir el sitio ideal para ubicar paradas de buses según la densidad poblacional.
    - Segmentar imágenes según patrones de colores y geometrías.

## Las matemáticas que vamos a necesitar

La cortina de fondo: Varias técnicas que usamos para que los computadores aprendan están inspiradas en el mundo natural.

- Redes neuronales artificiales: Están inspiradas en el cerebro humano.
- Aprendizaje por refuerzo: Está inspirado en las teorías de la psicología conductual.
- Algoritmos evolutivos: Los teorías de Charles Darwin.

Temas matemáticos generales a repasar:

- Funciones y trigonométrica.
- Algebra lineal. [Repositorio complemento al curso -- Álgebra Lineal Aplicada para Machine Learning](https://github.com/francomanca93/algebra-aplicada-python)
- Optimización de funciones.
- Calculo diferencial.

Temas de probabilidad y estadistica a repasar:

- Probabilidad básica.
- Combinaciones y permutaciones.
- Variables aleatorias y distribuciones.
- Teorema de Bayes.
- Pruebas estadísticas.

> - La conclusión es que si no tienes buenas bases de matemáticas, es muy difícil tener un “entendimiento real” de machine learning e inteligencia artificial. Serán como “cajas negras”.
>
> - La estrategia del curso será de desarrollo de software y ciencia de la computación.
> - Scikit Learn nos ayudará a cubrir algunos vacios conceptuales de una manera que beneficie a nuestro modelo.

# 2. Iniciar un proyecto con sklearn

## Configuración de nuestro entorno Python

Los entornos virtuales nos permiten isolar multiples dependencias para el desarrollo de proyecto, puede pasar por ejemplo cuando trabajas con diferentes versiones de python o de django.

Python 3 trae la creación y manejo de entornos virtuales como parte del modulo central.

Entorno virtual con Python

Para crear un entorno virtual utilizas:

`python3 -m venv .NOMBRE-ENTORNO`

Nota:.NOMBRE-ENTORNO es el nombre de del ambiente

Para activarlo:

`source -m ./.env/bin/activate`

Si queremos desactivarlo:

`deactivate`

Si deseamos ver las librerías instaladas en el ambiente:

`pip freeze`

## Instalación de librerías en Python

Librerias y sus versiones con las que trabajaremos.

Si las copiamos en un archivo `requirements.txt` y luego con el comando `pip install -r requirements.txt` podremos instalarlas a todas.

```
numpy==1.17.4
scipy==1.3.3
joblib==0.14.0
pandas==0.25.3
matplotlib==3.1.2
scikit-learn==0.22
```

## Datasets que usaremos en el curso

> [Datasets e informacion sobre ellos en el repo](datasets)

- [World Happiness Report](https://www.kaggle.com/unsdsn/world-happiness): Es un dataset que desde el 2012 recolecta variables sobre diferentes países y las relaciona con el nivel de felicidad de sus habitantes.

> **Nota: Este data set lo vamos a utilizar para temas de regresiones**

- [The Ultimate Halloween Candy Power Ranking](https://www.kaggle.com/fivethirtyeight/the-ultimate-halloween-candy-power-ranking): Es un estudio online de 269 mil votos de más de 8371 IPs deferentes. Para 85 tipos de dulces diferentes se evaluaron tanto características del dulce como la opinión y satisfacción para generar comparaciones. 

> **Nota: Este dataset lo vamos a utilizar para temas de clustering**

- [Heart disease prediction](https://www.kaggle.com/c/SAheart): Es un subconjunto de variables de un estudio que realizado en 1988 en diferentes regiones del planeta para predecir el riesgo a sufrir una enfermedad relacionada con el corazón. 

> **Nota: Este data set lo vamos a utilizar para temas de clasificación.**

# 3. Optimización de features

## ¿Cómo afectan nuestros features a los modelos de Machine Learning?

**¿Qué son los features?** Son los atributos de nuestro modelo que usamos para realizar una interferencia o predicción. Son las variables de entrada.

Más features simpre es mejor, ¿verdad?
**La respuesta corta es: NO**

En realidad si tenemos variables que son irrelevantes pasarán estas cosas:

- Se le abrirá el paso al ruido.
- Aumentará el costo computacional.
- Si introducimos demasiados features y estos tienen valores faltantes, se harán sesgos muy significativos y vamos a perder esa capacidad de predicción.

> Nota: Hacer una buena selección de nuestro features, hará que nuestros algoritmos corran de una manera mas eficiente.

Una de las formas de saber que nuestros features han sido bien seleccionados es con el **sesgo (bias)** y la **varianza**.

- Una mala selección de nuestro features nos puede llevar a alguno de esos dos escenarios indeseados.

![varianzas-vs-sesgo](https://imgur.com/hyJhWFL.png)

Algo que debemos que recordar es que nuestro modelo de ML puede caer en uno de 2 escenarios que debemos evitar:

![over-under-balan](https://imgur.com/rWib1tG.png)

- **Underfitting**: Significa que nuestro modelo es demasiado simple, en donde nuestro modelo no está captando los features y nuestra variable de salida, por lo cual debemos de investigar variables con mas significado o combinaciones o transformaciones para poder llegar a nuestra variable de salida.

- **Overfitting**: Significa que nuestro modelo es demasiado complejo y nuestro algoritmo va a intentar ajustarse a los datos que tenemos, pero no se va a comportar bien con los datos del mundo real. Si tenemos overfiting lo mejor es intentar seleccionar los features de una manera mas critica descartando aquellos que no aporten información o combinando algunos quedándonos con la información que verdaderamente importa.

> - Cuando tenemos un sesgo (bias) alto lo que se hace es añadir mas features, aumentar el numero de datos no ayudara mucho.
> - Cuando tenemos un varianza (variance) alto lo que se hace es aumentar el numero de datos para que nuestro modelo generalice mejor.

**¿Qué podemos hacer para solucionar estos problemas?**

- Aplicar técnicas reducción de la dimensionalidad. Utilizaremos el algoritmo de PCA.
- Aplicar la técnica de la regulación, que consiste en penalizar aquellos features que no le estén aportando o que le estén restando información a nuestro modelo.
- Balanceo: Se utilizará Oversampling y Undersampling en problemas de rendimiento donde tengamos un conjunto de datos que está desbalanceado, por ejemplo en un problema de clasificación donde tenemos muchos ejemplos de una categoría y muy pocos de otra.

## Introducción al algoritmo PCA (Principal Component Analysis)

[Analisis de componentes principales (PCA) - Explicancion intruitiva en youtube](https://www.youtube.com/watch?v=AniiwysJ-2Y&t)

**¿Por qué usaríamos este algoritmo?**

- Porque en machine learning es normal encontrarnos con problemas donde tengamos una enorme cantidad de features en donde hay relaciones complejas entre ellos y con la variable que queremos predecir.

**¿Donde se puede utilizar un algoritmo PCA?**

- Nuestro dataset tiene un número alto de features y no todos sean significativos.
- Hay una alta correlación entre los features.
- Cuando hay overfiting.
- Cuando implica un alto coste computacional.

**¿En que consiste el algoritmo PCA?**

Básicamente en reducir la complejidad del problema:

1. Seleccionando solamente las variables relevantes.
2. Combinándolas en nuevas variables que mantengan la información más importante (varianza de los features).

![reduccion-dimensionalidad](https://imgur.com/s9Q0diS.png)

**¿Cuales son pasos para llevar a cabo el algoritmo PCA?**

1. Calculamos la matriz de covarianza para expresar las relaciones entre nuestro features.
2. Hallamos los vectores propios y valores propios de esta matriz, para medir la fuerza y variabilidad de estas relaciones.
3. Ordenamos y escogemos los vectores propios con mayor variabilidad, esto es, aportan más información.

**¿Qué hacer si tenemos una PC de bajos recursos?**

- Si tenemos un dataset demasiado exigente, podemos usar una variación como IPCA.
- Si nuestros datos no tienen una estructura separable linealmente, y encontramos un KERNEL que pueda mapearlos podemos usar KPCA.

## Preparación de datos para PCA e IPCA

[Script preparando los datos para PCA e IPCA](pca.py)

- Importamos las librerias a utilizar
- Guardamos los feature y el target
- Normalizamos los datos.

```txt
# La estandarizacion que hace sklearn con StandardScaler es:
z = x-u / s

x = valor
u = media
s = desviacion estandar
```

- Partimos el conjunto de entrenamiento.
  - `X` representa los features. Normalmente es una matriz, por eso la mayúscula.
  - `y` representa el target. Siempre es un vector, nunca una matriz, por eso la minúscula.

- Para añadir replicabilidad usamos el random state
  - `random_state` es para dejar estáticos los valores aleatorios que te genera, de forma que al volverlo a correr siga trabajando con esos valores aleatorios y no te genere nuevos valores aleatorios.

## Implementación del algoritmo PCA e IPCA

Estamos trabajando bajo el dataset de pacientes con riesgo a padecer una enfermedad cardiaca. Con este dataset pretendemos que utilizando ciertas variables de los pacientes, por ejemplo su edad, su sexo, su presión sanguínea y un indice de dolor que pueden sentir al realizar ejercicio físico.

Vamos a intentar hacer una clasificación binaria, entre si el paciente tiene una enfermedad cardiaca o no la tiene, el objetivo es hacer una clasificación básica, pero que nos dé una información relevante, maximizando la información de todos estos features.

En esta sección lo que hicimos fue:

- Configuracion de la regresión logística
- PCA
  1. Llamamos y configuramos nuestro algoritmo PCA. El número de componentes es opcional. Si no le pasamos el número de componentes lo asignará de esta forma: n_components = min(n_muestras, n_features)
  2. Entrenando algoritmo de PCA
  3. Configuramos los datos de entrenamiento con PCA
  4. Entrenamos la regresion logistica con datos del PCA
  5. Calculamos nuestra exactitud de nuestra predicción
- IPCA
  - Haremos una comparación con incremental PCA, haremos lo mismo para el IPCA. El parámetro batch se usa para crear pequeños bloques, de esta forma podemos ir entrenandolos poco a poco y combinarlos en el resultado final
  - Mismos pasos a partir del 2 que para PCA.

[Comparación de PCA vs IPCA](pca-vs-ipca.py)
![pca-vs-ipca](https://imgur.com/3pNvf3I.png)

> Conclusión
>
> El rendimiento de los dos algoritmos es casi exactamente el mismo, pero hay que considerar que nuestro dataset tenia 13 fetures originalmente para intentar predecir una clasificación binaria y utilizando PCA, solo tuvimos que utilizar 3 features artificiales que fueron los que nos   devolvió PCA para llegar a un resultado coste computacional y estamos utilizando información que es realmente relevante para nuestro modelo.

## Kernels y KPCA

[Script aplicando KPCA](kpca.py)

Ya conocemos los algoritmos PCA e IPCA, ¿Que otras alternativas tenemos?

Una alternativa son los Kernels. Un **Kernel** es una función matemática que toma mediciones que se comportan de manera no lineal y las proyecta en un espacio dimensional más grande en donde sen linealmente separables.

**Y, ¿esto para que puede servir?**

Sirve para casos donde los datos no son linealmente separables. El la primera imagen no es posible separarlos con una linea y en la imagen 2 si lo podemos hacer mediante Kernels. Lo que hace la función de Kernels es proyectar los puntos en otra dimensión y así volver los datos linealmente separables.

![kernels](https://imgur.com/T3OOW2u.png)

**¿Que tipo de funciones para Kernels nos podemos encontrar?**

![](https://imgur.com/hGfKF92.png)

**Ejemplos de funciones de Kernels en datasets aplicados a un clasificador:**

![](https://imgur.com/475H5GC.png)

> Conclusión:
>
> El aumento de dimensiones puede ayudar a resolver problemas de clasificación.

## Regularización

La regularización es una técnica que consiste en disminuir la complejidad de nuestro modelo a través de una penalización aplicada a sus variables más irrelevantes.

**¿Como hacemos lo anterior?**

Introducimos un mayor Sesgo sobre las variables y disminuimos la Varianza. Con esto logramos mejorar la generalización de la predicción de nuestro modelo con los datos y lograr evitar o reducir el Overfitting.

![](https://imgur.com/FJyWxar.png)

Podemos apreciar en la gráfica 1, hay un sobre ajuste, ya que la linea roja se acopla muy bien para los datos de prueba, pero no para los datos de entrenamiento. La linea roja en los datos de prueba da una mala generalización, una mala aproximación.

Pero para poder aplicar regularización necesitamos un termino adicional el concepto de **perdida** **(loss)**. El concepto de perdida nos dice que tan lejos están nuestras predicciones de los datos reales, esto quiere decir que entre menor sea la perdida mejor será nuestro modelo.

**Perdida en entrenamiento y en validación**

![](https://imgur.com/vbUJg5r.png)

Podemos ver en la gráfica que la perdida tiende a disminuir, porque en algún momento van a ser vistos, van a ser operados y el modelo va a tender a ajustarse a esos datos de entrenamiento, pero lo que tenemos que mirar es cómo se va a comportar en el mundo real.

En el conjunto de validación o pruebas es muy normal que nuestra perdida comience a disminuir porque hay una buena generalización, pero llega un punto donde nuevos valores comienza a introducirse donde esa perdida vuelve a comenzar a subir ese es el punto donde en general se considera que comienza a haber sobreajuste. Es la perdida la medida que vamos a utilizar para poder aplicar la regularización.

> Conclusiones:
>
> - La regularización aumenta el sesgo y disminuye la varianza con el objetivo de mejorar la generalización del modelo.
> .
> - PCA: Combinábamos variables creando así variables artificiales.
> - Regularización: Se penaliza a las variables que aportan menos información.
> - Ambas buscan disminuir la complejidad del modelo.

### Tipos de regularización

[Ridge vs Lasso Regression | StatQuest with Josh Starmer | Youtube](https://www.youtube.com/watch?v=Xm2C_gTAl8c)

- **L1 Lasso**: Para reducir la complejidad a través de eliminación de features que no aportan demasiado al modelo.
  - Penaliza a los features que aporta poca información volviéndolos cero, eliminado el ruido que producen en el modelo.

![lasso](https://imgur.com/k5g7MPc.png)

![lasso-pic](https://imgur.com/IzzR5mc.png)

- **L2 Ridge**: Reducir la complejidad disminuyendo el impacto de ciertos features a nuestro modelo.
  - Penaliza los features poco relevantes, pero no los vuelve cero. Solamente limita la información que aportan a nuestro modelo.

![ridge](https://imgur.com/WcrxFsJ.png)

![ridge-pic](https://imgur.com/kn0gywb.png)

- **ElasticNet**: Es una combinación de las dos anteriores.

> Conclusión
> Lasso vs Ridge.
>
> 1. No hay un campeón definitivo para todos los problemas.
> 2. Si hay pocos features que se relacionen directamente con la variable a predecir: **Probar Lasso**.
> 3. Si hay varios features relacionados con la variable a predecir: **Probar Ridge**.

## Implementación de Lasso y Ridge

Implementaremos las tecnicas de regularizacion. Para esto utilizaremos dos regresores que vienen por defecto en Scikit Learn y que de una manera automatizada nos integra un modelo lineal con su respectiva regularización.

Utilizaremos el dataset del reporte de la felicidad mundial. Dataset que mide varios factores en diferentes paises, tales como, el indice de corrupción, nivel que nos indica que tan fuerte son las relaciones de familia, el desarrollo per capita económico y nos intenta dar una variable continua para medir la felicidad del pais en cuestión.

[Script aplicando regularizacion](regularization.py)

1. Vamos a elegir los features que vamos a usar
2. Definimos nuestro objetivo, que sera nuestro data set, la columna score.
3. Definimos nuestros regresores, los seleccionamos, configuramos y entrenamos. Vamos calcular la prediccion que nos arroja con la funcion predict y luego le lanzaremos el test.
   1. Linear Regression
   2. Lasso
   3. Ridge

4. Loss Function - Funciones de perdida para cada modelo.
   - Calculamos la perdida para cada uno de los modelos que entrenamos

## Explicación resultado de la implementación

```shell
Linear loss: 0.0000000569
Lasso Loss: 0.5168427944
Ridge loss: 0.0066128808
```

Menor perdida es mejor, esto quiere decir que hubo menos equivocacion entre los valores esperados y los valores predichos.

```shell
================================
Coef LASSO
[1.00763927 0.         0.         0.         0.         0.         0.32076227]
================================
Coef RIDGE
[[1.05791328 0.964398   0.87157588 0.87910868 0.61100802 0.77385103 0.96904502]]
================================
```
Los coeficientes obtenidos en Lasso y Ridge es un arregla que tiene el mismo tamaño que las columnas que utilizamos, nuestros features.

```py
X = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
```

Los numeros mas grandes dentro del arreglo, significa que la columna en si esta teniendo mas peso en el modelo que estamos entrenando.

- **Lasso**
  - Los valores que Lasso haya hecho 0, nos indica que el algoritmo no te dio la atencion necesaria o no los considero importante. Analizar porque hizo eso nuestro algoritmo Lasso ya esta tarea nuestra como Data Scientist.
- **Ridge**
  - En Ridge ninguno de los coeficientes han sido 0, sino que fueron disminuidos, esto se hace precisamente la regresión Ridge

# 4. Regresiones robustas

## Valores atípicos

- Un valor atípico es cualquier medición que se encuentre por fuera del comportamiento general de una muestra de datos.
- Pueden indicar variabilidad, errores de medición o novedades.

### ¿Por qué son problemáticos?

1. Pueden generar sesgos importantes en los modelos de ML.
2. A veces contienen información relevante sobre la naturaleza de los datos.
3. Detección temprana de fallos.

### ¿Cómo identificarlos?

#### Analíticamente, con métodos estadísticos

- **Z - Score**: Mide la distancia (en desviaciones estándar) de un punto dado a la media.
- Técnicas de clustering como **DBSCAN**. [Clusting with Scikit Learn - More Info](https://dashee87.github.io/data%20science/general/Clustering-with-Scikit-with-GIFs/)
  - Consiste en considerar a zonas muy densas como clusters, mientras que los puntos que carecen de **‘vecinos’** no pertenecen a ningún conjunto y por lo tanto se clasifican como ruido (o **outliers**).
  - Una ventaja de está técnica es que no se requiere que se especifique el número de clusters (como en K-means, por ejemplo), en cambio se debe especificar un número mínimo de datos que constituye un cluster y un parámetro epsilon que está relacionado con el espacio entre vecinos.

  ![dbscan](https://imgur.com/cpDAOhS.gif)

- Si `q< Q1-1.5IQR` ó `q > Q3+1.5IQR`. [Articulo en medium con explicación ampliada](https://towardsdatascience.com/why-1-5-in-iqr-method-of-outlier-detection-5d07fdc82097)

#### Graficamente

- Con Boxplot.
  - El grafico de caja de una buena forma para detectar los valores atípicos en un set de datos, a su vez también es aconsejable (dependiendo del caso) eliminarlos para que nuestro análisis sea lo más confiable posible.

    ![boxplot](https://imgur.com/oq5CiJ9.png)

## Regresiones Robustas en Scikit-learn

¿Como podemos lidiar con valores atípicos?

- Se pueden tratar desde la etapa de pre procesamiento intentando eliminarlo y transformarlo de alguna manera.
- Hay veces que la unica menera de lidiar con ellos es cuando estamos aplicando nuestro modelo de Machine Learning.
  - Scikit Learn nos ofrece un meta estimador que nos permite configurar diferentes estimadores para lidiar con los valores atipicos, de una manera facil de implementar. A estas técnicas se las conoce como **Regresiones Robustas**

### Tipos | Más utilizadas

- **RANSAC**:
  - Selecciona una muestra aleatoria de los datos asumiendo que esa muestra se encuentra dentro de los valores inliners, con estos datos se entrena el modelo y se compara su comportamiento con respecto a los otros datos.
  - El procedimiento anterior se repite tantas veces como se indique y al finalizar el algoritmo escoge la combinación de datos que tenga la mejor cantidad de inliners, donde los valores atípicos puedan ser discriminados de forma efectiva.
  - [More info about RANSAC algorithm in Wikipedia](https://es.wikipedia.org/wiki/RANSAC)
  - [sklearn.linear_model.RANSACRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html?highlight=ransac#sklearn.linear_model.RANSACRegressor)
  - [Robust linear model estimation using RANSAC](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html)

![ransac](https://imgur.com/y2GudLP.png)

![ransac2](https://imgur.com/U0gcEja.png)

- **Huber Reggresor**:
  - No elimina los valores atípicos sino que los penaliza.
  - Realiza el entrenamiento y si el error absoluto de la perdida alcanza cierto umbral (epsilon) los datos son tratados como atípicos.
  - El valor por defecto de epsilon es 1.35 ya que se ha demostrado que logra un 95% de eficiencia estadística.
  - [Hube Regressor in Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html).
  - [HuberRegressor vs Ridge on dataset with strong outliers](https://scikit-learn.org/stable/auto_examples/linear_model/plot_huber_vs_ridge.html#sphx-glr-auto-examples-linear-model-plot-huber-vs-ridge-py)

![huber](https://imgur.com/yqw9c3j.png)

## Preparación de datos para la regresión robusta

Lo que haremos es corromper los datos del dataset de felicidad agregando valores atipicos (**outliers**) y poder experimentar y aplicar nuestras regresiones robustas.

Una manera mas profesional de trabajar con modelo y estimadores, al trabajar con mas de uno es utilizar diccionarios. Esto nos permite hacer que nuestro codigo sea mas efectivo, sin repetir tanto código y poder hacer pruebas mas elaboradas. Lo que veniamos haciendo es prepararlos, entrenarlos y medirlos uno por uno, pero Scikit Learn nos permite hacer todo esto de una manera mas efectiva y sencilla.

```py

estimadores = {
    'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
    'RANSAC': RANSACRegressor(), # Meta estimador
    'HUBER': HuberRegressor(epsilon=1.35)
}

```

- **SVM (Suppot Vector Machine)**: Con el parámetro C podemos controlar la penalización por error en la clasificación. Si C tiene valores amplios entonces, se penaliza de forma más estricta los errores, mientras que si escogemos un C pequeño seremos menos estrictos con los errores. En otras palabras, si C es pequeño aumenta el sesgo y disminuye la varianza del modelo.
- **RANSAC**: Al ser un meta estimador, podemos pasarle como parámetros diferentes estimadores, para nuestro caso vamos a trabajar de una forma genenérica.
- **HUBER**: El valor de epsilon es 1.35. Utilizamos este valor ya que se ha demostrado que logra un 95% de eficiencia estadística.

## Implementación regresión robusta

[Script implementando regresiones robustas](robust.py)

Gracias a los diccionarios, loops y scikit learn podemos operar de forma mas ordenada y secuencial.
Con scikit learn, todos los estimadores tienen la misma interfaz de funciones. En general gracias a esto no hace falta implemntar funciones especificadas para cada uno de los algoritmos.

Los que hacemos en nuestro loop a traves del diccionario de los modelos es:

1. Entrenar
2. Predecir
3. Medir
4. Graficar valores

Obtenemos lo siguiente:

![svm](https://imgur.com/eHZAtfw.png)

![ransac](https://imgur.com/GM3KrFo.png)

![huber](https://imgur.com/ujJtzfo.png)

> [Robust Regression for Machine Learning in Python - More Info - Excellent article](https://machinelearningmastery.com/robust-regression-for-machine-learning-in-python/)

# 5. Métodos de ensamble aplicados a clasificación

## ¿Qué son los métodos de ensamble?

Sirven cuando queremos probar diferentes estimadores y llegar a una unica conclusión a través de un consenso.

1. Combinar difentes métodos de ML con diferentes configuraciones y aplicar un método para lograr un consenso.
2. La diversidad es una muy buena opción. Probar diferentes modelos con diferentes parámetros nos permite abrir mas el abanico de soluciones a un mismo problema.
3. Los métodos de ensable se han destacado por ganar competencias en ML.

Hay dos estrategias. Bagging y Boosting

### Bagging

Significa **B**ootstrap **Agg**regation

¿Que tal si en lugar de depender de la opinión de un solo "experto" consultamos la opinión de varios expertos en paralelo e intentamos lograr un consenso?
> Bueno, **asi trabaja Bagging**

![bagging1](https://imgur.com/ZAjuwVt.png)
![bagging2](https://imgur.com/0wpRSAX.png)
![bagging3](https://imgur.com/PDngmWg.png)

Modelos de ensable basados en Bagging

1. Random Forest
2. Volting Classifiers/Regressors
3. Se puede aplicar sobre cualquier familia de modelo de ML.

### Boosting

Boosting significa impulsar/propulsar.

¿Y si probamos otro enfoque? Le pedimos a un expero su creterio sobre un problema. Medimos su posible error, y luego usando ese error calculado le pedimos a otro experto su juicio sobre el  mismo problema.

> Bueno, **asi trabaja Boosting**

![boosting](https://imgur.com/TRW92ol.png)

Modelos de ensable basados en Bagging

1. AdaBoost
2. Gradient Tree Boosting
3. XGBoost

Resumen con imágenes:

![bagg-boost1](https://imgur.com/xaC0Ptk.png)
![bagg-boost2](https://imgur.com/kFyPySh.png)
![bagg-boost3](https://imgur.com/HQKlOcx.png)

> [What is the difference between Bagging and Boosting? by QuantDare](https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/)

## Preparación de datos para implementar métodos de ensamble

Utilizaremos un meta estimador que tiene Scikit Learn llamado **Bagging Classifier**. Al ser un meta estimador podemos adaptarlo a las diferentes familias de estimadores y Scikit Learn lo configurara de forma automática para que se convierta en un método de ensamble.

Utilizaremos el dataset de afecciones cardiacas. Teniamos diferentes datos de pacientes, donde la meta finalmente era clasificarlos en si el paciente tenia o no una afección cardiaca.

## Implementación de Bagging

[Script implementando Bagging](bagging.py)

- Implementaremos el clasificador `KNeighborsClassifier()` y veremos su accuracy.
- Tambien implementaremos Bagging con una serie de algoritmos de clasificación en un diccionario y veremos su accuracy.

```py
# Accuracy with only KNeighborsClassifier
knn_class = KNeighborsClassifier().fit(X_train, y_train)
knn_pred = knn_class.predict(X_test)

# Accuracy Bagging...
classifier = {
    'KNeighbors': KNeighborsClassifier(),
    'LogisticRegression' : LogisticRegression(),
    'LinearSCV': LinearSVC(),
    'SVC': SVC(),
    'SGDC': SGDClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomTreeForest' : RandomForestClassifier(random_state=0)
}

# for loop...
```

Los accuracy que obtuvimos en pantalla fueron:

```shell
Accuracy with only KNeighborsClassifier: 0.724025974025974
Accuracy Bagging with KNeighbors: 0.7727272727272727

Accuracy Bagging with LogisticRegression: 0.8474025974025974
Accuracy Bagging with LinearSCV: 0.8376623376623377
Accuracy Bagging with SVC: 0.7077922077922078
Accuracy Bagging with SGDC: 0.7402597402597403

Accuracy Bagging with DecisionTree: 0.9675324675324676
Accuracy Bagging with RandomTreeForest: 0.9805194805194806

```

El accuracy sin Bagging es menor que utilizando Bagging para el algoritmo de clasificación `KNeighborsClassifier`. Con esto concluimos en esta parte que implementando Bagging podemos mejorar el accuracy de nuestro modelo configurandolo de forma correcta.

La otra conclusión que podemos obtener luego de haber entrenado otros modelos de clasificación utilizando Bagging es que los accuracy son variados. Los de mejor performance son los algoritmos basados en `Tree`, `DecisionTree` y `RandomTreeForest`, donde el mejor fue el segundo.

## Implementación de Boosting

[Script implementando Bagging](boosting.py)

Implementaremos [GradientBoostingClassifier](https://en.wikipedia.org/wiki/Gradient_boosting) un algoritmo boosting basado en arboles de decisión. Podemos encontrar mas información en la [documentación oficial de Scikit Learn](https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting)

Hicimos la implementación con 50 estimadores y obtuvimos lo siguiente:

```shell
GradientBoostingClassifier accuracy : 0.9642857142857143
```

Luego analizamos un rango de estimadores y graficamos sus resultaados para observar cual seria un numero adecuado de estimadores.

![](https://imgur.com/C6e10Sn.png)

# 6. Clustering

## Estrategias de Clustering

Los [algoritmos de clustering](https://es.wikipedia.org/wiki/An%C3%A1lisis_de_grupos) son las estrategias que podemos usar para agrupar los datos de tal manera que todos los datos pertenecientes a un grupo sean lo más similares que sea posible entre sí, y lo más diferentes a los de otros grupos.

Cada circulo corresponde a un elemento, sus coordenadas representan sus características, los colores son el resultado del agrupamiento, que en este caso identificó 3 grupos.

![](https://imgur.com/X251zh8.png)

### Casos de aplicación de clustering

1. No conocemos con anterioridad las etiquetas de nuestros datos (Aprendizaje no supervisado).
2. Queremos descubrir patrones ocultos a simple vista.
3. Queremos identificar datos atípicos.

- Casos de uso de aplicación:
  - **Cuando sabemos cuántos grupos “k” queremos en nuestro resultado**.
    - Si es el caso, por ejemplo en una empresa de marketing y sabemos que los segmentos de clientes es bajo, medio alto, en este caso es recomendable usar k-means, o bien, spectral clustering.
  - **Cuando queremos que el algoritmo descubra la cantidad de grupos “k” óptima según los datos que tenemos**.
    - Por otro lado si no conocemos cuantos grupos o cuantas categories tenemos y solo queremos experimenter, la solución puede ser Meanshift, clustering jerárquico o DBScan.

## Implementación de Batch K-Means

[Implementacion del algoritmo](k_means.py)

Asumiremos que sabemos los grupos que implementariemos en el resultado final

Utilizaremos el dataset de [candi](datasets/candy.csv). Este nos dice las caracterisicas de diferentes caramelos. Podemos conocer mas el dataset en el su [readme](datasets/readme-dataset-candy.pdf).

Usamos la implementación [Mini Batch K-Means](https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans). Esta es una variante del algoritmo K-Means que usa mini batches (lotes) que reduce el tiempo de computo. La unica diferencia es que la calidad de los resultados es reducida.

Documentation: [sklearn.cluster.MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)

No implementamos el metodo del codo, al conocerce el datasets utilizamos la cantidad de cluster adecuada, aunque el metodo de seleccion de estos no fue el adecuado, ya que debe usarse el metodo de codo u otro.

![kmeans_paitplot](https://imgur.com/K9DqUE8.png)

En el sigueinte scatter se graficarón las siguientes columnas del datasets y se coloreo bajo una columna nueva creada gracias a la clusterizacion que hizo el algoritmos K Means.

- **sugarpercent**: ​ El percentil de azúcar en el que recae dentro del mismo dataset.
- **pricepercent**: El percentil de precio por unidad dentro del que se encuentra respecto al dataset.
- **winpercent**:​ Porcentaje de victorias de acuerdo a 269.000 emparejamientos al azar.

> Se observan los 4 diferentes colores, ya que se eligieron 4 clusters. Se puede observar una clara clusterizacion cuando se compara respecto a winpercent las variables pricepercent, sugarpercent.

## Implementactión de Mean-Shift

[Implementacion de Mean-Shift](meanshift.py)

Puede suceder que lo que necesitemos sea simplemente dejar que el algoritmo decida cuantas categorías requiere. Esto lo podremos hacer con el algoritmo [Mean-Shift](https://scikit-learn.org/stable/modules/clustering.html#mean-shift). El algoritmo de la clustering tiene como objetivo descubrir manchas en una densidad uniforme de muestras. O sea, diferenciar y clusterizar. Documentación oficial en [sklearn.cluster.MeanShift](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html).

![meanshift](https://imgur.com/hNuP2VY.png)

El algoritmo nos devolvio 3 clusters, porque le pareció que esa era la cantidad correcta teniendo en cuenta como se distrubuye la densidad de nuestros datos. Podemos ver eso mismo en el gráfico anterior.

> Se observan los 3 diferentes colores, clusters generados automaticamente por el algoritmo MeanShift. Se puede observar una clara clusterizacion cuando se compara respecto a winpercent las variables pricepercent, sugarpercent.

> **NOTA**: En la [documentación (en Scalability)](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html) se advierte que el algoritmo tiene una complejidad algorítmica que escala a **O(T*n^2)** a medida que el número de registros aumenta.

# 7. Optimización paramétrica

Hasta ahora hemos estudiado y hecho:

- Aprender a lidiar con Features antes de mandarlo al entrenamiento.
- Aprender modelo espeficidos para resolver problemas de gran complejidad.

Ahora nos toca la etapa final del proceso de Machine Learning, esto es:

- Validacion de lo que se ha hecho. Scikit Learn nos ofrece realizar este tipo de tareas de una manera casi automatizada.

## Validación de nuestros modelos. Hold Out y Cross Validation

- **La última palabra siempre la van a tener los datos.**
  - Todas nuestras intuiciones no tiene nada que hacer frente a lo que digan los datos y las matemáticas que aplicamos sobre estos datos. Por eso es importante siempre tener rigurosidad a la hora de evaluar los resultados que estamos recibiendo.

- **Necesitamos mentalidad de testeo.**
  - No se trata solamente de probar un poco al principio y un poco al final, sino que tendremos que probar constantemente durante todo el proceso, para poder encontrar cuál es la solución óptima que realmente nos soluciona el problema que tenemos pendiente, todo esto:
    - con varias formas
    - con varios conjuntos de datos
    - con varias configuraciones de parámetros
    - con varias distribuciones de nuestros datos

- **Todos los modelos son malos, solamente algunos son útiles.**
  - Todos los modelos que nosotros hacemos en últimas son una sobre simplificación de lo que pasa realmente. Entonces nunca nuestros modelos van a corresponder con la realidad al cien por ciento. Si jugamos lo suficiente y si somos lo suficientemente hábiles para configurar, vamos a llegar a un punto donde el modelo que estamos trabajando va a ser útil para ciertos casos específicos dentro del mundo real.

### Tipos de validación

#### Hold-Out

Se trata de dividir nuestros datos entrenamiento/pruebas, básicamente consiste en usar porcentajes fijos, por lo regular 70% de entrenamiento y 30% de pruebas.

![hold-out](https://imgur.com/SrmKn9O.png)
![hold-out-strategy](https://imgur.com/XYUiTib.png)

**¿Cuándo utilizar Hold-on?**

- Se requiere un prototipado rápido.
- No se tiene mucho conocimiento en ML.
- No se cuenta con abundante poder de cómputo.

#### K-Folds

Usar validación cursada K-Fold, aquí vamos a plegar nuestros datos k veces, el k es un parámetro que nosotros definimos y en esos pliegues vamos a utilizar diferentes partes de nuestro dataset como entrenamiento y como test, de tal manera que intentemos cubrir todos los datos de entrenamiento y de test, al finalizar el proceso.

![k-fold](https://imgur.com/dJvoief.png)
![k-fold-strtegy](https://imgur.com/Hfxy5UO.png)

**¿Cuándo utilizar K-Folds?**

- Recomendable en la mayoría de los casos.
- Se cuenta con un equipo suficiente para desarrollar ML.
- Se require la integración con técnicas de optimización paramétrica.
- Se tiene más tiempo para las pruebas.

#### LOOCV

Validación cruzada LOOCV, Leave One Out Cross Validation. Este es el método más intensivo, ya que haremos una partición entre entrenamiento y pruebas, porque vamos a hacer entrenamiento con todos los datos, salvo 1 y vamos a repetir este proceso tantas veces hasta que todos los datos hayan sido probados.

![loocv](https://imgur.com/fTW7139.png)

**¿Cuándo utilizar LOOCV?**

- Se tiene gran poder de computo
- Se cuetan con pocos datos para poder dividir por Train/Test
- Cuando se quiere probar todos los casos posibles (para personas con TOC)

## Implementación de K-Folds Cross Validation

[Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html#k-fold)

**[sklearn.model_selection.cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score)**

Se hizo una implementacion básica de Cross validation para calcular un score.

- Determinamos en cuantas partes queremos dividir nuestro dataset.
- De cada uno obtenemos una lista de scores, estos son errores medios cuadraticos de cada split.
- Sacamos la media de esos errores
- Obtenemos el valor final real sacando su valor absoluto.

```shell
****************************************************************
---- IMPLEMENTACION BÁSICA ----
****************************************************************
Los tres MSE fueron:  [-0.84508789 -0.15576388 -0.74578906]
================================
-0.5822136106357311
================================
El MSE promedio fue:  0.5822136106357311
```

**[sklearn.model_selection.KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html?highlight=kfold#sklearn.model_selection.KFold)**

Se hizo una implementacion mas sofisticada de K Fold Cross validation para calcular separar los datos en sus respectivos dataset y poder entrenar a nuestro modelo manualmente.

```shell
****************************************************************
---- IMPLEMENTACION DETALLADA ----
****************************************************************
Los tres MSE fueron:  [0.016092903866001332, 0.01771079365873104, 0.004653134441603091]
El MSE promedio fue:  0.012818943988778489
```

## Optimización de hiperparametros | Hyperparameter Optimization

Familiarizados con el concepto de Cross Validation vamos a utilizar este mismo principio de fondo para lograr automatizar un poco la selección y optimización de nuestros modelos.

**Problema**: Parece que encontramos un modelo de aprendizaje que parece funcionar, pero esto puede implicar que ahora tenemos que **encontrar la optimización de cada uno de los parámetros de este modelo**, encontrar el que mejor se ajuste y el que mejor resultado nos de.

1. Es facil perderse entre los conceptos de tantos parámetros. Tenemos flexibilidad para algoritmos básicos de Machine Learning, pero facil perderse.
2. Es difícil medir la sensibilidad de los mismos manualmente.
3. Es COSTOSO, en tiempo humano y computacionalmente.

Scikit Learn nos ofrece enfoques para automatizar el proceso de optimización paramétrica. Existen 3 enfoques principales, estos son:

1. Optimización manual
2. Optimizacion por grilla de parámetros | GridSearchCV
3. Optimizacion por búsqueda aleatorizada | RandomizedSearchCV

### Optimización manual

1. Escoger el modelo que queremos ajustar.
2. Buscar en la documentación de Scikit-Learn
3. Identificar parámetros y ajustes. Parámetros que vamos a necesitar y cuáles son los posibles ajustes que vamos a requerir para cada uno de estos parámetros.
4. Probar combinaciones una por una iterando a través de listas.

### Optimizacion por grilla de parámetros | GridSearchCV

Es una forma organizada, exhaustiva y sistematica de probar todos los parametros que le digamos que tenga que probar, con los respectivos rangos de valores que le aportemos.

1. Definir una o varias métricas que queremos optimizar.
2. Identificar los posibles valores que pueden tener los parámetros.
3. Crear un diccionario de parámetros.
4. Usar Cross Validation.
5. Entrenar el modelo (e ir por un café)

La grilla de parámetros nos define GRUPOS DE PARÁMETROS que serán probados en todas sus combinaciones (Un grupo a la vez)

Ejemplo:

![svm-gridsearch-optimized](https://imgur.com/SdgSupv.png)

### Optimizacion por búsqueda aleatorizada | RandomizedSearchCV

Si no tenemos tanto tiempo para una prueba tan exhaustiva o queremos combinaciones aleatorias usaremos este metodo. Es lo mismo que el caso anterior, pero busca de forma aleatoria los parametros y Scikit Learn selecciona los mejores de las combinaciones aleatorias que se hicieron.

En este método, definimos escalas de valores para cada uno de los parámetros seleccionados, el sistema probará varias iteraciones (Configurables según los recursos) y mostrará la mejor combinación encontrada.

Ejemplo:

![svm-randomized-search-optimized](https://imgur.com/CrCFl3W.png)

### GridSearchCV vs RandomizedSearchCV

- **GridSearchCV**
  - Cuando se quiera realizar un estudio a fondo sobre las implicaciones de los parámetros.
  - Se tenga tiempo.
  - Se tenga poder de procesamiento.

- **RandomizedSearchCV**
  - Cuando se quiera explorar posibles optimizaciones.
  - Haya poco tiempo.
  - Haya poco poder de procesamiento.

![GridSearch-vs-RandomizedSearch](https://imgur.com/UWdrx7j.png)

## Implementación de Randomized

[Implementación de Randomized](randomized.py)

Se implementó el optimizador RandomizedSearchCV utilizando RandomForestRegressor. El diccionario con los parámetros fue:

```py

parameters = {
        'n_estimators': range(4, 16), # rango de arboles que compondran el bosque
        'criterion': ['mse', 'mae'], # lista de criterio de 
        'max_depth': range(2, 11) # rango de profundidad de los arboles
    }

```

Luego pusimos a trabajar nuestro optimizador:

```py

    # n_iter=10, son 10 iteracion del optimizador. Toma 10 combinaciones al azar del diccionario
    # cv = 3, parte en 3 parte el set de datos que le pasemos, para hacer Cross validation
    rand_est = RandomizedSearchCV(reg, parameters, 
                                  n_iter=10, 
                                  cv=3, 
                                  scoring='neg_mean_absolute_error',
                                  ).fit(data, target)

```

Los valores obtenidos fueron:

```shell

================================================================
Mejores estimadores
----------------------------------------------------------------
RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=10, max_features='auto', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=4, n_jobs=None, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)
================================================================
Mejores parametros
----------------------------------------------------------------
{'n_estimators': 4, 'max_depth': 10, 'criterion': 'mse'}
================================================================
Pruebas
----------------------------------------------------------------
Predict: 7.52625012375
Real:    7.537
Name: 0, dtype: float64
================================================================

```

> Podemos ver los parámetros seleccionados por nuestro optimizador en 'Mejores parámetros'. Con esos parámetros seleccionados hemos predicho un valor y obtuvimos lo que se ve en 'Pruebas', valores muy acertados.

## BONUS: Auto Machine Learning

Scikit-learn nos permite semi-automatizar la optimización de nuestros modelos con GridSearchCV y RandomizedSearchCV, ¿Cuál es el límite de esta automatización? Haciendonos esta pregunta nace un nuevo concepto Automated Machine Learning.

> **Automated Machine Learning (AutoML)**, es un concepto que en general pretende la completa automatización de todo el proceso de Machine Learning, desde la extracción de los datos hasta su publicación final de cara a los usuarios.

Sin embargo, este ideal aún está en desarrollo en la mayoría de las etapas del proceso de Machine Learning y aún se depende bastante de la intervención humana.

Podemos encontrar más información leyendo el siguiente enlace: [Qué es Automated Machine Learning: la próxima generación de inteligencia artificial](https://itmastersmag.com/noticias-analisis/que-es-automated-machine-learning-la-proxima-generacion-de-inteligencia-artificial/)

Existe una implementación de este concepto utilizando Scikit Learn llamado auto-sklearn. Esto nos ayudará a llevar un paso más lejos nuestro proceso de selección y optimización de modelos de machine learning. Dado que automáticamente prueba diferentes modelos predefinidos y configuraciones de parámetros comunes hasta encontrar la que más se ajuste según los datos que le pasemos como entrada. Con esta herramienta podrás entrenar modelos tanto de clasificación como de regresión por igual.

[Lista de los clasificadores disponibles](https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/classification)

[Lista de los regresores disponibles](https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/regression)

Se puede añadir modelos personalizados al proceso siguiendo los pasos descritos en la documentación.

### auto-sklearn

Esta herramienta es una librería basada en los algoritmos de scikit-learn, aunque hay que tener presente que es una librería externa y se debe instalar siempre por aparte. En todo caso al ser una librería de Python se puede combinar sin ningún problema con el resto de nuestro código desarrollado para scikit-learn, incluso permitiendo la exportación de modelos ya entrenados para su posterior uso.

[Documentación de auto-sklearn](https://automl.github.io/auto-sklearn/master/index.html)

# 8. Salida a producción

## Revisión de nuestra arquitectura de código

Ahora vamos a convertir los scripts que tenemos en un código que sea modular y extensible con facilidad para que nuestra arquitectura pueda salir a producción de una manera exitosa.

Una estructura de carpetas que sea organizada para poder gestionar todo lo que vas a necesitar en cualquier proceso de Machine Learning.

Carpetas:

- [in](project/in): Carpeta que contendrá archivos de entrada, datos que alimentarán a nuestros modelos.
- [out](project/out): Carpeta que contendrá el resultado de la exportacion de nuestros modelos, visualizaciones, datos en excel o csv, etc.
- [models](project/models): Carpeta que contedrá a los modelos.

Archivos:
Cada clase será un archivo que tenga su propia responsabilidad y se encargue específicamente de una tareas concreta.

- [main.py](project/main.py): Metodo principal de ejecucion. Ejecutará todo el flujo de datos. Se encargaría de controlar el flujo de todo el código de principio a fin.
- [load.py](project/load.py): Archivo que se encarga de cargar los datos desde in o una DB
- [utils.py](project/utils.py): Todos los metodos que se reutilizaran una y otra vez.
- [models.py](project/models.py): Irá toda la parte de ML como tal.

## Importar y exportar modelos con Sklearn

Se creo la clase Models que contiene:

[models.py](project/models.py)

- Metodo grid_training(): Metodo para seleccionar al mejor modelo con el mejor score. Trabaja sobre los atributos, que son diccionarios de modelos y sus respectivos rangos y opciones de parámetros. Se utiliza el optimizador Grid y se selecciona finalmente el mejor modelo y el que mas score entrega de estos.
- Atributos:
  - reg: Atributo que contiene a los regresores en diccionarios. Estos son los modelos que se utilizarán
  - params: Atributo que contiene a los parámetros de cada modelo en diccionario.

[utils.py](project/utils.py)

Funcion para exportar a nuestros modelos en [pickle](https://www.datacamp.com/community/tutorials/pickle-python-tutorial).

## Creación de una API con Flask para el modelo

- Instalamos flask
- Eliminamos load.py, ya que no lo utilizaremos finalmente
- Creamos el archivo [server.py](project/server.py) para crear un servidor local para nuestra API.
  - En este creamos la funcion predict(), que será la expuesta en nuestro servidor con el metodo GET, en la direccion 8080/predict y que muestra la prediccion hecha. La prediccion se hace con datos de pruebas y con nuestro modelo que exportamos al archivo [best_model.pkl](project/models/best_model.pkl)

Tenemos entonces un JSON que tiene una llave que se llama predicción y el valor de la predicción que nos generó nuestro modelo según los datos que le pasamos de configuración.

```
http://127.0.0.1:8080/predict
```

![json-servidor](https://imgur.com/lwhSk7b.png)

Así podemos entonces ver un ejemplo de cómo podríamos salir a producción.

Ya el JSON tendríamos que tratarlo, si estamos desarrollando una aplicación móvil o una plataforma web, podríamos trabajarlo con JavaScript o desde Android sin importar la naturaleza lo que estemos haciendo.

Con esto ya tenemos las predicciones y tenemos un sistema que se conecta a nuestro modelo y nos trae los resultados de una manera extensible, modular, fácil de utilizar y que podemos convertir en la solución que estamos buscando.

Así damos por finalizado la construcción de la arquitectura para salir a producción de nuestro modelo Inteligencia artificial.

--------------------------
# Conclusiones

--------------------------

## Manejo de features

[Optimización de features](#3-optimización-de-features)

En el curso aprendimos cómo tratar con nuestro features y como seleccionarlos para extraer la información más importante. Esto es optimización de features a través de PCA, IPCA, KPCA. Tambienn Regularización e implementación de Lasso y Ridge

--------------------------

## Algoritmos de ML



También como construir algunos modelos de Machine Learning aún para casos bastante complejos como los que vimos.

Nos adentramos en las tres areas de Machine Learning mas importantes como son:

[Regresiones robustas](#4-regresiones-robustas): Estudiamos sobre Regresiones robustas y como implementarlas para evitar valores atípicos.

[Métodos de ensamble aplicados a clasificación](#5-métodos-de-ensamble-aplicados-a-clasificación): Estudiamos métodos de ensamble aplicados a clasificación, preparamos datos e implementamos Bagging y Boosting.

[Clustering](#6-clustering): Estudiamos estrategias de Clustering y como implementar Batch K-Means y Mean-Shift

--------------------------

## Validacion y optimizacion de hiperparametros

[Optimización paramétrica](#7-optimización-paramétrica)

Se le dedico un modulo completo a como validar nuestros modelos. Conocimos en profundiad los tipos de validación (Hold-Out, K-Folds, LOOCV). Esto se lo conoce como Cross Validation.

Luego en el mismo modulo conocimos y estudiamos sobre Optimización paramétrica o Hyperparameter Optimization. Implementamos GridSearchCV y RandomizedSearchCV

--------------------------

## Como exponer un modelo en produccion

[Salida a producción](#8-salida-a-producción)

Finalmente cómo sacarlos a producción a través de una APIrest

Formamos una arquitectura de archivos y carpetas para nuestro código, importar y exportar modelos con Sklearn y creamos una APIrest con Flask para el modelo.
