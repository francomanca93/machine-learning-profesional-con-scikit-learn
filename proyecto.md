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
- [5. Métodos de ensamble aplicados a clasificación](#5-métodos-de-ensamble-aplicados-a-clasificación)
- [6. Clustering](#6-clustering)
- [7. Optimización paramétrica](#7-optimización-paramétrica)
- [8. Salida a producción](#8-salida-a-producción)

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

# 5. Métodos de ensamble aplicados a clasificación

# 6. Clustering

# 7. Optimización paramétrica

# 8. Salida a producción
