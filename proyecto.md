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

# 4. Regresiones robustas

# 5. Métodos de ensamble aplicados a clasificación

# 6. Clustering

# 7. Optimización paramétrica

# 8. Salida a producción
