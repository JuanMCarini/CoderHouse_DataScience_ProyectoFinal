# Resumen
 
A lo largo de este proyecto, se ha trabajado sobre una _Encuesta Anual de Hogares_ realizada por el Gobierno de la Ciudad de Buenos Aires para el año 2019. Sobre dicho dataset se ha realizado un análisis de datos exploratorio, definiendo y analizando sus variables y estableciendo correlaciones a nivel binario y multivariable. Finalmente, en pos de alcanzar los objetivos específicos del proyecto, se han utilizado modelos de clasificación.

El objetivo de aplicar modelos de clasificación ha sido encontrar el modelo que traiga mejores resultados a fin de poder predecir la variable Target, el Nivel Máximo Educativo, utilizando al resto de las variables.

En ese sentido, se ha implementado un árbol de clasificación y un bosque aleatorio, parametrizando, en cada uno de los casos, ciertas variables convenientes. Asimismo, para ambos modelos, se ha implementado algoritmos de optimización  a fin de seleccionar los mejores parámetros para el problema de optimización y mitigar el overfitting del los modelos de partida. 

Con respecto a las conclusiones alcanzadas, se han visto buenos resultados en los modelos optimizados con hiperparámetros, los cuales han performado de manera esperada, alcanzando modelos robustos.

# Abstract

Throughout this project, we have worked on an _Annual Household Survey_ conducted by the Government of the City of Buenos Aires for the year 2019. An exploratory data analysis has been carried out on this dataset, defining and analyzing its variables and establishing correlations between them. Finally, in order to achieve the specific objectives of the project, classification models have been used.

The objective of applying classification models has been to find the model that brings the best results to predict the Target, the Highest Level of Education, using the rest of the variables.

In this sense, a classification tree and a random forest have been implemented, parameterizing, in each case, certain variables. Likewise, for both models, optimization algorithms have been implemented in order to select the best parameters for the optimization problem and mitigate the overfitting of the starting models. 

Regarding the conclusions reached, good results have been seen in the models optimized with hyperparameters, which have performed as expected, reaching robust models.

# Estructura de archivos
* :ringed_planet: Proyecto Final.ipynb: Jupyter Notebook del proyecto final, con código en PYthon
* :books: encuesta-anual-hogares-2019.csv: Base de datos con la que trabajaremos en el proyecto.
* :world_map: comunas.geojson: Mapa con las comunas de la Ciudad Autónoma de Buenos Aires. Será útil para realizar gráficos
* :hammer_and_wrench::snake: Funciones: Lista de funciones en formato .py para su uso en el jupyter notebook
    * Exploratory_Data_Analysis.py: Funciones para el análisis exploratorio de datos
    * Modelado.py: Funciones para la generación de modelos para su entrenamiento
* :chart: Presentación
    * Presentación Proyecto Final: en formato .pdf y .tex
    * CSV: Tablas utilizadas para construcción de la presentación
    * Imágenes: Gráficos utilizados para la presentación
* :closed_book: Informe
    * Informe Proyecto Final: en formato .pdf y .tex
    * CSV: Tablas utilizadas para el informe
    * Imágenes: Gráficos utilizados para el informe