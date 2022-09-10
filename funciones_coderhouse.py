#En este script vamos a declarar todas las funciones que vamos a utilizar para así tener un notebook más limpio

from pandas.api.types import is_numeric_dtype
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# EDA
def diccionario():
    '''
    Esta función trae el diccionario con las etiquetas de las variables, las cuales nos permiten generar etiquetas para los 
    gráficos de forma rápida, así como obtener mayor información sobre las mismas.
    '''
    etiquetas = {
    'id'                          : 'Clave que identifica la vivienda',
    'nhogar'                      : 'La variable id + nhogar = clave que identifica a cada hogar',
    'miembro'                     : 'Variables id + nhogar + miembro = clave que identifica a cada persona',
    'comuna'                      : 'Comuna donde reside la persona encuestada',
    'edad'                        : 'Edad de la persona encuestada',
    'sexo'                        : 'Sexo de la persona encuestada',
    'parentesco_jefe'             : 'Parentesco entre la persona encuestada y el jefe de hogar',
    'situacion_conyugal'          : 'Situación conyugal de la persona encuestada',
    'num_miembro_padre'           : 'Número de miembro que corresponde al padre',
    'num_miembro_madre'           : 'Número de miembro que corresponde a la madre',
    'estado_ocupacional'          : 'Situación ocupacional de la persona encuestada',
    'cat_ocupacional'             : 'Categoría ocupacional de la persona encuestada',
    'calidad_ingresos_lab'        : 'Calidad de la declaración de ingresos laborales totales',
    'ingreso_total_lab'           : 'Ingreso total laboral percibido el mes anterior', 
    'calidad_ingresos_no_lab'     : 'Calidad de la declaración de ingresos no laborales totales', 
    'ingreso_total_no_lab'        : 'Ingreso total no laboral percibido el mes anterior',
    'calidad_ingresos_totales'    : 'Calidad de ingresos totales individuales', 
    'ingresos_totales'            : 'Ingreso total individual percibido el mes anterior',
    'calidad_ingresos_familiares' : 'Calidad de ingresos totales familiares',
    'ingresos_familiares'         : 'Ingresos totales familiares percibido el mes anterior',
    'ing_per_cap_familiar'        : 'Ingreso familiar per capita percibido el mes anterior', 
    'estado_educativo'            : 'Asistencia (pasada o presente) o no a algún establecimiento educativo', 
    'sector_educativo'            : 'Sector al que pertenece el establecimiento educativo al que asiste',
    'nivel_actual'                : 'Nivel cursado al momento de la encuesta',
    'nivel_max_educativo'         : 'Máximo nivel educativo que se cursó',
    'años_escolaridad'            : 'Años de escolaridad alcanzados',
    'lugar_nacimiento'            : 'Lugar de nacimiento de la persona encuestada',
    'afiliacion_salud'            : 'Afiliación de salud de la persona encuestada',
    'hijos_nacidos_vivos'         : 'Tiene o tuvo hijos nacidos vivos',
    'cant_hijos_nac_vivos'        : 'Cantidad de hijos nacidos vivos',
    'dominio'                     : '¿la vivienda se ubica en una villa de emergencia?',
    'Target'                      : 'Nivel máximo educativo'
    }
    return etiquetas

#Tomamos el código visto en clase para tener un vistazo de las diversas medidas estadísticas de cada variale

def univariado_info(df,etiquetas):
  '''Calculo de informacion estadistias y genericas de cada columna de un dataframe
  df: dataframe
  etiquetas: diccionario con las etiquetas del dataframe
  '''

  #Creamos un dataframe con columnas especificas:
  df_info = pd.DataFrame(columns=['Cantidad', 'Tipo' , 'Missing', 'Unicos', 'Numeric'])
  #loop de todas las variables del dataframe
  for col in df:

      #obtengo info de la columna
      data_series = df[col]
      #lleno dataframe con las columnas iniciales
      df_info.loc[col] = [data_series.count(), data_series.dtype, data_series.isnull().sum(), data_series.nunique(), is_numeric_dtype(data_series)]

  #calculo el describe 
  df_describe = df.describe(include='all').T[['top', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
  #calculo sesgo y curtosis
  df_stats = pd.DataFrame([df.skew(), df.kurtosis()], index=['sesgo', 'kurt']).T

  return pd.concat([pd.Series(etiquetas,name='Etiqueta'),pd.concat([df_info,df_describe, df_stats], axis=1)], axis=1).fillna('-')

def hist_box(DataFrame:pd.DataFrame, #dataset
             x, #variable del gráfico
             limite,# frecuencia del rango para la etiqueta del eje x
             bins,
             titulo:str):
  '''
  Armamos una función para graficar y jugar con el nivel del filtrado de la variable y obtener un histograma 
  que permita apreciar mejor la distribución de la variable sin tantos outliers

  DataFrame:pd.DataFrame, -> dataset
  x -> variable del gráfico
  limite -> frecuencia del rango para la etiqueta del eje x
  bins -> divisiones del  histograma
  titulo -> título superior del gráfico
  '''
  fig, ax = plt.subplots(figsize=(20,10),nrows=2,ncols=1,sharex=True)
  sns.histplot(x=x,
              data=DataFrame[DataFrame[x]<limite] ,
              color='#004488',
              ax=ax[1],
              bins= bins,
              kde=True)
  
  ax[0].boxplot(x=DataFrame[DataFrame[x]<limite][x],vert=False)
  ax[0].set_title(f"{titulo} menor a {limite}",size=25)
  ax[1].set_title("")
  ax[0].set(xticks=range(0,limite+1,int(limite/20)))
  #fig.savefig('Informe/Imagenes/AUIngFam.png')


def freq_table(df:pd.DataFrame, col):
    '''
    Tabla de frecuencias relativas y absolutas
    df: DataFrame
    col:variable del dataframe para analizar
    '''
  #seleccion de data
    data = df[col]

    #verificacion de columna y si queremos cortes de intervalos
    if not is_numeric_dtype(data):
    #or not with_cuts:
        #generaion de tabla de frecuenca ,con info absoluta
        freq_tab = pd.crosstab(data, columns='FreqAbs').sort_values('FreqAbs', ascending=False)
    
    #calculo del resto de la tabla de frecuencia
    freq_tab['FreqRel'] = freq_tab['FreqAbs'] / freq_tab['FreqAbs'].sum()
    freq_tab[['FAbsAcumulada', 'FAbsRelativa']] = freq_tab[['FreqAbs','FreqRel']].cumsum()
    return freq_tab

def corrFilter(x: pd.DataFrame, thres: float):
    '''
    Tabla de correlaciones filtrado a partir del puntaje de las mismas
    x: DataFrame
    thres: valor mínimo de la correlación para ser filtrada en el dataframe
    '''
    #generate corr 
    xCorr = x.corr('spearman')
    #filter corr by thres
    xFiltered = xCorr[((xCorr >= thres) | (xCorr <= -thres)) & (xCorr !=1.000)]
    #change dataframe format
    xFlattened = xFiltered.unstack().drop_duplicates().reset_index().sort_values(0, ascending= False).dropna()
    #rename columns
    xFlattened.columns = ['Variable_1', 'Variable_2', 'corr_value']
    return xFlattened

# Modelado

def info_nulos(data:pd.DataFrame,ruta:str):
    """
    Crea un dataset limpio con los valores nulos y lo guarda para el informe

    data: dataframe
    ruta: ruta del archivo para el informe
    """
    nulos = data.isnull().sum().where(lambda x:x>0).dropna().apply(int).sort_values()
    if len(nulos)>0:
        print(f'Variables con valores nulos:\n\n{nulos}')
    else: print(f'El dataset no tiene valores nulos')
    nulos.to_csv(ruta)


def get_dataframe_info_to_csv(df:pd.DataFrame,ruta:str):
    """
    Función que me de un archivo .csv para poder pasar el .info 
    de forma más o menos automatica al informe. 

    input
       df -> DataFrame
       ruta -> ruta donde vamos a guardar el archivo .csv
    """

    df_types = pd.DataFrame(df.dtypes)
    df_nulls = df.count()
    
    df_null_count = pd.concat([df_types, df_nulls], axis=1)
    df_null_count = df_null_count.reset_index()
    
    # Reassign column names
    col_names = ["features", "types", "non_null_counts"]
    df_null_count.columns = col_names
    
    df_null_count.to_csv(ruta)

    # Generamos un pipeline para procesar la base

#armo una función para procesar dataframes con el pipeline
def pipe(X: pd.DataFrame,preprocessor:Pipeline,cat_col:list):
  '''
  Función para pasar dataframes por un pipeline
  x: dataframe
  preprocesor: pipeline
  '''
  #obtencio. de los procesos
  preprocessor.fit(X)
  #trasnformacion
  array_enc = preprocessor.transform(X)
  #genero nombres de columnas
  columns_enc = np.append(X.select_dtypes(exclude='object').columns, 
                          preprocessor.named_transformers_['cat']['ohe'].get_feature_names_out(cat_col))
  X = pd.DataFrame(array_enc, columns = columns_enc, index = X.index)
  return X

# Armamos una función para combinar el pipeline y la división de la base de datos
def procesador(data:pd.DataFrame,test_size:int,random_state:int):
    """
    Procesador que devuelve el train y el test procesado y listo para usar en nuestros modelos

    data: dataframe
    test_size: proporción de train y test
    random_state: semilla para replicar nuestro trabajo
    """
    #obtener columnas categoricas
    cat_col = data.select_dtypes(include='O').columns.tolist()

    #generar a pipeline para numericas 
    numeric_transformer = Pipeline(steps=[("scaler", MinMaxScaler())])

    #generar un pipeline para categoricas
    categorical_transformer = Pipeline(steps=[("ohe",OneHotEncoder(handle_unknown="ignore"))])

    #funcion que realiza el trabajo para cada pipeline y luego unirlo
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_exclude=object)),
            ("cat", categorical_transformer, make_column_selector(dtype_include = object)),
        ])
    #creo un dataset train y otro test 70-30 %%
    train, test = train_test_split(data, test_size = test_size,random_state=random_state, shuffle=True)
    # spliteo de datos y target del test
    X_train = train.loc[:,train.columns != 'Target'] 
    y_train = train.Target
    X_test = test.loc[:,test.columns != 'Target'] 
    y_test = test.Target

    # Paso el train y el test por el pipeline
    X_train = pipe(X_train,preprocessor,cat_col)
    X_test  = pipe(X_test,preprocessor,cat_col)

    #Chequeamos que las bases procesados por el one hot encoding tengan la misma cantidad de columnas
    if X_train.shape[1]!=X_test.shape[1]:
        print('Las particiones train y test tienen una cantidad diferente de columnas')
    else:
        print('Las particiones train y test tienen la misma cantidad de columnas')
    
    return X_train, y_train, X_test, y_test

def Classification_Report_to_csv(y_test, y_pred_test, ruta:str):
    """Función para guardar los classification reports"""
    clas_rep = classification_report(y_test,y_pred_test, output_dict=True,target_names=['inicial','primario','secundario','superior'])
    clas_rep = pd.DataFrame(clas_rep).transpose().to_csv(ruta)



def print_scores(model ,X_train , Y_train,y_test,predictions):
    """Armamos una función para imprimir el valor de la varianza y el sesgo
    model: modelo que generamos (arbol decisorio, randomforest,etc)
    X_train: features para entrenar
    y_train: target para entregar
    y_test: target para testear performance
    predictions: predicciones del modelo
    """
    sesgo = round(model.score(X_train, Y_train),4)
    varianza = round(sesgo-accuracy_score(y_test,predictions),4)

    print('Análisis de sesgo y varianza')
    print("------------------------------------------")
    print(f"La varianza tiene un valor de {varianza}")
    print(f"El sesgo tiene un valor de {sesgo}")
    return sesgo, varianza

def predictores(df:pd.DataFrame,modelo,threshold:int):
    print("Importancia de los predictores en el modelo")
    print("-------------------------------------------")
    feature_importances_df = pd.DataFrame(
        {"feature": list(df.columns), "importance": modelo.feature_importances_}
    ).sort_values("importance", ascending=False)

    #variables que dan 0 de importancia, las usaremos para quitarlas del próximo modelo y ahorrar tiempo de cómputo
    sin_imp = feature_importances_df[feature_importances_df['importance']==0]['feature'].to_list()

    # Mostrar
    return feature_importances_df[feature_importances_df['importance']>threshold] , sin_imp
