# En este script vamos a declarar todas las funciones que vamos a utilizar en la unidad donde desarrollamos los modelos analiticos:

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector


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


def ClassRept_to_csv(y_test, y_pred_test, ruta:str):
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