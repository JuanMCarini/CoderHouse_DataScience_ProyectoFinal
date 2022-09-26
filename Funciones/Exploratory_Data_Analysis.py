# En este script vamos a declarar todas las funciones que vamos a utilizar en la unidad donde desarrollamos el EDA:

from pandas.api.types import is_numeric_dtype
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


def univariado_info(df,etiquetas):
    '''Calculo de informacion estadistias y genericas de cada columna de un dataframe
    df: dataframe
    etiquetas: diccionario con las etiquetas del dataframe
    ''' 
    # Creamos un dataframe con columnas especificas:
    df_info = pd.DataFrame(columns=['Cantidad', 'Tipo' , 'Missing', 'Unicos', 'Numeric'])
    # Loop de todas las variables del dataframe
    for col in df:  
        #obtengo info de la columna
        data_series = df[col]
        #lleno dataframe con las columnas iniciales
        df_info.loc[col] = [data_series.count(), data_series.dtype, data_series.isnull().sum(), data_series.nunique(), is_numeric_dtype(data_series)]   
    # Calculo el describe 
    df_describe = df.describe(include='all').T[['top', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    # Calculo sesgo y curtosis
    df_stats = pd.DataFrame([df.skew(numeric_only=True), df.kurtosis(numeric_only=True)], index=['sesgo', 'kurt']).T  

    return pd.concat([pd.Series(etiquetas,name='Etiqueta'),pd.concat([df_info,df_describe, df_stats], axis=1)], axis=1).fillna('-')


def hist_box(DataFrame:pd.DataFrame, #dataset
             x, #variable del gráfico
             limite,# frecuencia del rango para la etiqueta del eje x
             bins,
             titulo:str,
             ruta:str):
    '''
    +Armamos una función para graficar y jugar con el nivel del filtrado de la variable y obtener un histograma 
    que permita apreciar mejor la distribución de la variable sin tantos outliers   
    DataFrame:pd.DataFrame, -> dataset
    x -> variable del gráfico
    limite -> frecuencia del rango para la etiqueta del eje x
    bins -> divisiones del  histograma
    titulo -> título superior del gráfico
    ruta -> ruta donde se guardara la imagen
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
    ax[0].set_ylabel("", size=15)
    ax[1].set_title("")
    ax[1].set_xlabel("Ingreso per capita familiar", size=15)
    ax[1].set_ylabel("Conteo", size=15)
    ax[0].set(xticks=range(0,limite+1,int(limite/20)))
    
    for i in [0,1]:
        ax[i].tick_params(axis='both',
                labelsize=15
                )
        ax[i].set_facecolor('white')

    fig.savefig(ruta)


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