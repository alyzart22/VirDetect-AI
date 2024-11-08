#Alida Zarate
#23 febrero primera version de modelo deeplearning DUMMY: Mod 13 Oct 2022 para probar ya mis modelos descentes XD
#De entrada necesita 2 datos, path, nombre_bd
#modificado septiembre 2023 para recibir parametros y usarlo en ibt pc

import os
import time
import sys #para pasar parametro
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split #Este metodo divide los datos

#print(tensorflow.__version__)

#------Metodos----------------
def pasar_bd_DF(bd):
    pasar_df=pd.DataFrame()
    pasar_df[0]=bd[0]
    pasar_df[1]=bd[1]
    pasar_df[2]=bd[2]
    return pasar_df

def porcentajes(total, valor):
    return round( ((100/total)*valor), 2)

#-----------------------------

#Datos Entrada:
'''path='D:/UAEM/BIO/Dataset/datos_dumi/'
nombre_bd='db_dummy.csv' '''

'''path='D:/UAEM/BIO/Dataset/datos_liz/'
nombre_bd='db_liz_test.csv' '''

'''path='D:/UAEM/BIO/Dataset/datos_dummi/'
nombre_bd='dataset_dummi.csv' '''

#longitud="100_3"
'''longitud="muestra_pro_100_10"
path='D:/UAEM/BIO/Dataset/datos_dummi_kmers/'+longitud+'/'
nombre_bd= 'dataset_dummi_'+longitud+'.csv' '''

'''#Eucariontes30
longitud="100_10"
path='D:/UAEM/BIO/Dataset/JULIO_2022/eucariontes30/datos_kmer/'+longitud+'/'
nombre_bd= 'dataset_euk30_'+longitud+'.csv' '''

#procarioentes30
'''
longitud="100_10"
path='D:/UAEM/BIO/Dataset/JULIO_2022/procariontes30/datos_kmer/'+longitud+'/'
nombre_bd= 'dataset_pro30_'+longitud+'.csv'
'''
longitud = sys.argv[1]#prefijo de la subcarpeta, aqui se pueden guardar diferentes tamaños de kmers
path = sys.argv[2]#path principal donde esta las carpetas a las cual acceder
path = path+longitud+'/'
nombre_bd = sys.argv[3]+longitud+'.csv'#Prefijo del archivo de entrada el cual se va a splitear

#------------------------------

#Pasar los datos generales a un DataFrame
#bd_dummi = pd.read_csv ('D:/UAEM/BIO/Dataset/datos_dumi/db_dummy.csv', header= None)
print("PATH:",path+nombre_bd)
bd_dummi = pd.read_csv (path+nombre_bd, header= None)
dataset_general=pasar_bd_DF(bd_dummi)
total=len(dataset_general)

#Obtener el numero de clases dentro del dataset general, por que se usa para la funcion de split en stratify
Idclases=dataset_general[0]
#guardamos las clases totales, lapasamos a Dataframe para guardarla mas abajo
total_clases = pd.unique(Idclases)
clases = pd.DataFrame()
clases[0]=total_clases #Abajo terminamos de exportar en csv


#Agarrar el dataset general y partirlo para obtener el conjunto test
X_train, X_test = train_test_split(dataset_general, test_size=0.10, random_state=42, stratify=Idclases)#0.05 5% del 100, 0.03 3%
#X_train, X_test = train_test_split(dataset_general, test_size=0.10, random_state=0, stratify=Idclases)#0.05 5% del 100, 0.03 3%

#Obtener lista de clases del datset train, por que se usa para la funcion de split en stratify
Idclasestrain=X_train[0]

#Agarrar el dataset X_train y lo partimos para obtener el conjunto val
X_train, X_val= train_test_split(X_train, test_size=0.10, random_state=0, stratify=Idclasestrain)#0.1   0.15 =14%

#Generar los archivos csv y guardarlos localmente
X_train.to_csv(path+'train_'+longitud+'.csv', header= None, index=False)
X_val.to_csv(path+'val_'+longitud+'.csv', header= None, index=False)
X_test.to_csv(path+'test_'+longitud+'.csv', header= None, index=False)
clases.to_csv(path+'clases_'+longitud+'.csv', header= None, index=False)

#Resumen de los tamaños de los datasets
print(":::::::::: Summarize Dataset::::::::::::::")
print("Train: Filas/columnas:", X_train.shape, "%",porcentajes(total, len(X_train)) )
print("Dev: Filas/columnas:", X_val.shape, "%",porcentajes(total, len(X_val)) )
print("Test: Filas/columnas:", X_test.shape, "%",porcentajes(total, len(X_test)) )
print("Número de clases:", len(clases))
print(":::::::::::::::::::::::::::::::::::::::::::")


