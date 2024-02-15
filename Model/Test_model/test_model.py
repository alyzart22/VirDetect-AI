# -*- coding: utf-8 -*-
#Alida Zarate
#4 mayo - Programa que carga un modelo DL y evalua rendimiento // mod 5 Diciembre /mod 2 de marzo 2023
#pide path y el nombre de los archivos necesarios(archivos test y del modelo)
# COMO CORRER python3 api_model_analisis.py /scratch/azarate/ali_project/Data/euk30/euk30_100_10/ euk30 100_10 /scratch/azarate/ali_project/Data/euk30/euk30_100_10/modelo3.h5
# COMO CORRER python api_model_analisis.py D:/UAEM/BIO/Dataset/JULIO_2022/eucariontes30/datos_kmer/100_10/ euk30 100_10 D:/modelo4.h5

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Esto es para evitar un mensaje de tensor 
import csv
from pyexpat import model
import tensorflow as tf #agrege yo
import numpy as np #Array
import pandas as pd #DF
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from keras.models import load_model
from progress.bar import Bar
#from tensorflow.keras.metrics import Recall, Precision

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from keras import backend as K
from sklearn.metrics import matthews_corrcoef, confusion_matrix

from datetime import datetime

'''gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)'''



#metodo que leer el archivo y lo guarda en una lista
def agregar_a_muestra(csv_filepath, muestras):
    with open(csv_filepath) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            muestras.append(line)
    return muestras
#Metodo que se le pasa la directiva de los splits 
def get_datos(ruta, nombre_archivo):
    directiva_archivo=ruta+nombre_archivo
    muestras=[]
    #llama la metodo que lee y guarda el contenido del archivo a a analizar
    muestras = agregar_a_muestra(directiva_archivo, muestras)
    #mezcla las muestras
    #samples=shuffle(samples)
    #numero de ejemplos en el data set completo
    muestras = muestras[:]
    num_muestras=len(muestras)
    return muestras,num_muestras


#Metodo para pasar de letras a numerico
def conv_letra_num(secuencia):
    #print(secuencia)
    Lette_dict = {'X': 0,'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,'Z': 21,'U':22, 'B':23, 'O':24,'J':25}
    seq_list = list(secuencia)
    seq_list = [Lette_dict[base] for base in seq_list]
    return seq_list
    
def recorre_lineas(lista_lineas):
    #Almacena la columna donde estan los kmers
    lineas_kmeros=lista_lineas[-1]
    #Almacena la columna donde estan las etiquetas
    lineas_etiquetas=lista_lineas[0]
    #Convertir a letras el conjunto de lineas
    kmers_convertidos=conv_letra_num(lineas_kmeros)
    return kmers_convertidos, lineas_etiquetas

#Hacer generador
def generador(muestra, tamano_batch, num_clases):
    total_de_muestras=len(muestra)
    while 1:
        
        for x in range(0, total_de_muestras, tamano_batch):
            #crea los chunks a partir de las muestras
            muestras_batch=muestra[x:x+tamano_batch]          
            lista_datos=[]
            lista_etiquetas=[]
            #recorremos cada linea del batch para pasar a categorico
            for muestras_batch in muestras_batch:
                linea_nueva, etiqueta = recorre_lineas(muestras_batch)
                lista_datos.append(linea_nueva)
                lista_etiquetas.append(etiqueta)               
            #SECUENCIAS
            #transforma la lista a una matriz (batch_sizex121)
            x_data=np.array(lista_datos)
            #print(x_data)
            
            x_data=to_categorical(x_data,26, dtype='uint8')
            
            #print(x_data)
            #ETIQUETAS
            df_batch_etiquetas=pd.DataFrame(np.array(lista_etiquetas), index=None)
            y_data=df_batch_etiquetas.iloc[:,0:1]
            y_data=np.array(y_data)#Esto lo puse pa ver si mejoraba el error

            y_data=to_categorical(y_data, num_classes=num_clases, dtype='uint16')
            print("bytes_x",x_data.nbytes)
            print("bytes_y",y_data.nbytes)

            #print(":::::::::::SHAPE")

            # :::ERROR SOLVED::: PAsar a tensores por q si lo dejaba como numpy.ndarray saltaba error 
            #x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
            #y_data = tf.convert_to_tensor(y_data, dtype=tf.float32) #float32
                        # :::ERROR SOLVED::: PAsar a tensores por q si lo dejaba como numpy.ndarray saltaba error 
            x_data = tf.convert_to_tensor(x_data, dtype=tf.int8)
            y_data = tf.convert_to_tensor(y_data, dtype=tf.int8) #float32


            #print("*************SHAPE ")
            #print(x_data.shape)
            
            #Generamos un generador
            #print(y_data.shape)
            #print(type(y_data))
            #print(y_data)
            yield x_data,y_data

#Funciones de metricas
def metricas(originales, predichos):
    print("::::::::::::::::::::::::  METRICAS MANUALES  ::::::::::::::::::::::::::::::")
    TP = tf.math.count_nonzero(predichos * originales)
    TN = tf.math.count_nonzero((predichos - 1) * (originales - 1))
    FP = tf.math.count_nonzero(predichos * (originales - 1))
    FN = tf.math.count_nonzero((predichos - 1) * originales)
    #print(TP, TN, FP, FN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    #print("Precision:", precision)
    #print("recall:", recall)
    #print("f1:", f1)
    return TP, TN, FP, FN

#def mcc(TP, TN, FP, FN):
    
def metricas_sk(originales, predichos):
    print("::::::::::::::::::::::::  METRICAS SK  ::::::::::::::::::::::::::::::")
    precision, recall, f1, _ = score(originales, predichos)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(f1))

def keras_calculate_mcc_from_conf(confusion_m):
    """tensor version of MCC calculation from confusion matrix"""
    # as in Gorodkin (2004)
    N = K.sum(confusion_m)
    up = N * tf.linalg.trace(confusion_m) - K.sum(tf.matmul(confusion_m, confusion_m))
    down_left = K.sqrt(N ** 2 - K.sum(tf.matmul(confusion_m, K.transpose(confusion_m))))
    down_right = K.sqrt(N ** 2 - K.sum(tf.matmul(K.transpose(confusion_m), confusion_m)))
    mcc_val = up / (down_left * down_right + K.epsilon())
    return mcc_val

def graficar_clases(bien, mal, conjunto):
    plt.figure(figsize=(17, 10))
    sns.set_context("paper",font_scale=2, rc={"font.size":20,"axes.titlesize":25,"axes.labelsize":20})
    intervalos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
    
    factor=1/(len(bien)+len(mal))
    p_bien=len(bien)*factor
    p_bien=p_bien*100
    p_bien=round(p_bien,2)
    label_bien='Clases bien:'+str(p_bien)
    label_bien="{}%".format(label_bien)

    p_mal=len(mal)*factor
    p_mal=p_mal*100
    p_mal=round(p_mal,2)
    label_mal='Clases mal:'+str(p_mal)
    label_mal="{}%".format(label_mal)

    corrects = np.array(bien)*factor
    incorrects = np.array(mal)*factor
    plt.hist([bien,mal], weights=[corrects,incorrects], label=[label_bien,label_mal], bins=intervalos, alpha = 0.9, histtype= 'bar', color=['#79DF4C', '#EF3934'])
    plt.legend()
    #plt.ylim(0,1)
    plt.title(pre+longitud+'-clases bien/mal --> dataset:'+conjunto+ '- #clases:'+ str(len(bien)+len(mal)))
    plt.xlabel('Rangos de porcentajes mayoritarios')
    plt.ylabel('Frecuencia normalizada')
    plt.xticks(intervalos)
    plt.savefig(path+prefijo_modelo+'_'+data_test__+'_clases_bien_mal'+'.png',dpi=300)
    #plt.show()
def hacer_prediccion(kmer, model):

    kmer = np.expand_dims(kmer, axis=(0,-1)) #agregar dimencion al inicio y al final
    #print(kmer.shape)
    #print(kmer.ndim)
    #salida=model.predict(kmer)
    salida=model(kmer)

    salida_f = salida.numpy()
    max_value=salida_f.max()
    max_value=round(max_value,4)
    salida=np.rint(salida)
   
    #print("Salida_int----->",salida)
    #clase_predicha=np.argmax(salida, axis=0)
    clase_predicha=np.argmax(salida)
    #print("Salida----->",clase_predicha)
    return clase_predicha, max_value

def graficar_kmers_bien_mal(bien, mal, conjunto):
    print("Graficando kmers bien y mal")
    plt.figure(figsize=(17, 10))
    sns.set_context("paper",font_scale=2, rc={"font.size":20,"axes.titlesize":25,"axes.labelsize":20})
    #plt.style.use('whitegrid')
    sns.set_style("whitegrid")
    intervalos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
    #intervalos = [0.05,0.10,0.15, 0.20,0.25, 0.30,0.35, 0.40,0.45, 0.50,0.55, 0.60,0.65, 0.70,0.75, 0.80,0.85, 0.90,0.95,1]
    factor=1/(len(bien)+len(mal))
    p_bien=len(bien)*factor
    p_bien=p_bien*100
    p_bien=round(p_bien,2)
    label_bien='Clases bien:'+str(p_bien)
    label_bien="{}%".format(label_bien)

    p_mal=len(mal)*factor
    p_mal=p_mal*100
    p_mal=round(p_mal,2)
    label_mal='Clases mal:'+str(p_mal)
    label_mal="{}%".format(label_mal)

    corrects = np.array(bien)*factor
    incorrects = np.array(mal)*factor
    #counts= plt.hist([mal, bien], weights=[incorrects, corrects], label=[label_mal, label_bien], bins=intervalos, alpha = 0.9, histtype= 'bar', color=['#EF3934', '#79DF4C' ])
    counts_b=np.histogram(bien, weights=corrects, bins=intervalos)
    counts_m=np.histogram(mal, weights=incorrects, bins=intervalos)

    #print(counts_b[0])
    #print(counts_m[0])
    eje_x=[ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    eje_y_b=counts_b[0]*100
    eje_y_m=counts_m[0]*100

    plt.plot(eje_x,eje_y_b, color='#7AA500', marker='o' , markerfacecolor='#7AA500', markersize=8, label=label_bien)
    plt.plot(eje_x,eje_y_m , color='#EF3934', marker='o' , markerfacecolor='red', markersize=8, label=label_mal)
    for a,b in zip(eje_x, eje_y_b): 
        plt.text(a-0.01, b+1, str(round(b,2)) ,fontsize = 15, color='#496300')

    for a,b in zip(eje_x, eje_y_m): 
        plt.text(a-0.01, b-1.5, str(round(b,2)), fontsize = 15, color='#EF3934')

    plt.legend()
    #plt.ylim(0,1)
    plt.title(pre+longitud+'-kmers bien/mal --> dataset:'+data_test__+'--> # kmers :'+ str(conjunto))
    plt.xlabel('Rangos de probabilidades')
    plt.ylabel('Frecuencia normalizada')
    plt.xticks(intervalos)
    plt.savefig(path+prefijo_modelo+'_'+data_test__+'_kmers_bien_mal'+'.png',dpi=300)
    #plt.show()

def dividir_kmers_bien_mal(y_original, y_predicha, certezas):
    print("total kmers:", len(y_original))
    lista_kmers_bien_o=[]
    lista_kmers_bien_p=[]
    lista_kmers_mal_o=[]
    lista_kmers_mal_p=[]
    lista_kmers_certezas_bien=[]
    lista_kmers_certezas_mal=[]

    index_bien=[]
    index_mal=[]
    for i in range(len(y_original)):
        y_o=y_original[i]
        y_p=y_predicha[i]
        valor_certeza=certezas[i]
        if y_o == y_p:
            index_bien.append(i)
            lista_kmers_bien_o.append(y_o)
            lista_kmers_bien_p.append(y_p)
            lista_kmers_certezas_bien.append(valor_certeza)

        else:
            index_mal.append(i)
            lista_kmers_mal_o.append(y_o)
            lista_kmers_mal_p.append(y_p)
            lista_kmers_certezas_mal.append(valor_certeza)

    #kmers que superan un umbral
    count=0
    for i in range(len(lista_kmers_certezas_bien)):
        if lista_kmers_certezas_bien[i] > 0.79:
            count=count+1
    
    print(" ***** Kmers ****")
    print("Total de kmers predichos bien:", len(lista_kmers_bien_p))
    print("Total de kmers predichos bien > 0.79 certeza:", count)
    print(" ******** --- ********")

    kmers_mal=pd.DataFrame()
    kmers_mal[0]=index_mal
    kmers_mal[1]=lista_kmers_mal_o
    kmers_mal[2]=lista_kmers_mal_p
    kmers_mal[3]=lista_kmers_certezas_mal
    kmers_mal.to_csv(path+prefijo_modelo+'_'+data_test__+'_kmers_mal'+'.csv', header= None, index=False) 
    
    kmers_bien=pd.DataFrame()
    kmers_bien[0]=index_bien
    kmers_bien[1]=lista_kmers_bien_o
    kmers_bien[2]=lista_kmers_bien_p
    kmers_bien[3]=lista_kmers_certezas_bien
    kmers_bien.to_csv(path+prefijo_modelo+'_'+data_test__+'_kmers_bien'+'.csv', header= None, index=False) 

    factor_kmers_bien_mal= 100/len(y_original)
    porcentaje_kmers_bien= round(factor_kmers_bien_mal*(len(index_bien)), 3)
    porcentaje_kmers_bien=porcentaje_kmers_bien*100
    porcentaje_kmers_mal= round(factor_kmers_bien_mal*(len(index_mal)), 3)
    porcentaje_kmers_mal=porcentaje_kmers_mal*100
    print("Porcentaje de kmers bien:" , porcentaje_kmers_bien )
    print("Porcentaje de kmers mal:" , porcentaje_kmers_mal )
    
    graficar_kmers_bien_mal(lista_kmers_certezas_bien,lista_kmers_certezas_mal, len(y_original) )


#----------------------------------------------------

instanteInicial = datetime.now()

path = sys.argv[1] #D:/UAEM/BIO/Dataset/JULIO_2022/eucariontes30/datos_kmer/100_10/
pre = sys.argv[2] #pre='euk30'
longitud = sys.argv[3] #100_10
path_modelo = sys.argv[4] #D:/modelo4.h5
data_test = sys.argv[5] # name datates
split_test= int(sys.argv[6])
data_test_=data_test.split(".")
data_test__=data_test_[0]
modelo_=path_modelo.split(".")
print(modelo_)
modelo__=modelo_[0]
print(modelo__)
modelo___=modelo__.split("/")
prefijo_modelo=modelo___[-1]

name_test=data_test__+'.csv'
name_clases='clases_'+pre+'_'+longitud+'.csv'
print("Analisis modelo:", prefijo_modelo+'.h5')
print("PATH test:",path+name_test)
print("PATH salidas:",path)

print("::::::::::::::::::::::::  CARGAR DATOS  ::::::::::::::::::::::::::::::")
X_test, num_X_test= get_datos(path, name_test)
clases,num_clases = get_datos(path, name_clases)
print("tamano del set test:", num_X_test)
print("NÃºmero de clases:", num_clases)

print(":::::::::::::::::::::::::  CARGAR MODELO  :::::::::::::::::::::::::::::")

print("::::::::::::::::::::::::  Predict ::::::::::::::::::::::::::::::")
print("Total de datos Test ----->", len(X_test))

total_test=len(X_test)
chunks_test=total_test/split_test
validar_chunk=total_test % split_test
print("Numero de Chunks ----->", split_test)
print("Registros X Chunks ----->", chunks_test)
print("Modulo ----->", validar_chunk)

if (validar_chunk == 1):
    print("No puedes dividir el test en:", split_test, "Chunks Por que genera chunks flotantes")
    sys.exit(0)  # Terminar el programa con código de salida 0 (éxito)
#Pasamos a int el numero de chunks
chunks_test=int(chunks_test)
#Declaramos las listas vacias donde se guardaran las predicciones y clases
lista_class_out=[]
lista_class_out_f=[]
y_original=[]

#Cargamos el modelo
model = load_model(path_modelo)

test_generador=generador(X_test, chunks_test, num_clases)

#Ciclo dividido en X numero de chunks
for i in range (split_test):
    print("Prediccion proceso-->", i+1 )

    #test_generador=generador(X_test, len(X_test), num_clases)
    prueba=next(test_generador)
    entrada=prueba[0]
    y_original_=prueba[1]

    y_original_=np.array(y_original_, dtype='uint8')
    print(y_original_.dtype)
    print(entrada.dtype)
    entrada=np.array(entrada, dtype='uint8')


    #print(entrada)
    print("Salidas Originales:")
    print(type(y_original_))
    print(y_original_.shape)
    print(y_original_)
    y_original_=np.argmax(y_original_, axis=1)
    print(y_original_)
    print(type(y_original_))
    print(y_original_.shape)
    print("-------")


    
    bar = Bar('Loading', max=len(entrada) ,suffix='%(percent)d%%')
    for i in range(len(entrada)):
        clase_predicha, clase_predicha_f=hacer_prediccion(entrada[i], model)
        lista_class_out.append(clase_predicha)
        lista_class_out_f.append(clase_predicha_f)   
        y_original.append(y_original_[i]) 
        bar.next()#Con esto podemos el progreso del ciclo
    bar.finish() #Con este finalizamos la barra de progreso

    #Libreamos memoria
    del prueba
    del entrada
    del y_original_





print("Tamano de lista prediccion:",len(lista_class_out), len(lista_class_out_f), len(y_original))


print("ORIGINALES CLASS:",y_original)
#print("PREDICCION CLASS:",lista_class_out)
#print("Certeza de prediccion:",lista_class_out_f)

salida=pd.DataFrame()
salida['salidas_originales']=y_original
salida['salidas_predichas']=lista_class_out
salida['certeza']=lista_class_out_f
salida.to_csv(path+prefijo_modelo+'_'+data_test__+'_OUTPUTS_'+'.csv', header= True, index=False)

print("::::::::::::::    Analisis kmers    :::::::::::::::::::")
dividir_kmers_bien_mal(y_original, lista_class_out, lista_class_out_f  )

print("::::::::::::::::::::::::  Metricas ::::::::::::::::::::::::::::::")
#metricas_sk(y_original, y_class_predict)
#metricas_sk(y_original, lista_class_out)
#metricas(y_original, y_class_predict)

# Confusion matrix
print("MATRIX de CONFUSION")
label_rango=(len(clases))
labels=[]
for i in range(0,label_rango):
    labels.append(str(i))
#print(labels)


# Report
print("::::::::::::::: REPORTE:::::::::::")
print("Modelo:", prefijo_modelo, "Total de Clases:", num_clases)
reporte=classification_report(y_original, lista_class_out, target_names=labels, output_dict=True)
df_reporte = pd.DataFrame(reporte).transpose()
df_reporte_r=df_reporte.round(2)
df_reporte_r.to_csv(path+prefijo_modelo+'_'+data_test__+'_metricas_por_clases'+'.csv')

print("::::::::::::::: MCC:::::::::::")
print("Matthews correlation coefficient:",matthews_corrcoef(y_original, lista_class_out))

#cf = tf.math.confusion_matrix(y_original, y_class_predict)
cf=confusion_matrix(y_original, lista_class_out,normalize='true')
cf= cf.round(3)


matrix_confusion=pd.DataFrame(cf)
matrix_confusion.to_csv(path+prefijo_modelo+'_'+data_test__+'_matrix_confusion'+'.csv')

max_mc_clase=matrix_confusion.idxmax(axis=1)
max_mc_por=matrix_confusion.max(axis=1)
del matrix_confusion
del y_original
del lista_class_out
del lista_class_out_f
del salida
del df_reporte
del df_reporte_r
outs_promediada=pd.DataFrame()
outs_promediada[0]=labels
outs_promediada[1]=max_mc_clase
outs_promediada[2]=max_mc_por
outs_promediada.to_csv(path+prefijo_modelo+'_'+data_test__+'_out_avg_class'+'.csv', header= None, index=False) 
del outs_promediada

lista_clases_bien=[]
lista_clases_bien_index=[]
lista_clases_bien_p=[]
lista_clases_mal=[]
lista_clases_mal_index=[]
lista_clases_mal_p=[]
for i in range(len(labels)):
    if max_mc_clase[i]== i:
        lista_clases_bien_index.append(i)
        lista_clases_bien.append(max_mc_clase[i])
        lista_clases_bien_p.append(max_mc_por[i])
        #print("BIEN")
    else:
        #print("MAL")
        lista_clases_mal_index.append(i)
        lista_clases_mal.append(max_mc_clase[i])
        lista_clases_mal_p.append(max_mc_por[i])


clases_bien=pd.DataFrame()
clases_bien[0]=lista_clases_bien_index
clases_bien[1]=lista_clases_bien
clases_bien[2]=lista_clases_bien_p
clases_bien.to_csv(path+prefijo_modelo+'_'+data_test__+'_clases_bien'+'.csv', header= None, index=False)
del clases_bien


clases_mal=pd.DataFrame()
clases_mal[0]=lista_clases_mal_index
clases_mal[1]=lista_clases_mal
clases_mal[2]=lista_clases_mal_p
clases_mal.to_csv(path+prefijo_modelo+'_'+data_test__+'_clases_mal'+'.csv', header= None, index=False) 

graficar_clases(lista_clases_bien_p,lista_clases_mal_p , data_test__)

instanteFinal = datetime.now()
tiempo = instanteFinal - instanteInicial # Devuelve un objeto timedelta
#segundos = tiempo.seconds
print("El programa se ejecuto en --> "+str(tiempo.seconds)+'segundos')
