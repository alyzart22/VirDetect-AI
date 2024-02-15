import os
import sys #para pasar parametro
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import time
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import tensorflow as tf 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
# Memory growth must be set before GPUs have been initialized
		print(e)

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

#import tensorflow as tf
#gpus = tf.config.list_physical_devices(device_type = 'GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

'''
from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
'''
  
print(":::::::::::::::::::::::::::::::::: CHECK GPU ::::::::::::::::::::::::::::::::::::")
print("**** Version de TensorfLow ****",tf.__version__)
#gpu esta instalada?
print(tf.test.is_gpu_available())
print(tf.test.gpu_device_name())

#tensor gpu test
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#tensor check gpu de tensor2
print("Num GPUs Available_: ", len(tf.config.list_physical_devices('GPU')))

#Python checa mi gpu
from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if tf.test.gpu_device_name(): 
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")
print(":::::::::::::::::::::::::::::::::: FIN CHECK GPU ::::::::::::::::::::::::::::::::::::")


from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import L2, L1
from tensorflow.keras.layers import Input, Dropout, Activation,Conv1D, Add, MaxPooling1D, BatchNormalization, Conv2D, Add, MaxPooling2D, ReLU, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from datetime import datetime

import wandb
from wandb.keras import WandbCallback

import random #USado para shuffle batch registros
random.seed(10) #Inicializamos la semilla

print("::::::::::::::::: Fin check librerias:::::::::::::")

#-------------- Metodos-------------------------------
#Metodo que se le pasa la directiva de los splits 
def get_datos(ruta, nombre_archivo):
    #print(":::::::: get_datos :::::::::::::::::::")
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

#metodo que leer el archivo y lo guarda en una lista
def agregar_a_muestra(csv_filepath, muestras):
    #print(":::::::: agregar_a_muestra :::::::::::::::::::")
    with open(csv_filepath) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            muestras.append(line)
    return muestras

#Hacer generador
def generador(muestra, tamano_batch, num_clases):
    total_de_muestras=len(muestra)
    #Esta linea la agrege por que queria q los datos del batch fueran en orden random 31/oct/22
    #muestra_r=random.sample(muestra, len(muestra))
    print("Generador::::")
    #print(type(muestra))
    #print(len(muestra))
    #print(type(muestra_r))
    print(len(muestra))
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
            x_data=np.array(lista_datos, dtype='uint8')
            #print(x_data)
            
            x_data=to_categorical(x_data,26, dtype='uint8')
            #print(x_data)
            #ETIQUETAS
            df_batch_etiquetas=pd.DataFrame(np.array(lista_etiquetas), index=None)
            y_data=df_batch_etiquetas.iloc[:,0:1]
            y_data=np.array(y_data, dtype='uint16')#Esto lo puse pa ver si mejoraba el error

            y_data=to_categorical(y_data, num_classes=num_clases, dtype='uint16')
            #y_data=to_categorical(y_data, num_classes=num_clases)
            #bin_nums = ((nums.reshape(-1,1) & (2**np.arange(5))) != 0).astype(int)

            #print(":::::::::::SHAPE")

            # :::ERROR SOLVED::: PAsar a tensores por q si lo dejaba como numpy.ndarray saltaba error Esto en los nuevos entornor de sep2023 en aleph y ibt ya no fue necesario modificar a tensores
            #x_data = tf.convert_to_tensor(x_data, dtype=tf.int8)
            #y_data = tf.convert_to_tensor(y_data, dtype=tf.int8) #float32

            #print("*************SHAPE ")
            #print(x_data.shape)
            
            #Generamos un generador
            #print(y_data.shape)
            #print(type(y_data))
            #print(y_data)
            yield x_data,y_data

def recorre_lineas(lista_lineas):
    #Almacena la columna donde estan los kmers
    #lineas_kmeros=lista_lineas[1]
    lineas_kmeros=lista_lineas[2]
    #Almacena la columna donde estan las etiquetas
    #lineas_etiquetas=lista_lineas[0:1]
    lineas_etiquetas=lista_lineas[0] # por q no es la clase en si si no el codigo en orden 0-nclases
    #Convertir a letras el conjunto de lineas
    kmers_convertidos=conv_letra_num(lineas_kmeros)
    return kmers_convertidos, lineas_etiquetas

#Metodo para pasar de letras a numerico
def conv_letra_num(secuencia):
    #print(secuencia)
    Lette_dict = {'X': 0,'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,'Z': 21,'U':22, 'B':23, 'O':24,'J':25}
    seq_list = list(secuencia)
    seq_list = [Lette_dict[base] for base in seq_list]
    return seq_list

def tensor_to_image(tensor): #Para ver la imagen generada del generador
    #tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    print(tensor)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    print(tensor)
    return tensor
    cv2.imwrite(directorio+'imagen_proceso.png',tensor)

def residual_block(data, filters, kernel_tamano): #, d_rate
    shortcut = data
    bn1 = BatchNormalization(axis=-1)(data)
    act1 = Activation('relu')(bn1)
    #conv1 = Conv1D(filters, 1, dilation_rate=d_rate, padding='same', kernel_regularizer=regularizers.l2(0.0001))(act1)
    conv1 = Conv2D(filters,(kernel_tamano,kernel_tamano), strides=(1,1),padding='same', activation='relu')(act1) #,name='CONV1_RES' #same = agregar pading , strides sancadas de la conv bias_initializer='zeros',
    #bottleneck convolution
    bn2 = BatchNormalization(axis=-1)(conv1) 
    act2 = Activation('relu')(bn2)  
    #conv2 = Conv1D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(0.0001))(act2)
    conv2 = Conv2D(filters,(kernel_tamano,kernel_tamano), strides=(1,1),padding='same', activation='relu')(act2)#,name='CONV2_RES'
    #skip connection
    x = Add()([conv2, shortcut])
    return x


def modelo_ali_RES(input_shape_,clases_total, numero_filtros, kernel_tamano):
    print("MODELO: RESNET_300fc_3res_128filt_5x5",input_shape_)
    input_x = Input(shape=input_shape_)
    Z1 = Conv2D(numero_filtros,(kernel_tamano,kernel_tamano), strides=(1,1),padding='same',name='CONV1', activation='relu')(input_x) #same = agregar pading , strides sancadas de la conv bias_initializer='zeros',
    
    # bloques residuales
    res1 = residual_block(Z1, numero_filtros, kernel_tamano) #datos, filtros 128
    res2 = residual_block(res1, numero_filtros, kernel_tamano) #datos, filtros 128
    res3 = residual_block(res2, numero_filtros, kernel_tamano) #datos, filtros 128
    P1 = MaxPool2D(pool_size=(2,2), strides= 2, name='POOL1')(res3)
 ## FLATTEN
    F1 = Flatten(name='FLAT')(P1)
    ## Dense layer
    FC1 =Dense(200, activation='relu',   name='fc_1')(F1) #bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.0001),
    FC2= Dense(100, activation='relu', name='fc_2')(FC1) #bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.0001),
    #FC3= Dense(30, activation='relu', name='fc_3')(FC2)#bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.0001),
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    outputs_y = Dense(clases_total, activation='softmax',  name='fc_output')(FC2) #kernel_regularizer=regularizers.l2(0.0001), 
    modelo_ = Model(inputs=input_x, outputs=outputs_y)
    return modelo_   

# Funcion que grafica loss y accuracy
def plot_history(history,longitud, batch, ler_rate, grafica_save):
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy'+"_model:"+longitud+"_batch:"+str(batch)+ "_lr:"+str(ler_rate))
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss'+"_model:"+longitud+"_batch:"+str(batch)+ "_lr:"+str(ler_rate))
    plt.legend()
    plt.savefig(grafica_save, dpi=300)
    #plt.show()
#Funcion para guardar modelo
def guardar_modelo(path, nombre):
    model.save(path+nombre)
#Guardar modelo h5
#def guardar_modelo(path, nombre):
 #   model.save(path+nombre)
    
#-----------------------------------------------------
#Datos Entrada:
#python D:\ALI\modelos\modelo.py D:/ALI/modelos/ pro_100_10 128 0.001 8
#python D:\ALI\modelos\modelo.py pro_100_10 128 0.001 8

#Parametros pasados
#ruta = sys.argv[1]
longitud = sys.argv[1]#pro30_100_10
tamano_batch = sys.argv[2]#512
tamano_batch=int(tamano_batch)
learning_rate_ = sys.argv[3]#0.001
learning_rate_=float(learning_rate_)
epocas = sys.argv[4]#8
epocas = int(epocas) #BASIC, BN, BN_DO
tipo=sys.argv[5]#BASIC, BN, BN_DO
numero_filtros=sys.argv[6]
numero_filtros=int(numero_filtros) #128
kernel_tamano=sys.argv[7]
kernel_tamano=int(kernel_tamano)
longitud_kmer=int(sys.argv[8]) #100,300,500
path=sys.argv[9]+longitud+'/' #D:/ALI/modelos/pro30/
pc=sys.argv[10] # ALEPH, legion, mau, miztli plataforma donde se entrea 


print("longi_carpeta",longitud,"batch",tamano_batch,"lr" ,learning_rate_, "epochs", epocas, "modelo:",tipo,"Filtros:", numero_filtros, "kernel_size:",kernel_tamano,"x",kernel_tamano,"version_res3_fc_200_100", "pc:",pc)

#Correr
#python C:\Users\Ali\Documents\modelo_choose_cnn_batch_shuffle.py pro30_100_10 1024 0.001 16 BASIC D:/ALI/modelos/pro30/

#longitud='100_10'
#longitud='muestra_pro_100_10'
#path=ruta+longitud+'/'  
#path='D:/ALI/modelos/pro30/'+longitud+'/' 
#path = '/content/drive/MyDrive/Colab Notebooks/Modelos_Tesis_Ali_2022/datos_dummi_kmers/'+longitud+'/'
#path_save_logs= "/content/drive/MyDrive/VIRUS/out/"
name_train='train'+'_'+longitud+'.csv'
name_val='val'+'_'+longitud+'.csv'
name_test='test'+'_'+longitud+'.csv'
name_clases='clases'+'_'+longitud+'.csv'
logs_call="logs"+'_'+longitud
#Datos de la red
#------------------------
'''tamano_batch=16 #2,4,8,16,32,64,128,256 #64 0.01 32 0.001
learning_rate_=0.001
epocas=8'''
#------------------------
fecha=datetime.today().strftime('%Y-%m-%d-%H%M')
modelo_version=longitud+'_'+str(tamano_batch)+'_'+str(learning_rate_)+'_Ep_'+str(epocas)+'_'+tipo+'_'+fecha+'_NumFilt_'+str(numero_filtros)+'_KerSiz_'+str(kernel_tamano)+'x'+str(kernel_tamano)+'_res3_fc_200_100_'+pc+'.h5'
modelo_version_v2=longitud+'_'+str(tamano_batch)+'_'+str(learning_rate_)+'_Ep___'+str(epocas)+'_'+tipo+'_'+fecha+'_NumFilt_'+str(numero_filtros)+'_KerSiz_'+str(kernel_tamano)+'x'+str(kernel_tamano)+'_res3_fc_200_100_'+pc+'.h5'
#fecha=datetime.now().strftime("%Y%m%d-%H%M%S")
#fecha=datetime.today().strftime('%Y-%m-%d-%H%M')
#estructura_modelo= path_save_logs +"Modelo_"+longitud+'_'+ fecha +'_'+str(tamano_batch)+'_'+str(learning_rate_)+".png"
#grafica_metricas= path_save_logs +"Metrica_"+longitud+'_'+ fecha +'_'+str(tamano_batch)+'_'+str(learning_rate_)+".png"
estructura_modelo= path +"Modelo_"+longitud+'_'+str(tamano_batch)+'_'+str(learning_rate_)+'_Ep_'+str(epocas)+'_'+tipo+'_'+'_NumFilt_'+str(numero_filtros)+'_KerSiz_'+str(kernel_tamano)+'x'+str(kernel_tamano)+'_res3_fc_200_100_'+fecha+"_"+pc+".png"
grafica_metricas= path +"Metrica_"+longitud+'_'+str(tamano_batch)+'_'+str(learning_rate_)+'_Ep_'+str(epocas)+'_'+tipo+'_'+'_NumFilt_'+str(numero_filtros)+'_KerSiz_'+str(kernel_tamano)+'x'+str(kernel_tamano)+'_res3_fc_200_100_'+fecha+"_"+pc+".png"

#Cosa para usar WandB, inicializamos

wandb.init(project=logs_call+'-WandB', name=tipo+'_'+longitud+'_'+pc+'_'+ fecha +'_BS_'+str(tamano_batch)+'_Lr_'+str(learning_rate_)+'_Ep_'+str(epocas)+'_NumFilt_'+str(numero_filtros)+'_KerSiz_'+str(kernel_tamano)+'x'+str(kernel_tamano)+'_res3_fc_200_100', tags=pc)


#-----------------------------------------------------
print("::::::::::::::::::::::::  CARGAR DATOS ::::::::::::::::::::::::::::::")
#Pasamos a lista los datos de los splits guardados en local
X_train,num_X_train = get_datos(path, name_train)
X_val,num_X_val = get_datos(path, name_val)
clases,num_clases = get_datos(path, name_clases)
#X_test, num_X_test= get_datos(path, name_test)

steps_train=num_X_train // tamano_batch
steps_val=num_X_val // tamano_batch

print("tamano del set train:", num_X_train)
print("tamano del set val:", num_X_val)
print("Número de clases:", num_clases)
print("tamano de batch:", tamano_batch)
print("steps_train:", steps_train)
print("steps_val:", steps_val)
print("epocas", epocas)
print("Lr", learning_rate_)
print("Longitud", longitud)
print("tipo", tipo)
print("numero filtros:", numero_filtros)
print("kernel_size:", kernel_tamano)
print("Longitud del kmer:", longitud_kmer)




#-----------------------------------------------------
#Hacer generadores
train_generador= generador(X_train,tamano_batch, num_clases)
print(type(train_generador))
val_generador= generador(X_val,tamano_batch, num_clases)
print(type(val_generador))

#-----------------------------------------------------


#Estructura de la red
#Escoger que version de Red quieres correr 

if tipo == 'RES':
    model=modelo_ali_RES((longitud_kmer, 26, 1), num_clases, numero_filtros, kernel_tamano)

#model=modelo_ali((100, 26, 1), num_clases)
#https://keras.io/api/optimizers/
opt_sgd = SGD(learning_rate=learning_rate_, name="SGD")
opt_RMS = RMSprop(learning_rate=learning_rate_, name="RMSprop")
opt_Adam = Adam(learning_rate=learning_rate_, name="Adam")
opt_Adadelta = Adadelta(learning_rate=learning_rate_, name="Adadelta")
opt_Adamax = Adamax(learning_rate=learning_rate_, name="Adamax")
opt_Nadam = Nadam(learning_rate=learning_rate_, name="Adamax")



model.compile(optimizer=opt_Adam ,loss='categorical_crossentropy', metrics=['accuracy'])


model.summary()


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( filepath=path+modelo_version_v2, save_weights_only=False,monitor='val_accuracy',mode='max',save_best_only=True)

#-----------------------------------------------------

print("::::::::::::::::::::::::  Training ::::::::::::::::::::::::::::::")
tic = time.perf_counter()
print("Tiempo inicio:", tic)
history = model.fit_generator(train_generador,epochs=epocas,steps_per_epoch=steps_train,validation_data=val_generador,validation_steps=steps_val,verbose=1, callbacks=[model_checkpoint_callback, WandbCallback(save_weights_only=True)] ,initial_epoch=0, shuffle=True)
toc = time.perf_counter()
print("Tiempo Final:", toc)
print(f"Tiempo transcurrido ) {toc - tic:5.4f} seconds")

print("::::::::::::::::::::::::  Saving Model ::::::::::::::::::::::::::::::")
guardar_modelo(path, modelo_version)
print(history.history.keys())
print(history.history)
plot_history(history, longitud, tamano_batch, learning_rate_, grafica_metricas) 
