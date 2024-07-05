import os
import numpy as np # type: ignore
import scipy.io
from scipy.io import loadmat # type: ignore
import tensorflow as tf # Para red neuronal profunda
import numpy as np
import matplotlib.pyplot as plt
import time # Para tomar el tiempo de entrenamiento de la red
import math

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#CONVERTIR MATRICES DE: .mat  --> .npyDirectorios de entrada y salida: SUJETOS SANOS Y PACIENTES TEC

#(1) sujetos sanos: 1_HEMU - 2_DAOC - 3_DASI - 4_DABA - 5_HEFU - 6_JOBO - 7_ROMI - 8_FEGA - 9_GAGO - 10_MIMO - 11_JULE - 12_NIGA - 13_BYLA - 14_ARVA
#  - 15_CLSE - 16_PAAR - 17_VATO - 18_FEBE - 19_VINA - 20_CLHE - 21_MAIN - 22_ALSA - 23_MIRA - 24_LACA - 25_GOAC - 26_ANGL - 27_HC036101

#(2) pacientes tec: 1_DENI1005 - 2_KNOW1001 - 3_ALI0 - 4_BUTL - 5_HAGG - 6_HASTI007 - 7_BOAM - 8_DANE0005 - 9_GREG - 10_AITK - 11_RANS0000 - 12_JONES004 - 13_PERR - 14_SLAC
#  - 15_HEPPL010 - 16_RICHS010 - 17_KENT0007 - 18_STAN1002 - 19_MCDON022 - 20_PULL - 21_MORR1002 - 22_PARK - 23_HIGH - 24_NOBL - 25_COWL - 26_KHAN - 27_NOLA

# Directorios de las matrices complejas (coeficientes) de las senales PAM ,VSCd y VSCi de suejtos sanos o pacientes con tec. Se debe modificar estos directorios para ir
# generando los respectivos modelos de cada individuo.

# DIRECTORIOS

# SUJETO SANO O PACIENTE TEC POR ANALIZAR
# DIRECTORIO ejemplo: PACIENTE TEC --> 'TEC/1_DENI1005'
persona = 'SANOS/1_HEMU'
# Sector de la VSC por analizar (derecho o izquierdo):
sector = 'derecho'


#*********************************************************************************************************************************************************************************
#************************************************************************ Red Neuronal Profunda: U-net ***************************************************************************
#*********************************************************************************************************************************************************************************

#Directorios de entrada
# SENAL PAM
input_pam_dir = 'D:/TT/Memoria/onlyred_unet/signals_LDS/' + persona + '/PAMnoises_matrixcomplex_npy_tensor3d'
# VSC LADO DERECHO
output_vscd_dir = 'D:/TT/Memoria/onlyred_unet/signals_LDS/' + persona + '/VSCdnoises_matrixcomplex_npy_tensor3d'
# VSC LADO IZQUIERDO
output_vsci_dir = 'D:/TT/Memoria/onlyred_unet/signals_LDS/' + persona + '/VSCinoises_matrixcomplex_npy_tensor3d'


#Funcion para cargar los archivos .npy
def load_npy_files(input_dir):
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')])
     # verificar orden con que entrar los archivos en X e Y
    file_names = [os.path.basename(f) for f in files]
    print(f"Archivos: {file_names}\n")
    data = [np.load(f) for f in files]
    return np.array(data)

#CARGA DE DATOS DE ENTRADAS Y SALIDAS PARA LA RED (X: INPUTS; Y: OUTPUTS)
# Se identifica que sector del cerebro se desea analizar:
lado = ''
if sector == 'derecho':
    X = load_npy_files(input_pam_dir) # inputs
    Y = load_npy_files(output_vscd_dir) # outputs
    lado = 'vscd'
    print('Modelo para VSC: sector derecho del cerebro.\n')
else:
    X = load_npy_files(input_pam_dir) # inputs
    Y = load_npy_files(output_vsci_dir) # outputs
    lado = 'vsci'
    print('Modelo para VSC: sector izquierdo del cerebro.\n')
print('Abreviacion de sector a estudiar:', lado)


#Verificar las formas de los datos cargados (# entradas, filas, columnas, canales):
print(f"Shape de los inputs (X): {X.shape}")
print(f"Shape de los outputs (Y): {Y.shape}")

#Definir la U-Net
def unet_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Regularizer
    #l2_reg = tf.keras.regularizers.l2(l2_lambda)
    
    # Encoder
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs) #filtro original=64
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1) #filtro original=64
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1) #filtro original=128
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2) #filtro original=128
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2) #filtro original=256
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3) #filtro original=256
    
    # Decoder
    u4 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3) #filtro original=128
    u4 = tf.keras.layers.concatenate([u4, c2])
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u4) #filtro original=128
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4) #filtro original=128
    
    u5 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4) #filtro original=64
    u5 = tf.keras.layers.concatenate([u5, c1])
    c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5) #filtro original=64
    c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5) #filtro original=64
    
    outputs = tf.keras.layers.Conv2D(2, (1, 1), activation='linear')(c5)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model


#Definir la metrica NMSE ajustada para utilizar la varianza de los valores verdaderos
def nmse(y_true, y_pred):
    mse = tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred))
    var_true = tf.keras.backend.var(y_true)
    return mse / var_true

#****************
#HIPERPARAMETROS
#****************
max_epoch = 200
batchsize = 8
learning_rate = 0.0001
#l2_lambda = 0.01
validation_split = 0.2 # 80% entrenamiento & 20% validacion
# alpha: el lr min al que llegara el decaimiento sera el 10% del lr inicia
#alpha = 0.1
# decay steps: Numero de pasos de entrenamiento tras los cuales el learning rate decaera desde su valor inicial hasta el valor final determinado por alpha
#decay_steps = (int(X.shape[0]/batchsize))*max_epoch 
#print("Total pasos de decaimiento ->",decay_steps, "pasos.")


#CREACIÓN DEL MODELO U-NET
input_shape = X.shape[1:]  # forma del input a entrar. en este caso esta forma debe coincidir con las matrices que entran a la red tensor X = [#inputs, columnas, filas, canales]. Se omite #inputs
model = unet_model(input_shape)


#DEFINICION DE ALGORITMO OPTIMIZADOR, FUNCION DE PERDIDA Y METRICA
optimizer = tf.keras.optimizers.Adam(learning_rate)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[nmse])

#########################################################################################################
#########################################################################################################
#########################################################################################################
#ENTRENAMIENTO DE LA RED:
start_time = time.time()
#history = model.fit(X, Y, epochs=max_epoch, batch_size=batchsize, callbacks=[lr_scheduler], validation_split=validation_split)
history = model.fit(X, Y, epochs=max_epoch, batch_size=batchsize, validation_split=validation_split)
end_time = time.time()
total_time = end_time - start_time
min_time = total_time / 60
print(f'Tiempo total de entrenamiento: {min_time:.2f} minutos.')
#########################################################################################################
#########################################################################################################
#########################################################################################################

#Visualizar el NMSE
plt.plot(history.history['nmse'], label='NMSE (entrenamiento)')
plt.plot(history.history['val_nmse'], label='NMSE (validacion)')
plt.xlabel('Epoca')
plt.ylabel('NMSE')
plt.legend()

# Guardar grafica de curvas NMSE

# Directorio y nombre del archivo
output_dir_graphic = 'D:/TT/Memoria/onlyred_unet/signals_LDS/' + persona  # Reemplaza con tu directorio
output_file_graphic =  'unet_model_' + lado + '_' + sector + '_graphic.png'  # Reemplaza con tu nombre de archivo

# Guardar el gráfico
output_path_graphic = f'{output_dir_graphic}/{output_file_graphic}'
plt.savefig(output_path_graphic, format='png')
plt.show()

#GUARDAR UN MODELO ESPECIFICO
# Directorio en donde se almacenara el modelo
save_dir = 'D:/TT/Memoria/onlyred_unet/signals_LDS/' + persona
os.makedirs(save_dir, exist_ok=True)  # Crear el directorio si no existe

# Nombre del archivo del modelo
model_name = 'unet_model_' + lado + '_' + sector + '.keras'

# Ruta completa del archivo
model_path = os.path.join(save_dir, model_name)

# Guardar el modelo entrenado
model.save(model_path)




