import numpy as np
import tensorflow as tf 
from tensorflow import keras

#Definir la arquitectura de la red neuronal
modelo = keras.Sequential([ 
    keras.layers.Dense(units=1, activation='sigmoid', input_shape=(2,)) 
]) 
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Definir los datos de entrenamiento
X_entrenar = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
Y_entrenar = np.array([[0], [0], [0], [1]])

#Entrenar el Modelo
modelo.fit(X_entrenar, Y_entrenar, epochs=5000, verbose=0)

#Obtener los datos de entrada del usuario y validar los valores
entrada_valida = False 
while not entrada_valida: 
    entrada_a = int(input("Digite el primer número (debe ser 0 o 1): ")) 
    entrada_b = int(input("Digite el segundo número (debe ser 0 o 1): ")) 
    
    if entrada_a in [0, 1] and entrada_b in [0, 1]: 
        entrada_valida = True 
    else: 
        print("Los números deben ser 0 o 1. Inténtalo de nuevo.\n")

#Convertir los datos de entrada en un arreglo numpy y predecir la salida
X = np.array([[entrada_a, entrada_b]]) 
prediccion = modelo.predict(X)

#Mostrar el resultado
print("Salida predicha:") 
print(np.round(prediccion))