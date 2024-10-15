import numpy as np
from PIL import Image

#el usuario ingresa la foto 

#path = open("C:/Users/joaco/OneDrive/Desktop/UDESA/primer AÑO 1er CUATRI/Pensamiento Computacional/TPS/TP 2/castle.jpg", mode = 'r')

path = 'C:/Users/joaco/OneDrive/Desktop/UDESA/primer AÑO 1er CUATRI/Pensamiento Computacional/TPS/TP 2/castle.jpg'

#esta funcion devuelve la foto en forma de matriz en blanco y negro
def generar_matriz_en_gris(path):
    matriz_gris = np.array(Image.open(path).convert("L"))
    return matriz_gris


np.set_printoptions(threshold=np.inf)


print(generar_matriz_en_gris(path))

matriz_1 = generar_matriz_en_gris(path)


#filtro horizontal
def sobel_horizontal(matriz_1):
    sobel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])

    matriz_sobel_x = 1
    

    return matriz_sobel_x




#filtro vertical
def sobel_vertical(matriz_1):
    sobel_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

    matriz_sobel_y = 1

    return matriz_sobel_y
