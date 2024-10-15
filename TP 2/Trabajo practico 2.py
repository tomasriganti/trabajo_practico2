import numpy as np
from PIL import Image

#el usuario ingresa la foto 

#path = open("C:/Users/joaco/OneDrive/Desktop/UDESA/primer AÑO 1er CUATRI/Pensamiento Computacional/TPS/TP 2/castle.jpg", mode = 'r')

path = 'C:/Users/joaco/OneDrive/Desktop/UDESA/primer AÑO 1er CUATRI/Pensamiento Computacional/TPS/TP 2/castle.jpg'

#esta funcion devuelve la foto en forma de matriz con las 3 dimensiones (RGB)
def generar_matriz_rgb(path) -> np.array:
    matriz = np.array(Image.open(path))
    return matriz

np.set_printoptions(threshold=np.inf)

matriz_RGB = generar_matriz_rgb(path)

#medidas de la matriz_1
print(f'matriz: {matriz_RGB.shape}')  
          

#divido en 3 matrices, en RGB
matriz_rojo = matriz_RGB[ :, :, 0]
matriz_verde = matriz_RGB[ :, :, 1]
matriz_azul = matriz_RGB[ :, :, 2]

print(f'MATRIZ RED: {matriz_rojo.shape}')
print(f'MATRIZ GREEN: {matriz_verde.shape}')
print(f'MATRIZ BLUE: {matriz_azul.shape}')


#-----------------------------------------------------------------------------------------------------------------------------------------------------------

def matriz_con_padding(matriz: np.array, pad_width: int = 1) -> np.array:
    """
    Esta función aplica padding a la matriz usando el modo 'edge',
    replicando los bordes de la imagen original.
    
    :param matriz: La matriz a la que se le agregará padding.
    :param pad_width: El tamaño del padding a agregar.
    :return: La matriz con padding agregado.
    """
    matriz_padded = np.pad(matriz, pad_width=pad_width, mode='edge')
    return matriz_padded



# Agregar padding a las matrices de color
matriz_rojo_padded = matriz_con_padding(matriz_rojo)
matriz_verde_padded = matriz_con_padding(matriz_verde)
matriz_azul_padded = matriz_con_padding(matriz_azul)

# Verificar las dimensiones de las matrices con padding
print(f'MATRIZ RED con padding: {matriz_rojo_padded.shape}')
print(f'MATRIZ GREEN con padding: {matriz_verde_padded.shape}')
print(f'MATRIZ BLUE con padding: {matriz_azul_padded.shape}')
