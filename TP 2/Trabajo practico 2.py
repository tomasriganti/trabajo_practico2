import numpy as np
from PIL import Image

#el usuario ingresa la foto 

#path = open("C:/Users/joaco/OneDrive/Desktop/UDESA/primer AÑO 1er CUATRI/Pensamiento Computacional/TPS/TP 2/castle.jpg", mode = 'r')

path = '"C:/Users/tomas/ipc/imgs/castle.jpg"'

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

#---------------------------------------------------------------------------------

# Definir los kernels de Sobel
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1, -2, -1]])

def cant_filas_columnas(matriz):
    filas, columnas = matriz.shape 
    return filas, columnas




def apply_sobel(matriz):
    filas, columnas = cant_filas_columnas(matriz)
    
    # Crear una matriz de ceros para almacenar el resultado
    # La matriz de salida es de tamaño (filas-2) x (columnas-2) porque se excluyen los bordes
    sobel_result = np.zeros((filas-2, columnas-2))  
    
    # Iterar sobre cada píxel, excluyendo los bordes debido al padding
    for i in range(1, filas-1):
        for j in range(1, columnas-1):
            # Extraer la submatriz 3x3 alrededor del píxel actual
            submatriz = matriz[i-1:i+2, j-1:j+2]  # Ventana de 3x3
            
            # Aplicar la convolución con los kernels de Sobel
            gx = np.sum(sobel_x * submatriz)  # Convolución con sobel_x
            gy = np.sum(sobel_y * submatriz)  # Convolución con sobel_y
            
            # Calcular la magnitud del gradiente
            gradiente = np.sqrt(gx**2 + gy**2)
            
            # Almacenar el resultado en la matriz sobel_result
            sobel_result[i-1, j-1] = gradiente  # Se guarda en i-1, j-1 porque excluimos los bordes
    
    return sobel_result

# Aplicar Sobel a cada canal
sobel_r = apply_sobel(matriz_rojo_padded)
sobel_g = apply_sobel(matriz_verde_padded)
sobel_b = apply_sobel(matriz_azul_padded)

# Promediar los resultados de los tres canales
matriz_energia = (sobel_r + sobel_g + sobel_b) / 3

# Convertir el resultado a una imagen y mostrarla o guardarla
sobel_image = Image.fromarray(matriz_energia.astype(np.uint8))
sobel_image.show()

#--------------------------------------------------------------------------------------

def calcular_energia_acumulada(matriz_energia):
    filas, columnas = matriz_energia.shape
    
    # Crear una matriz de energía acumulada con las mismas dimensiones que la matriz de energía
    energia_acumulada = np.zeros((filas, columnas))
    
    # La primera fila es la misma que la matriz de energía
    energia_acumulada[0, :] = matriz_energia[0, :]
    
    # Propagar la energía acumulada hacia abajo
    for i in range(1, filas):
        for j in range(columnas):
            # Obtener los valores de los píxeles anteriores (fila superior)
            # Se usa np.inf para los bordes fuera de la matriz
            arriba_izquierda = energia_acumulada[i-1, j-1] if j > 0 else np.inf
            arriba = energia_acumulada[i-1, j]
            arriba_derecha = energia_acumulada[i-1, j+1] if j < columnas - 1 else np.inf
            
            # Sumar el valor mínimo de los tres vecinos anteriores al valor actual de la energía
            energia_acumulada[i, j] = matriz_energia[i, j] + min(arriba_izquierda, arriba, arriba_derecha)
    
    return energia_acumulada


def identificar_costura(energia_acumulada):
    filas, columnas = energia_acumulada.shape
    # Crear una lista para almacenar la columna de cada píxel de la costura
    costura = np.zeros(filas, dtype=np.int)
    
    # Empezamos por la posición de menor energía en la última fila
    costura[-1] = np.argmin(energia_acumulada[-1, :])
    
    # Ahora recorremos desde la penúltima fila hacia la primera
    for i in range(filas - 2, -1, -1):
        j = costura[i + 1]  # Columna del píxel debajo de la costura actual
        
        # Verificamos los vecinos superiores (arriba izquierda, arriba, arriba derecha)
        if j == 0:  # Si está en la primera columna
            pixel_pos = np.argmin(energia_acumulada[i, j:j + 2])  # Solo comparamos j y j+1
            costura[i] = j + pixel_pos
        elif j == columnas - 1:  # Si está en la última columna
            pixel_pos = np.argmin(energia_acumulada[i, j - 1:j + 1])  # Solo comparamos j-1 y j
            costura[i] = j + pixel_pos - 1
        else:  # Si está en el medio
            pixel_pos = np.argmin(energia_acumulada[i, j - 1:j + 2])  # Comparamos j-1, j, j+1
            costura[i] = j + pixel_pos - 1
    
    return costura

#La función np.argmin() devuelve el índice del valor mínimo en el array que le pases. En este caso, le estamos pasando la última fila de la matriz de energía acumulada.
# Esto nos dará la columna donde se encuentra el valor mínimo de energía en la última fila, que será el punto de inicio de la costura.

#--------------------------------------------------------------------

def eliminar_costura(imagen, costura, direccion='vertical'):
    """
    Elimina una costura de una imagen. 
    La costura puede ser vertical (eliminar columnas) u horizontal (eliminar filas).
    
    Args:
        imagen (numpy.ndarray): La imagen de la cual se eliminará la costura.
        costura (numpy.ndarray): Un array con las posiciones de la costura a eliminar.
        direccion (str): 'vertical' para eliminar una columna, 'horizontal' para eliminar una fila.
        
    Returns:
        numpy.ndarray: Imagen con la costura eliminada.
    """
    if direccion == 'horizontal':
        # Transponemos la imagen para eliminar filas como si fueran columnas
        imagen = np.transpose(imagen, (1, 0, 2))
    
    filas, columnas, canales = imagen.shape
    nueva_imagen = np.zeros((filas, columnas - 1, canales), dtype=np.uint8)
    
    for i in range(filas):
        j = costura[i]  # Columna a eliminar en la fila i
        nueva_imagen[i, :, :] = np.delete(imagen[i, :, :], j, axis=0)  # Eliminar el píxel en la columna j
    
    if direccion == 'horizontal':
        # Volvemos a transponer la imagen para devolverla a su orientación original
        nueva_imagen = np.transpose(nueva_imagen, (1, 0, 2))
    
    return nueva_imagen

def reducir_imagen(imagen, nuevo_ancho, nuevo_alto):
    """
    Reduce el tamaño de la imagen eliminando costuras verticales y horizontales.
    
    Args:
        imagen (numpy.ndarray): Imagen original.
        nuevo_ancho (int): Ancho deseado.
        nuevo_alto (int): Altura deseada.
        
    Returns:
        numpy.ndarray: Imagen redimensionada.
    """
    filas, columnas, _ = imagen.shape

    # Reducir el ancho eliminando columnas
    while columnas > nuevo_ancho:
        # Calcular la matriz de energía acumulada y la costura vertical
        energia_acumulada = calcular_energia_acumulada(matriz_energia)
        costura = identificar_costura(energia_acumulada)
        
        # Eliminar la costura vertical (columna)
        imagen = eliminar_costura(imagen, costura, direccion='vertical')
        columnas -= 1
    
    # Reducir la altura eliminando filas
    while filas > nuevo_alto:
        # Calcular la matriz de energía acumulada y la costura horizontal
        energia_acumulada = calcular_energia_acumulada(matriz_energia)
        costura_horizontal = identificar_costura(energia_acumulada)
        
        # Eliminar la costura horizontal (fila)
        imagen = eliminar_costura(imagen, costura_horizontal, direccion='horizontal')
        filas -= 1
    
    return imagen

def reducir_imagen(imagen, nuevo_ancho, nuevo_alto):
    """
    Reduce el tamaño de la imagen eliminando costuras verticales y horizontales.
    
    Args:
        imagen (numpy.ndarray): Imagen original.
        nuevo_ancho (int): Ancho deseado.
        nuevo_alto (int): Altura deseada.
        
    Returns:
        numpy.ndarray: Imagen redimensionada.
    """
    filas, columnas, _ = imagen.shape

    # Reducir el ancho eliminando columnas
    while columnas > nuevo_ancho:
        # Calcular la matriz de energía acumulada y la costura vertical
        energia_acumulada = calcular_energia_acumulada(matriz_energia)
        costura = identificar_costura(energia_acumulada)
        
        # Eliminar la costura vertical (columna)
        imagen = eliminar_costura(imagen, costura, direccion='vertical')
        columnas -= 1
    
    # Reducir la altura eliminando filas
    while filas > nuevo_alto:
        # Calcular la matriz de energía acumulada y la costura horizontal
        energia_acumulada = calcular_energia_acumulada(matriz_energia)
        costura_horizontal = identificar_costura(energia_acumulada)
        
        # Eliminar la costura horizontal (fila)
        imagen = eliminar_costura(imagen, costura_horizontal, direccion='horizontal')
        filas -= 1
    
    return imagen
