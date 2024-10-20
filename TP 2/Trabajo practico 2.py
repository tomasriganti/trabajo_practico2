import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import tkinter as ttk

def solicitar_ancho_largo() -> tuple[str, int, int]:
    """
    Solicita al usuario la ruta de la imagen y las nuevas dimensiones.

    Arguments:
        None

    Returns:
        Tuple con la ruta (str), el nuevo ancho (int) y el nuevo alto (int).
    """
    path = input("Ingrese la ruta a la imagen: ")
    ancho = int(input("Ingrese el nuevo ancho de la imagen: "))
    alto = int(input("Ingrese el nuevo alto de la imagen: ")) 

    return path, ancho, alto


#el usuario ingresa los datos 
path, ancho_nuevo, alto_nuevo = solicitar_ancho_largo()

def generar_matriz_rgb(path: str) -> np.array:
    """
    Genera una matriz RGB a partir de una imagen en la ruta dada.

    Arguments:
        path (str): Ruta a la imagen.

    Returns:
        np.array: Matriz RGB de la imagen.
    """
    matriz = np.array(Image.open(path))
    return matriz

#matriz RGB
matriz_RGB = generar_matriz_rgb(path)

cant_columnas, cant_filas, cant_d = matriz_RGB.shape

if (ancho_nuevo < 0) or (alto_nuevo < 0) or (ancho_nuevo > cant_columnas) or (alto_nuevo > cant_filas):
    print("El ancho de la imagen no puede ser negativo ni mayor al ancho original.")
          
#divido en 3 matrices, en RGB
matriz_rojo = matriz_RGB[ :, :, 0] #Red
matriz_verde = matriz_RGB[ :, :, 1] #Green
matriz_azul = matriz_RGB[ :, :, 2] #Blue

def matriz_con_padding(matriz: np.array, ancho: int = 1) -> np.array:
    """
    Aplica padding a la matriz dada con un borde de ancho especificado.

    Arguments:
        matriz (np.array): Matriz original.
        ancho (int): Ancho del padding a agregar. Default es 1.

    Returns:
        np.array: Matriz con padding aplicado.
    """
    matriz_padded = np.pad(matriz, pad_width=ancho, mode='edge')
    return matriz_padded

# Agregar padding a las matrices en cada color
matriz_rojo_padded = matriz_con_padding(matriz_rojo)
matriz_verde_padded = matriz_con_padding(matriz_verde)
matriz_azul_padded = matriz_con_padding(matriz_azul)

def sobel_combinado(matriz: np.array) -> tuple[np.array, np.array]:
    """
    Aplica el operador Sobel en las direcciones X e Y.

    Arguments:
        matriz (np.array): Matriz en escala de grises.

    Returns:
        tuple: Matrices resultantes de aplicar el Sobel en X e Y.
    """
    filas, columnas = matriz.shape
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1, -2, -1]])

    sobel_result_x = np.zeros((filas-2, columnas-2))
    sobel_result_y = np.zeros((filas-2, columnas-2))

    for i in range(1, filas-1):
        for j in range(1, columnas-1):
            submatriz = matriz[i-1:i+2, j-1:j+2]
            gx = np.sum(sobel_x * submatriz)
            gy = np.sum(sobel_y * submatriz)
            sobel_result_x[i-1, j-1] = gx
            sobel_result_y[i-1, j-1] = gy

    return sobel_result_x, sobel_result_y

# Aplicar Sobel a cada canal
sobel_x_red, sobel_y_red = sobel_combinado(matriz_rojo_padded)
sobel_x_green, sobel_y_green = sobel_combinado(matriz_verde_padded)
sobel_x_blue, sobel_y_blue = sobel_combinado(matriz_azul_padded)

# Promediar los resultados de los tres canales con su sobel correspondiente
prom_sobel_x = (sobel_x_red + sobel_x_green + sobel_x_blue)/3
prom_sobel_y = (sobel_y_red + sobel_y_green + sobel_y_blue)/3

#aplico la formula de energia
matriz_energia = np.sqrt((prom_sobel_x)**2 + (prom_sobel_y)**2) 

def calcular_energia_acumulada(matriz: np.array) -> np.array:
    """
    Calcula la energía acumulada en cada punto de la matriz.

    Arguments:
        matriz (np.array): Matriz de energía inicial.

    Returns:
        np.array: Matriz de energía acumulada.
    """
    filas, columnas = matriz.shape
    energia_acumulada = np.zeros((filas, columnas))
    energia_acumulada[0, :] = matriz[0, :]

    for i in range(1, filas):
        for j in range(columnas):
            arriba_izquierda = energia_acumulada[i-1, j-1] if j > 0 else np.inf
            arriba = energia_acumulada[i-1, j]
            arriba_derecha = energia_acumulada[i-1, j+1] if j < columnas - 1 else np.inf
            energia_acumulada[i, j] = matriz[i, j] + min(arriba_izquierda, arriba, arriba_derecha)

    return energia_acumulada

#matriz de energia acumulada
matriz_acumulada = calcular_energia_acumulada(matriz_energia)

def eliminar_costura_columnas(matriz1: np.array, matriz2: np.array) -> tuple[np.array, np.array]:
    """
    Elimina una costura vertical de la matriz1 y la matriz2.

    Arguments:
        matriz1 (np.array): Matriz de energía.
        matriz2 (np.array): Matriz de colores RGB.

    Returns:
        tuple: Matriz de energía y matriz RGB modificadas.
    """
    matriz1 = matriz1.tolist()
    matriz2 = matriz2.tolist()
    
    num_filas = len(matriz1)
    
    lista_actual_1 = matriz1[-1]
    j = lista_actual_1.index(min(lista_actual_1))
    lista_actual_1.pop(j)

    lista_actual_2 = matriz2[-1]
    lista_actual_2.pop(j)
    
    for i in range(-2, -(num_filas + 1), -1):
        lista_actual_1 = matriz1[i]
        lista_actual_2 = matriz2[i]
        
        if j == 0:
            sub_lista_1 = lista_actual_1[j:j+2]
        elif j == (len(lista_actual_1) - 1):
            sub_lista_1 = lista_actual_1[j-1:j+1]
        else:
            sub_lista_1 = lista_actual_1[j-1:j+2]
        
        nuevo_min = min(sub_lista_1)
        j = lista_actual_1.index(nuevo_min)
        
        lista_actual_1.pop(j)
        lista_actual_2.pop(j)

    matriz1 = np.array(matriz1)
    matriz2 = np.array(matriz2)
    
    return matriz1, matriz2

def eliminar_costura_filas(matriz1: np.array, matriz2: np.array) -> tuple[np.array, np.array]:
    """
    Elimina una costura horizontal de la matriz1 y la matriz2.

    Arguments:
        matriz1 (np.array): Matriz de energía.
        matriz2 (np.array): Matriz de colores RGB.

    Returns:
        tuple: Matriz de energía y matriz RGB modificadas.
    """
    matriz1_transpuesta = np.transpose(matriz1, axes=(1, 0)) 
    matriz2_transpuesta = np.transpose(matriz2, axes=(1, 0, 2)) 

    matriz1_modificada, matriz2_modificada = eliminar_costura_columnas(matriz1_transpuesta, matriz2_transpuesta)
    
    matriz1_final = np.transpose(matriz1_modificada, axes=(1, 0))
    matriz2_final = np.transpose(matriz2_modificada, axes=(1, 0, 2)) 
    
    return matriz1_final, matriz2_final

def ajustar_matriz(matriz_1: np.array, matriz_2: np.array, columnas_deseadas: int, filas_deseadas: int) -> tuple[np.array, int]:
    """
    Ajusta las dimensiones de la matriz a las columnas y filas deseadas, eliminando costuras.

    Arguments:
        matriz_1 (np.array): Matriz de energía.
        matriz_2 (np.array): Matriz RGB.
        columnas_deseadas (int): Número deseado de columnas.
        filas_deseadas (int): Número deseado de filas.

    Returns:
        tuple: Matriz RGB ajustada y el número de pasos realizados.
    """
    filas, columnas = matriz_2.shape[:2]
    n = 1
    
    while columnas > columnas_deseadas:
        matriz_1, matriz_2 = eliminar_costura_columnas(matriz_1, matriz_2)
        filas, columnas = matriz_2.shape[:2] 
        guardar_imagen(matriz_2, n)
        n += 1 

    while filas > filas_deseadas:
        matriz_1, matriz_2 = eliminar_costura_filas(matriz_1, matriz_2)
        filas, columnas = matriz_2.shape[:2]
        guardar_imagen(matriz_2, n)
        n += 1

    return matriz_2, n

def guardar_imagen(matriz: np.array, n: int) -> None:
    """
    Guarda una imagen en formato JPG a partir de la matriz dada.

    Arguments:
        matriz (np.array): Matriz RGB de la imagen.
        n (int): Número para nombrar el archivo.

    Returns:
        None
    """
    matriz = np.clip(matriz, 0, 255).astype(np.uint8)
    imagen = Image.fromarray(matriz)
    nombre_archivo = (f'{n}.jpg')
    imagen.save(nombre_archivo)

def actualizar_imagen(valor: int, label_imagen: tk.Label) -> None:
    """
    Actualiza la imagen mostrada en la ventana según el valor del deslizador.

    Arguments:
        valor (int): Valor actual del deslizador.
        label_imagen (tk.Label): Label donde se muestra la imagen.

    Returns:
        None
    """
    nombre_imagen = f'{valor}.jpg'
    nueva_imagen = Image.open(nombre_imagen)
    nueva_imagen_tk = ImageTk.PhotoImage(nueva_imagen)
    label_imagen.config(image=nueva_imagen_tk)
    label_imagen.image = nueva_imagen_tk

def mostrar_imagen_con_deslizador(path: str, n: int) -> None:
    """
    Muestra una imagen con un control deslizante para actualizarla.

    Arguments:
        path (str): Ruta a la imagen inicial.
        n (int): Número máximo del deslizador.

    Returns:
        None
    """
    # Crear la ventana principal
    ventana = tk.Tk()
    ventana.title("Imagen con Deslizador")

    # Cargar la imagen inicial
    imagen_original = Image.open(path)
    imagen_tk = ImageTk.PhotoImage(imagen_original)

    label_imagen = tk.Label(ventana, image=imagen_tk)
    label_imagen.image = imagen_tk
    label_imagen.pack()

    # Crear un Frame para colocar el deslizador
    frame_control = ttk.Frame(ventana)
    frame_control.pack(fill=tk.X, pady=10)

    # Crear una barra deslizante (de 0 a n como ejemplo)
    deslizador = tk.Scale(frame_control, from_=0, to=n, orient=tk.HORIZONTAL, command=lambda valor: actualizar_imagen(int(valor), label_imagen))
    deslizador.pack(fill=tk.X)

    # Ejecutar la ventana
    ventana.mainloop()

#matriz final
matriz_final, n = ajustar_matriz(matriz_acumulada, matriz_RGB, ancho_nuevo, alto_nuevo)

mostrar_imagen_con_deslizador(path, n)
