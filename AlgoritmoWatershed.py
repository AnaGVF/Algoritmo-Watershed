##############################################

# Proyecto Final: Algoritmo Watershed

# Nombre: Ana Graciela Vassallo Fedotkin
# Expediente: 278775

# Nombre: César Balderas Guillén
# Expediente: 278837

# Fecha: 7 de Junio de 2021.

##############################################

# Importación de Librerías a usar 
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Leer la imagen original 
imgOriginal = cv2.imread("img/coins.png", 2)

# Leer la imagen a manipular y convertirla a escala de grises 
img = cv2.imread("img/coins.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Dimensiones de la imagen
dimensiones = img.shape
r = dimensiones[0]
c = dimensiones[1]

imgBinarizada = np.zeros((r, c), dtype=np.uint8)

# Función Imagen Binarizada Inversa 
# La Función de Umbralización Inversa lo que hace es convertir todos los valores de los pixeles encontrados en la imagen que son mayores o iguales a 127 e igualarlos a 0; de lo contrario se igualan al valor
# máximo de 255 
def umbralizacionBinaria(gray):
    for i in range(r):
        for j in range(c):
            if(int(gray[i][j] >= 127)):
                imgBinarizada[i][j] = 0
            else:
                imgBinarizada[i][j] = 255

    return imgBinarizada

resultadoBinarizado = umbralizacionBinaria(gray)

# Filtro Mediana 
# El Filtro Mediana lo que hace es recopilar los valores de pixeles que vienen debajo de un filtro y se toma la mediana de esos valores. El resultado de esa mediana es puesto a cada pixel central de 
# los calculados en la imagen 
imagenFiltroMediana = np.zeros(gray.shape, np.uint8)

i = 1
j = 1
for i in range(r-1):
    for j in range(c-1):
        vecinosOrdenados = []
        vecinosOrdenados.extend((resultadoBinarizado[i-1, j-1], resultadoBinarizado[i, j-1], resultadoBinarizado[i+1, j-1], resultadoBinarizado[i-1, j], resultadoBinarizado[i, j], resultadoBinarizado[i+1, j], resultadoBinarizado[i-1, j+1], resultadoBinarizado[i, j+1], resultadoBinarizado[i+1, j+1]))
        vecinosOrdenados.sort()
        valorMedio = int((len(vecinosOrdenados) + 1) / 2)
        imagenFiltroMediana[i, j] = vecinosOrdenados[valorMedio]

# Dilatación
# Aplicando esta función sobre la imagen obtendremos el efecto de expandir o ampliar la región de la imagen que estamos trabajando.
def dilatacionBasica(imagenFiltroMediana):
    i = 1
    j = 1

    for i in range(r-1):
        for j in range(c-1):
            if(imagenFiltroMediana[i, j] < imagenFiltroMediana[i+1, j]):
                imagenFiltroMediana[i, j] = imagenFiltroMediana[i+1, j]

    for i in range(r-1):
        for j in range(c-1):
            if(imagenFiltroMediana[i, j] < imagenFiltroMediana[i, j+1]):
                imagenFiltroMediana[i, j] = imagenFiltroMediana[i, j+1]

    return imagenFiltroMediana


dilatada = dilatacionBasica(imagenFiltroMediana)

# Función Distancia 
# Lo que hace es tomar las intensidades de niveles de gris y las cambia para distanciar sus respectivas distancias del valor 0 más cercano.

# Lo que se está haciendo aquí es encontrar la Sure Foreground Area, el cual es el área en donde los límites de los objetos se encuentran con el fondo.
dist_transform = cv2.distanceTransform(dilatada, cv2.DIST_L2, 3)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)

# Unknown Region 
# Se encuentra la región desconocida donde no se sabe con exactitud si se trata de los objetos o del fondo en sí. Aquí se realiza una resta entre el área dilatada y el área segura de la imagen.
unknownRegion = cv2.subtract(dilatada, sure_fg)

# Etiquetado de Markers
# Los Markers lo que hacen es etiquetar las regiones que son seguras con enteros positivos y las áreas que no son seguras son etiquetadas con un 0. Lo que hace es etiquetar el fondo de la imagen con 0 y
# todos los demás objetos con enteros positivos empezando con 1. 
ret, markers = cv2.connectedComponents(sure_fg)
# Añadir 1 a las etiquetas para asegurarse de que el fondo no es 0, sino 1
markers = markers + 1
# Etiquetar la región desconocida con 0
markers[unknownRegion == 255] = 0

# Watershed 
# Toma una copia de los markers, revisa cuáles valores son positivos para asignarles el color rojo correspondiente a los bordes y dejar en claro cuál objeto es cual.
# Como resultado, todos las monedas encontradas van a tener su propio borde rojo tomado del watershed.
water = markers.copy()
for i in range(0, markers.shape[0]):
    for j in range(0, markers.shape[1]):
        if markers[i][j] > 1:                                
            for a in range(-10, 10):           
                for b in range(-10, 10):       
                    water[i+b][j+a] = markers[i][j]

water[water == 0] = -1

# Se le asigna el color rojo a los bordes de las monedas correspondientes 
img[water == -1] = [255, 0, 0]

######################### IMAGENES SUBPLOT #####################################

#Imagen Original#
plt.subplot(2, 4, 1)
plt.imshow(imgOriginal, cmap='gray')
plt.title("1. Original")
plt.xticks([]), plt.yticks([])

#Imagen Binarizada#
plt.subplot(2, 4, 2)
plt.imshow(resultadoBinarizado, cmap='gray')
plt.title("2. Imagen Binarizada")
plt.xticks([]), plt.yticks([])

#Imagen Filtro Mediana#
plt.subplot(2, 4, 3)
plt.imshow(imagenFiltroMediana, cmap='gray')
plt.title("3. Imagen Filtro Mediana")
plt.xticks([]), plt.yticks([])

#Imagen Dilatada#
plt.subplot(2, 4, 4)
plt.imshow(dilatada, cmap='gray')
plt.title("4. Imagen Dilatada")
plt.xticks([]), plt.yticks([])

#Distancia#
plt.subplot(2, 4, 5)
plt.imshow(dist_transform, cmap='gray')
plt.title("5. dist_transform")
plt.xticks([]), plt.yticks([])

#Sure Foreground Area#
plt.subplot(2, 4, 6)
plt.imshow(sure_fg, cmap='gray')
plt.title("6. sure foreground area")
plt.xticks([]), plt.yticks([])

#Unknown Region#
plt.subplot(2, 4, 7)
plt.imshow(unknownRegion, cmap='gray')
plt.title("7. Unknown Region")
plt.xticks([]), plt.yticks([])

#Final#
plt.subplot(2, 4, 8)
plt.imshow(img, cmap='gray')
plt.title("8. Final")
plt.xticks([]), plt.yticks([])

plt.show()