# Algoritmo Watershed

# Nombre: Ana Graciela Vassallo Fedotkin
# Expediente: 278775

# Nombre: César Balderas Guillén
# Expediente: 278837

# Fecha: 7 de Junio de 2021.

import cv2
from matplotlib import pyplot as plt
import numpy as np

# Leer la imagen original #
imgOriginal = cv2.imread("img/coins.png", 2)

img = cv2.imread("img/coins.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Dimensiones #
dimensiones = img.shape
r = dimensiones[0]
c = dimensiones[1]

imgBinarizada = np.zeros((r, c), dtype=np.uint8)

# Función Imagen Binarizada Inversa #
def umbralizacionBinaria(gray):
    for i in range(r):
        for j in range(c):
            if(int(gray[i][j] >= 127)):
                imgBinarizada[i][j] = 0
            else:
                imgBinarizada[i][j] = 255

    return imgBinarizada

resultadoBinarizado = umbralizacionBinaria(gray)

# Filtro Mediana #
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
dist_transform = cv2.distanceTransform(dilatada, cv2.DIST_L2, 3)

# Imagen Umbralizada 2da Vez 
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
# Encontrando sure foreground area
sure_fg = np.uint8(sure_fg)

# Borders 
unknownRegion = cv2.subtract(dilatada, sure_fg)

# Etiquetado de Markers
ret, markers = cv2.connectedComponents(sure_fg)
# Añadir 1 a las etiquetas para asegurarse de que el fondo no es 0, sino 1
markers = markers + 1
# Etiquetar la región desconocida con 0
markers[unknownRegion == 255] = 0

# Watershed 
water = markers.copy()
for i in range(0, markers.shape[0]):
    for j in range(0, markers.shape[1]):
        if markers[i][j] > 1:                                
            for a in range(-10, 10):           
                for b in range(-10, 10):       
                    water[i+b][j+a] = markers[i][j]

water[water == 0] = -1

img[water == -1] = [255, 0, 0]

######################### IMAGENES SUBPLOT #####################################

# Imagen Original 
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