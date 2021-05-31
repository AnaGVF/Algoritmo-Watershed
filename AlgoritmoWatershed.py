# Algoritmo Watershed

# Nombre: Ana Graciela Vassallo Fedotkin
# Expediente: 278775

# Nombre: César Balderas Guillén
# Expediente: 278837

# Fecha: 2 de Junio de 2021.

import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread("img/coins.png", 2)

#Dimensiones#
dimensiones = img.shape
r = dimensiones[0]
c = dimensiones[1]

imgBinarizada = np.zeros((r, c), dtype=np.uint8)

# Función Imagen Binarizada
def umbralizacionBinaria(img):
    for i in range(r):
        for j in range(c):
            if(int(img[i][j] >= 127)):
                imgBinarizada[i][j] = 255
            else:
                imgBinarizada[i][j] = 0

    return imgBinarizada

resultado = umbralizacionBinaria(img)

#Dilatación#
kernel=np.ones((3,3), np.uint8)
dilatada = cv2.dilate(resultado, kernel, iterations=5)


plt.subplot(1, 2, 1)
plt.imshow(resultado, cmap='gray')
plt.title("Original")
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(dilatada, cmap='gray')
plt.title("Original")
plt.xticks([]), plt.yticks([])

plt.show()


