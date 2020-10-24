'''
Taller 5 --> Calibración de cámara y cámara proyectiva
Leidy Carolina Pulido Feo
Eliana Andrea Romero Leon
'''


import numpy as np
import cv2
import glob
import os
import json
from camera_model import *


"""
PUNTO 1: Calibración de Cámara
"""
camara = int(input("Ingrese el número de la cámara que quiere calibrar \n "
                   "En caso de querer ejecutar la cámara proyectiva selccione la cámara del celular\n "
                   "1 --> Fotos cámara computador\n "
                   "2 --> Fotos cámara celular \n "))

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((7 * 7, 3), np.float32)     #Número de puntos ancho y largo del tablero
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane. #Las proyecciones son las equinas de cada imagen

#Path de imagenes dependiendo la cámara que se desee calibrar
if camara == 1:
    path = 'J:/Proc.Imagenes/Imagenes/Taller5/Calibration_Images/Computer_Caro_2'
    path_file = os.path.join(path, 'Compu_*.jpeg')
else:
    path = 'J:/Proc.Imagenes/Imagenes/Taller5/Calibration_Images/Phone_Caro_2'
    path_file = os.path.join(path, 'Phone_*.jpeg')

images = glob.glob(path_file)   #Lista de imágenes

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)    #Encontrar las esquinas y se especifica su tamaño

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)  #Lista de puntos 3D

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)    #Encontrar los píxeles con precisión sub pixel
        imgpoints.append(corners2)  #Lista de píxeles

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 7), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(250)

cv2.destroyAllWindows()

#Función de OpenCV para calibrar la cámara, devuelve:
#ret: true o false si se pudo hacer la operación
#mtx: matríz de intrínsecas
#dist: parámetros de distorsión
#rvecs: matríz de rotación
#tvecs: vector de traslación ---> de cada una de las imagenes
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Los parámetros intrínsecos de la cámara son:\n",mtx)

# Error del modelo de la cámara con respecto a los puntos ya hallados
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)  #Se encuentran los puntos de la imagen através de una proyección
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)   #Cálculo del error de los puntos recién encontrados y los anteriores
    mean_error += error

print("Total error: {}".format(mean_error / len(objpoints)))

#Guardar un archivo .JSON
file_name = 'calibration.json'  #Archivo con identificadores y valores
json_file = os.path.join(path, file_name)

#Creación de "diccionario"
data = {
    'K': mtx.tolist()   # Parámetros Intrínsecos
}

#Para escribir diccionario en el archivo json
with open(json_file, 'w') as fp:
    json.dump(data, fp, sort_keys=True, indent=1, ensure_ascii=False)

# Abrir el archivo json y recuperar los datos
# with open(json_file) as fp:
#     json_data = json.load(fp)
# print(json_data)
"""
PUNTO 2: Cámara Proyectiva
"""
if camara == 2:
    print("------------------------------------------- \n A continuación se va a ejecutar la cámara proyectiva \n "
          "Para la configuración de los parámetros algunas opciones son: \n "
          "1 --> tilt = 0, pan = 5, distancia = 2, altura = 1 \n "
          "2 --> tilt = 30, pan = 0, distancia = 3, altura = 2 \n ")

    file_name2 = 'configuration.json'  # Archivo con identificadores y valores
    json_file2 = os.path.join(path, file_name2)

    data2 = {'K': mtx.tolist(),  # Lista de la matriz de instrínsecas
            'tilt': int(input("Ingrese el valor de tilt deseado: ")),
            'pan': int(input("Ingrese el valor de pan deseado: ")),
            'd': int(input("Ingrese el valor de distancia deseada: ")),
            'h': int(input("Ingrese el valor de altura deseada: "))}

    with open(json_file2, 'w') as fp:
        json.dump(data2, fp, sort_keys=True, indent=1, ensure_ascii=False)  # Se escribe el archivo

    # Abrir el archivo json y recuperar los datos de configuración
    with open(json_file2) as fp:
        json_data2 = json.load(fp)
    #print(json_data2)

    # Parámetros de configuración
    mtx = json_data2['K']   #Matríz intrínseca de la cámara
    tilt = json_data2['tilt']   #Tilt
    pan = json_data2['pan']     #Pan
    distancia = json_data2['d'] #Distancia
    altura = json_data2['h']    #Altura

    width =  int(mtx[0][2])*2   #Ancho de la Imagen (Centro en x "matriz K" * 2)
    height = int(mtx[1][2])*2   #Alto de la Imagen (Centro en y "matriz K" * 2)

    #Diseño de cubo, ubicación de los 8 puntos
    cube_3D_comp = np.array([[0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]])

    #Configuración de cámara proyectiva
    R = set_rotation(tilt, pan, 0)  #Rotación; uso de tilt y pan
    t = np.array([0, -distancia, altura])   #Vector de translación
    camera = projective_camera(mtx, width, height, R, t)    #Cámara proyectiva
    cube_2D_comp = projective_camera_project(cube_3D_comp, camera)  #Proyección del cubo según la cámara proyectiva

    image_projective = 255 * np.ones(shape=[camera.height, camera.width, 3], dtype=np.uint8)

    #Dibujo de líneas del cubo
    #Tapa inferior
    cv2.line(image_projective, (cube_2D_comp[0][0], cube_2D_comp[0][1]), (cube_2D_comp[1][0], cube_2D_comp[1][1]), (163, 73, 164), 3) #1-2
    cv2.line(image_projective, (cube_2D_comp[1][0], cube_2D_comp[1][1]), (cube_2D_comp[2][0], cube_2D_comp[2][1]), (163, 73, 164), 3) #2-3
    cv2.line(image_projective, (cube_2D_comp[2][0], cube_2D_comp[2][1]), (cube_2D_comp[3][0], cube_2D_comp[3][1]), (163, 73, 164), 3) #3-4
    cv2.line(image_projective, (cube_2D_comp[3][0], cube_2D_comp[3][1]), (cube_2D_comp[0][0], cube_2D_comp[0][1]), (163, 73, 164), 3) #4-1
    #Tapa superior
    cv2.line(image_projective, (cube_2D_comp[4][0], cube_2D_comp[4][1]), (cube_2D_comp[5][0], cube_2D_comp[5][1]), (163, 73, 164), 3) #5-6
    cv2.line(image_projective, (cube_2D_comp[5][0], cube_2D_comp[5][1]), (cube_2D_comp[6][0], cube_2D_comp[6][1]), (163, 73, 164), 3) #6-7
    cv2.line(image_projective, (cube_2D_comp[6][0], cube_2D_comp[6][1]), (cube_2D_comp[7][0], cube_2D_comp[7][1]), (163, 73, 164), 3) #7-8
    cv2.line(image_projective, (cube_2D_comp[7][0], cube_2D_comp[7][1]), (cube_2D_comp[4][0], cube_2D_comp[4][1]), (163, 73, 164), 3) #8-5
    #Líneas Laterales
    cv2.line(image_projective, (cube_2D_comp[0][0], cube_2D_comp[0][1]), (cube_2D_comp[4][0], cube_2D_comp[4][1]), (163, 73, 164), 3) #4-8
    cv2.line(image_projective, (cube_2D_comp[1][0], cube_2D_comp[1][1]), (cube_2D_comp[5][0], cube_2D_comp[5][1]), (163, 73, 164), 3) #3-7
    cv2.line(image_projective, (cube_2D_comp[2][0], cube_2D_comp[2][1]), (cube_2D_comp[6][0], cube_2D_comp[6][1]), (163, 73, 164), 3) #2-6
    cv2.line(image_projective, (cube_2D_comp[3][0], cube_2D_comp[3][1]), (cube_2D_comp[7][0], cube_2D_comp[7][1]), (163, 73, 164), 3) #1-5

    cv2.imshow("Image", image_projective)
    cv2.imwrite("D:/Desktop/Paraborrar/imagen_proyectiva.jpeg", image_projective)   #Almacenar imagen
    cv2.waitKey(0)
else:
    print("------------------------------------------- \n "
          "Si desea ejecutar el segundo punto de cámara proyectiva debe seleccionar 'Fotos cámara celular'")