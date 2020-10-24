import numpy as np


class projective_camera:
    def __init__(self, K, width, height, R, t):
        self.K = K #Matriz de intrínsecas ->  fx, 0, cx], [0, fy, cy], [0, 0, 1.0] .
        self.width = width #Ancho.
        self.height = height #Largo.
        self.R = R #Rotación.
        self.t = t #Traslación.


def projective_camera_project(p_3D, camera):

    p_3D_ = np.copy(p_3D) #Copia del punto 3D.
    for i in range(3):
        p_3D_[:, i] = p_3D_[:, i] - camera.t[i] #Se toma el punto 3D y se remueve las componentes de translación de la cámara.

    p_3D_cam = np.matmul(camera.R, p_3D_.T) #Se rotan los puntos. SE DEBE CHEQUEAR QUE Z NO DE NEGATIVO
    p_2D = np.matmul(camera.K, p_3D_cam) #Proyección del punto.

    for i in range(2):
        p_2D[i, :] /= p_2D[2, :] #Se deshomogeniza el punto.

    p_2D = p_2D[:2, :] #Se deshomogeniza el punto.
    p_2D = p_2D.transpose() #Se transpone.
    p_2D = p_2D.astype(int) #Conversión a entero.

    return p_2D


def set_rotation(tilt, pan=0, skew=0): #tilt : que tan abajo esta la cámara, si esta horizontal es 0, pan : giro de izquierda a derecha, skew : rotación de la cámara.

    R = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]) #Camara que mira hacia el horizonte.
    theta_x = tilt * np.pi / 180 #Se transforma el tilt a radianes.
    theta_y = skew * np.pi / 180 #Se transforma el skew a radianes.
    theta_z = pan * np.pi / 180 #Se transforma el pan a radianes.

    #Matrices de rotación.
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]]) 
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])

    R_ = np.matmul(np.matmul(Rz, Ry), Rx) #Multiplicación de las componentes de la matriz de rotación hallada.
    R_new = np.matmul(R, R_) #Nueva matriz de rotación, tiene en cuenta la rotación inicial y la que se realizó anteriormente.

    return R_new