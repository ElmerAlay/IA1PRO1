import numpy as np

class Data:
    def __init__(self, data_set_x, data_set_y, max_value=1):
        self.m = data_set_x.shape[1]
        self.n = data_set_x.shape[0]
        #self.x = data_set_x / max_value #escalacion de variables
        if data_set_x.shape[1] == 1:
            self.x = self.escalamiento2(data_set_x.T)  
        else:
            self.x = self.escalamiento(data_set_x.T)        
        self.y = data_set_y

    def escalamiento2(self, X):
        genero = 0
        edad = 0
        anio = 0
        dist = 0

        for i in X:
            genero= int(i[0])
            edad=int(i[1])/100
            anio=int(i[2])/10000
            dist=float(i[3])/100
            print("distancia: ", dist)

        result = [[genero, edad, anio, dist]]
        result = np.array(result)
        
        return result.T

    def escalamiento(self, X):
        genero = []
        edad = []
        anio = []
        dist = []
        
        for i in X:
            genero.append(int(i[0]))
            edad.append(int(i[1]))
            anio.append(int(i[2]))
            dist.append(float(i[3]))

        edad_esc = []
        for i in edad:
            edad_esc.append((i-min(edad))/(max(edad)-min(edad)))
        
        anio_esc = []
        for i in anio:
            anio_esc.append((i-min(anio))/(max(anio)-min(anio)))
        
        dist_esc = []
        for i in dist:
            dist_esc.append((i-min(dist))/(max(dist)-min(dist)))

        result = []
        for i in range(len(X)):
            result.append([genero[i], edad_esc[i], anio_esc[i], dist_esc[i]])

        result = np.array(result)
        
        return result.T