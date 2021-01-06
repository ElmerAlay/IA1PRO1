import random
import numpy as np
from Util.ReadFile import get_dataFile
from Neural_Network.Data import Data
from Neural_Network.Model import NN_Model

class Nodo:
    def __init__(self, solucion=[], fitness=0):
        self.solucion = solucion
        self.fitness = fitness

class Genetico:

    def __init__(self): 
        self.hiper = [
            [0.05,0.01,0.025,0.005,0.001,0.0025,0.0005,0.0001,0.00025,0.00005],
            [0.05,0.01,0.5,0.1,0,1,2,3,4,5],
            [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],
            [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        ]

    def inicializarPoblacion(self):
        #Inicializo el arreglo de la población
        poblacion = []

        #creando los individuos
        for i in range(9):
            #Creo un nodo que va a almacenar los genes
            individuo = Nodo()
            #Creo el arreglo de genes
            genes = []
            
            #Creando los genes
            for j in range(4):
                #agrego los genes al arreglo
                genes.append(self.hiper[j][random.randint(0,9)])

            #colocamos la solución al nodo
            individuo.solucion = genes
            #agrego este individuo a la poblacion
            poblacion.append(individuo)

        #ultimo individuo
        individuo = Nodo()
        individuo.solucion = [self.hiper[0][9],self.hiper[1][1],self.hiper[2][9],self.hiper[3][9]]
        poblacion.append(individuo)

        #retorno la población
        return poblacion

    def evaluarFitness(self, solucion):
        valorFitness = 0
        train_X, train_Y, val_X, val_Y, test_X, test_Y = get_dataFile()

        train_set = Data(train_X, train_Y)
        val_set = Data(val_X, val_Y)
        test_set = Data(test_X, test_Y)

        capas1 = [train_set.n, 9, 6, 3, 1]
        nn2 = NN_Model(train_set, capas1, alpha=solucion[0], iterations=solucion[2], lambd=solucion[1], keep_prob=solucion[3])
        nn2.training(False)
        nn2.predict(train_set)
        nn2.predict(test_set)
        valorFitness, resultado = nn2.predict(val_set)

        return valorFitness

    def verificarCriterio(self, poblacion, generacion):
        result = None

        #calcular valor fitness en los individuos
        for individuo in poblacion:
            individuo.fitness = self.evaluarFitness(individuo.solucion)

        if generacion==7:
            result = True

        return result

    def ordenar(self, poblacion):
        for x in range(10):
            for i in range(9):
                if poblacion[i].fitness<poblacion[i+1].fitness:
                    tmp = poblacion[i+1]
                    poblacion[i+1] = poblacion[i]
                    poblacion[i] = tmp

        return poblacion 

    def seleccionarPadres(self, poblacion):
        mejoresPadres = []
        
        #Ordenamos la población de acuerdo al valor fitness de mayor a menor
        poblacion = self.ordenar(poblacion)

        #Obtenemos los 5 individuos más altos
        mejoresPadres.append(poblacion[0])
        mejoresPadres.append(poblacion[1])
        mejoresPadres.append(poblacion[2])
        mejoresPadres.append(poblacion[3])
        mejoresPadres.append(poblacion[4])

        return mejoresPadres

    def cruzar(self, padre1, padre2):
        hijo = []
        for i in range(4):
            r = random.random()
            if r > 0.5:
                hijo.append(padre2[i])
            else:
                hijo.append(padre1[i])

        return hijo

    def mutar(self, solucion):
        pos = random.randint(0,3)
        solucion[pos]=random.randint(0,9)

        return solucion

    def emparejar(self, padres):
        nuevaPoblacion = padres

        hijo1 = Nodo()
        hijo1.solucion = self.cruzar(padres[0].solucion, padres[1].solucion)
        hijo1.solucion = self.mutar(hijo1.solucion)

        hijo2 = Nodo()
        hijo2.solucion = self.cruzar(padres[2].solucion, padres[4].solucion)
        hijo2.solucion = self.mutar(hijo2.solucion)

        hijo3 = Nodo()
        hijo3.solucion = self.cruzar(padres[0].solucion, padres[2].solucion)
        hijo3.solucion = self.mutar(hijo3.solucion)

        hijo4 = Nodo()
        hijo4.solucion = self.cruzar(padres[2].solucion, padres[3].solucion)
        hijo4.solucion = self.mutar(hijo4.solucion)

        hijo5 = Nodo()
        hijo5.solucion = self.cruzar(padres[0].solucion, padres[4].solucion)
        hijo5.solucion = self.mutar(hijo5.solucion)

        nuevaPoblacion.append(hijo1)
        nuevaPoblacion.append(hijo2)
        nuevaPoblacion.append(hijo3)
        nuevaPoblacion.append(hijo4)
        nuevaPoblacion.append(hijo5)

        return nuevaPoblacion

    def imprimirPoblacion(self, poblacion):
        for individuo in poblacion:
            print('Individuo: ', individuo.solucion, ' Fitness: ', individuo.fitness)

    def algoritmo(self):
        #inicialización de variables
        generacion = 0
        poblacion = self.inicializarPoblacion()
        fin = self.verificarCriterio(poblacion, generacion)

        while(fin==None):
            padres = self.seleccionarPadres(poblacion)
            poblacion = self.emparejar(padres)
            generacion += 1
            fin = self.verificarCriterio(poblacion, generacion)

        arregloMejorIndividuo = self.ordenar(poblacion)
        mejorIndividuo = arregloMejorIndividuo[0]

        return mejorIndividuo.solucion
