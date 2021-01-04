import numpy as np
#np.set_printoptions(threshold=100000) #Esto es para que al imprimir un arreglo no me muestre puntos suspensivos


class NN_Model:

    def __init__(self, train_set, layers, alpha=0.3, iterations=300000, lambd=0, keep_prob=1):
        self.data = train_set
        self.alpha = alpha
        self.max_iteration = iterations
        self.lambd = lambd
        self.kp = keep_prob
        # Se inicializan los pesos
        self.parametros = self.Inicializar(layers)
        self.cant_layers = len(layers)-1

    def Inicializar(self, layers):
        parametros = {}
        L = len(layers)
        print('layers:', layers)
        for l in range(1, L):
            #np.random.randn(layers[l], layers[l-1])
            #Crea un arreglo que tiene layers[l] arreglos, donde cada uno de estos arreglos tiene layers[l-1] elementos con valores aleatorios
            #np.sqrt(layers[l-1] se saca la raiz cuadrada positiva de la capa anterior ---> layers[l-1]
            parametros['W'+str(l)] = np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1])
            parametros['b'+str(l)] = np.zeros((layers[l], 1))
            #print(layers[l], layers[l-1], np.random.randn(layers[l], layers[l-1]))
            #print(np.sqrt(layers[l-1]))
            #print(np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1]))

        return parametros

    def training(self, show_cost=False):
        self.bitacora = []
        for i in range(0, self.max_iteration):
            y_hat, temp = self.propagacion_adelante(self.data)
            cost = self.cost_function(y_hat)
            gradientes = self.propagacion_atras(temp)
            self.actualizar_parametros(gradientes)
            if i % 50 == 0:
                self.bitacora.append(cost)
                if show_cost:
                    print('Iteracion No.', i, 'Costo:', cost, sep=' ')


    def propagacion_adelante(self, dataSet):
        # Se extraen las entradas
        X = dataSet.x
        A = []
        Z = []
        D = []
        
        for i in range(1, self.cant_layers+1):
            W = self.parametros["W"+str(i)]
            b = self.parametros["b"+str(i)]

            if i==1:
                Z.append(np.dot(W, X) + b)
            else:
                Z.append(np.dot(W, A[i-2]) + b)

            #Dropout invertido
            if i!=self.cant_layers:
                Ai = self.activation_function('relu', Z[i-1])

                Di = np.random.rand(Ai.shape[0], Ai.shape[1])
                Di = (Di < self.kp).astype(int)
                D.append(Di)

                Ai *= Di
                Ai /= self.kp
                A.append(Ai)
            else:
                A.append(self.activation_function('sigmoide', Z[i-1]))

        temp = []
        for i in range(len(D)):
            temp.append(Z[i])
            temp.append(A[i])
            temp.append(D[i])
        temp.append(Z[len(Z)-1])
        temp.append(A[len(A)-1])

        temp = tuple(temp)

        #En A3 va la predicción o el resultado de la red neuronal
        return A[len(A)-1], temp

    def propagacion_atras(self, temp):
        # Se obtienen los datos
        m = self.data.m
        Y = self.data.y
        X = self.data.x
        W = []
        dA = []
        dZ = []
        dW = []
        db = []

        temps = list(temp)
        for i in range(self.cant_layers-1,-1, -1):
            W.append(self.parametros["W"+str(i+1)])

            dZi = 1.
            dAi = 1.
            dWi = 1.
            if i==(self.cant_layers)-1:
                dZi= temps[(i*3)+1] - Y
                dZ.append(dZi)
            else:
                dAi = np.dot(W[((self.cant_layers)-1)-(i+1)].T, dZ[i-1])
                dAi *= temps[(i*3)+2]
                dAi /= self.kp
                dA.append(dAi)

                dZi = np.multiply(dAi,np.int64(temps[(i*3)+1]>0))
                dZ.append(dZi)
                
            if i==0:
                dWi = 1. / m * np.dot(dZi, X.T) + (self.lambd/m) * W[((self.cant_layers)-1)-i]
            else:
                dWi = 1. / m * np.dot(dZi, temps[((i-1)*3)+1].T) + (self.lambd/m) * W[((self.cant_layers)-1)-i]
            dW.append(dWi)

            db.append(1. / m * np.sum(dZi, axis=1, keepdims=True))
        
        gradientes = {}
        gradientes["dZ"+str(len(dZ))] = dZ[0]
        gradientes["dW"+str(len(dW))] = dW[0]
        gradientes["db"+str(len(db))] = db[0]
        for i in range(1, len(dZ)):
            gradientes["dA"+str(len(db)-(i-1))] = dA[i-1]
            gradientes["dZ"+str(len(dZ)-i)] = dZ[i]
            gradientes["dW"+str(len(dW)-i)] = dW[i]
            gradientes["db"+str(len(db)-i)] = db[i]

        return gradientes

    def actualizar_parametros(self, grad):
        # Se obtiene la cantidad de pesos
        L = len(self.parametros) // 2
        for k in range(L):
            self.parametros["W" + str(k + 1)] -= self.alpha * grad["dW" + str(k + 1)]
            self.parametros["b" + str(k + 1)] -= self.alpha * grad["db" + str(k + 1)]

    def cost_function(self, y_hat):
        # Se obtienen los datos
        Y = self.data.y
        m = self.data.m
        # Se hacen los calculos
        temp = np.multiply(-np.log(y_hat), Y) + np.multiply(-np.log(1 - y_hat), 1 - Y)
        result = (1 / m) * np.nansum(temp)
        # Se agrega la regularizacion L2
        if self.lambd > 0:
            L = len(self.parametros) // 2
            suma = 0
            for i in range(L):
                suma += np.sum(np.square(self.parametros["W" + str(i + 1)]))
            result += (self.lambd/(2*m)) * suma
        return result

    def predict(self, dataSet):
        # Se obtienen los datos
        m = dataSet.m
        Y = dataSet.y
        p = np.zeros((1, m), dtype= np.int)
        # Propagacion hacia adelante
        y_hat, temp = self.propagacion_adelante(dataSet)
        # Convertir probabilidad
        for i in range(0, m):
            p[0, i] = 1 if y_hat[0, i] > 0.5 else 0
        exactitud = np.mean((p[0, :] == Y[0, ]))
        print("Exactitud: " + str(exactitud))
        return exactitud


    def activation_function(self, name, x):
        result = 0
        if name == 'sigmoide':
            result = 1/(1 + np.exp(-x))
        elif name == 'tanh':
            result = np.tanh(x)
        elif name == 'relu':
            result = np.maximum(0, x)
        
        #print('name:', name, 'result:', result)
        return result