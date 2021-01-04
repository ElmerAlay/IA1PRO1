from Util.ReadFile import get_dataFile
from Util import Plotter
from Neural_Network.Data import Data
from Neural_Network.Model import NN_Model

ONLY_SHOW = False

# Cargando conjunto de datos
train_X, train_Y, val_X, val_Y = get_dataFile()

if ONLY_SHOW:
    Plotter.plot_field_data(train_X, train_Y)
    # Plotter.plot_field_data(val_X, val_Y)
    print("Entradas de entrenamiento:", train_X.shape, sep=' ')
    print("Salidas de entrenamiento:", train_Y.shape, sep=' ')
    print("Entradas de validacion:", val_X.shape, sep=' ')
    print("Salidas de validacion:", val_Y.shape, sep=' ')
    exit()

# Definir los conjuntos de datos
train_set = Data(train_X, train_Y)
val_set = Data(val_X, val_Y)

# Se define las dimensiones de las capas
#capas1 = [Cantidad de variables que tiene el problema, capa 1, capa 2, Capa de salida]
#se tendría una red neuronal de 3 capas, la capa de entrada NO se toma en cuenta
#capas1 = [train_set.n, 10, 5, 1]
capas1 = [train_set.n, 10, 7, 1]

# Se define el modelo
nn1 = NN_Model(train_set, capas1, alpha=0.001, iterations=50000, lambd=0, keep_prob=0.5)
nn2 = NN_Model(train_set, capas1, alpha=0.01, iterations=50000, lambd=0.7, keep_prob=1)

# Se entrena el modelo
nn1.training(False)
nn2.training(False)

# Se analiza el entrenamiento
Plotter.show_Model([nn1, nn2])

print('Entrenamiento Modelo 1')
nn1.predict(train_set)
print('Validacion Modelo 1')
nn1.predict(val_set)

print('########################')
print('Entrenamiento Modelo 2')
nn2.predict(train_set)
print('Validacion Modelo 2')
nn2.predict(val_set)


