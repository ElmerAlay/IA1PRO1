import scipy.io
import csv
import numpy as np
import math

def get_dataFile():
    """data = scipy.io.loadmat('datasets/data.mat')

    train_X = data['X'].T
    train_Y = data['y'].T
    val_X = data['Xval'].T
    val_Y = data['yval'].T

    print(train_X.shape)
    print(train_Y.shape)
    print(val_X.shape)
    print(val_Y.shape)
    print("-----------------")
    #print('data[\'y\']', data['y'])
    #print('data[\'y\'].T', data['y'].T)

    return train_X, train_Y, val_X, val_Y"""

    results = []

    with open('./datasets/Dataset.csv') as File:
        reader = csv.DictReader(File)
        for row in reader:
            results.append(row)
    
    municipios = read_Municipios()

    data = []
    Y = []
    for i in results:
        genero = 1
        if i['Genero'].lower()=='masculino':
            genero = 0

        distancia = calc_distancia(i['cod_depto'], i['cod_muni'], municipios)

        data.append([genero, int(i['edad']), int(i['Anio']), float(distancia)])
        
        estado = 1
        if i['Estado'].lower()=='traslado':
            estado = 0
        Y.append([estado])

    data = np.array(data)
    Y = np.array(Y)
    
    #Para el trainig utilizamos el 70% de los datos
    slice_point = int(len(data) * 0.7)
    #Para la validación utilizamos el 15%
    slice_point2 = int(len(data) * 0.85)

    train_X = data[0:slice_point].T
    train_Y = Y[0:slice_point].T
    val_X = data[slice_point:slice_point2].T
    val_Y = Y[slice_point:slice_point2].T
    test_X = data[slice_point2:].T
    test_Y = Y[slice_point2:].T

    return train_X, train_Y, val_X, val_Y, test_X, test_Y

def read_Municipios():
    results = []

    with open('./datasets/Municipios.csv') as File:
        reader = csv.DictReader(File)
        for row in reader:
            results.append(row)

    return results

def calc_distancia(depto, municipio, results):
    lat1 = 0
    lon1 = 0
    R = 6372.795477598
    lat2 = 14.589246
    lon2 = -90.551449
    
    for i in results:
        if int(i['Muni'])==int(municipio) and int(i['Depto'])==int(depto):
            lat1 = float(i['Lat'])
            lon1 = float(i['Lon'])
            break

    rad=math.pi/180
    dlat=lat2-lat1
    dlon=lon2-lon1
    a=(math.sin(rad*dlat/2))**2 + math.cos(rad*lat1)*math.cos(rad*lat2)*(math.sin(rad*dlon/2))**2
    distancia=2*R*math.asin(math.sqrt(a))
    
    return distancia