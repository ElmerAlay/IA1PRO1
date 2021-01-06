from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import  secure_filename
from Util import ReadFile
from Neural_Network import Data
from Util.ReadFile import get_dataFile
from Util import Plotter
from Neural_Network.Data import Data
from Neural_Network.Model import NN_Model
from genetico import Genetico
from models import Model

import os
import numpy as np
import csv

nn2 = Model()
app = Flask(__name__)

#Inicializar Sesión
app.secret_key = 'mysecretkey'

@app.route('/')
def index():
    results = []

    with open('./datasets/Dataset.csv') as File:
        reader = csv.DictReader(File)
        for row in reader:
            results.append(row)

    deptos = []
    munis = []
    for i in results:
        flag = True 
        for j in deptos:
            if j['nombre']==i['nombre']:
                flag = False
        if flag==True:
            deptos.append({'cod':i['cod_depto'], 'nombre':i['nombre']})
        
        flag = True
        for j in munis:
            if j['nombre']==i['municipio']:
                flag = False
        if flag==True:
            munis.append({'cod':i['cod_muni'], 'nombre':i['municipio']})

    return render_template('index.html', deptos=deptos, munis=munis)

@app.route('/training', methods=['POST'])
def training():
    if request.method == 'POST':
        train_X, train_Y, val_X, val_Y, test_X, test_Y = get_dataFile() 

        train_set = Data(train_X, train_Y)
        val_set = Data(val_X, val_Y)
        test_set = Data(test_X, test_Y)

        capas1 = [train_set.n, 9, 6, 3, 1]

        gene = Genetico()
        hiper = gene.algoritmo()

        nn2.nn2 = NN_Model(train_set, capas1, alpha=hiper[0], iterations=hiper[2], lambd=hiper[1], keep_prob=hiper[3])
        #nn2.nn2 = NN_Model(train_set, capas1, alpha=0.005, iterations=100, lambd=1.5, keep_prob=0.8)

        nn2.nn2.training(False)

        nn2.nn2.predict(train_set)
        nn2.nn2.predict(val_set)
        nn2.nn2.predict(test_set)

        return redirect(url_for('index'))

@app.route('/predecir', methods=['POST'])
def predecir():
    if request.method == 'POST':
        genero = request.form['genero']
        anio = request.form['anio']
        depto = request.form['depto']
        muni = request.form['muni']
        edad = request.form['edad']

        print(depto, muni)
        results = ReadFile.read_Municipios()
        distancia = ReadFile.calc_distancia(int(depto), int(muni), results)

        val_X = np.array([[int(genero),int(edad),int(anio),float(distancia)]]).T
        val_Y = np.array([[1]]).T
        val_set = Data(val_X, val_Y)

        exactitud, resultado = nn2.nn2.predict(val_set)
        if resultado==1:
            flash("Se quedará")
        else:
            flash("Se trasladará")

        return redirect(url_for('index'))

@app.route('/show_graphic', methods=['POST'])
def show_graphic():
    if request.method=='POST':
        Plotter.show_Model([nn2.nn2])

        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(port=5000, debug=True)