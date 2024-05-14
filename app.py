from flask import Flask, render_template, request
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

app = Flask(__name__, template_folder='Prueba')  

# Funciones de utilidad
def cargar_datos_desde_url(url):
    with urllib.request.urlopen(url) as response:
        datos = json.loads(response.read().decode('utf-8'))
    return datos

def preparar_datos_y_entrenar_modelo(datos_propiedades):
    X = []
    titulos = []
    for propiedad in datos_propiedades:
        caracteristicas = [float(propiedad['precio']), int(propiedad['habitaciones']), int(propiedad['banios']), int(propiedad['pisos'])]
        X.append(caracteristicas)
        titulos.append(propiedad['titulo'])
    X = np.array(X)
    n_vecinos = min(len(datos_propiedades) - 1, 4)
    modelo = NearestNeighbors(n_neighbors=n_vecinos)
    modelo.fit(X)
    return modelo, titulos

def recomendar_propiedades_similares(modelo, datos_propiedades, caracteristicas_usuario):
    X = []
    titulos = []
    for propiedad in datos_propiedades:
        caracteristicas = [float(propiedad['precio']), int(propiedad['habitaciones']), int(propiedad['banios']), int(propiedad['pisos'])]
        X.append(caracteristicas)
        titulos.append(propiedad['titulo'])
    X = np.array(X)
    distancias, indices = modelo.kneighbors([caracteristicas_usuario])
    propiedades_similares = []
    for indice in indices[0]:
        propiedades_similares.append(datos_propiedades[indice])
    return propiedades_similares

# Rutas de la aplicación
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/buscar', methods=['POST'])
def buscar():
    datos_propiedades = cargar_datos_desde_url('http://localhost/inmo/chat/Datos.php')
    precio = float(request.form['precio'])
    habitaciones = int(request.form['habitaciones'])
    banios = int(request.form['banios'])
    pisos = int(request.form['pisos'])
    
    caracteristicas_usuario = [precio, habitaciones, banios, pisos]
    modelo, titulos = preparar_datos_y_entrenar_modelo(datos_propiedades)
    propiedades_similares = recomendar_propiedades_similares(modelo, datos_propiedades, caracteristicas_usuario)
    
    # Calcular la regresión lineal para la primera característica (precio)
    X = np.array([[float(propiedad['precio'])] for propiedad in propiedades_similares])
    y = np.array([[int(propiedad['habitaciones']), int(propiedad['banios']), int(propiedad['pisos'])] for propiedad in propiedades_similares])
    regresion = LinearRegression().fit(X, y)
    predicciones = regresion.predict(X)
    
    # Generar y guardar la gráfica de recomendación con la recta de regresión
    plt.figure(figsize=(10, 8))
    plt.scatter(X, y[:, 0], c='b', label='Habitaciones')
    plt.scatter(X, y[:, 1], c='r', label='Baños')
    plt.scatter(X, y[:, 2], c='g', label='Pisos')
    
    plt.plot(X, predicciones[:, 0], c='b', label='Regresión Habitaciones')
    plt.plot(X, predicciones[:, 1], c='r', label='Regresión Baños')
    plt.plot(X, predicciones[:, 2], c='g', label='Regresión Pisos')
    
    plt.xlabel('Precio')
    plt.ylabel('Características')
    plt.title('Relación entre Precio y Características de Propiedades Similares')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('grafica_recomendacion_regresions.png')  # Guardar la gráfica en el directorio del script
    
    return render_template('resultado.html', propiedades=propiedades_similares)

if __name__ == "__main__":
    app.run(debug=True)
