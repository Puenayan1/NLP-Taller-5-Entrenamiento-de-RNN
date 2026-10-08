Markdown
# NLP Taller 5: Clasificación Bilingüe de Ansiedad con RNN (Bi-LSTM)

Este repositorio contiene la implementación, entrenamiento y optimización de una Red Neuronal Recurrente (RNN) diseñada para detectar rasgos de ansiedad en secuencias de texto. El modelo es bilingüe y procesa un corpus de mensajes tanto en inglés como en su traducción al español, utilizando una arquitectura de capas **LSTM Bidireccionales** sobre TensorFlow.

## Características Principales

* **Clasificación Bilingüe:** Unifica y procesa datos en inglés y español para una clasificación binaria (Ansiedad vs. Otros estados).
* **Arquitectura Bi-LSTM:** Analiza el contexto de las oraciones en ambas direcciones, capturando dependencias semánticas complejas.
* **Optimización de Hiperparámetros:** Integra `Keras Tuner` (Hyperband) para encontrar la mejor combinación de dimensiones de Embedding y unidades LSTM.
* **Ejecución Local Optimizada:** Configuración calibrada para ejecutar la búsqueda de hiperparámetros en CPU con tiempos de convergencia viables.

---

## Pipeline del Proyecto

El siguiente diagrama ilustra el flujo de ejecución (MLOps) y la arquitectura de la red neuronal implementada:

### Flujo de Ejecución (MLOps)
```mermaid
flowchart TD
    A[(Combined Data es-ES.csv)] --> B[Preprocesamiento y Concatenación Bilingüe]
    B --> C[Split Train/Test]
    C --> D{Keras Tuner: Hyperband\nBúsqueda Optimizada}
    D -->|Prueba combinaciones| E[Evaluación de Hiperparámetros]
    E --> F[Entrenamiento Modelo Final\ncon Mejor Arquitectura]
    F --> G([Evaluación Final: Test Loss / Accuracy])
# Arquitectura de la Red Neuronal
#Fragmento de código
graph TD
    In([Input: Texto Raw]) --> L1[TextVectorization\nMax Tokens: 10,000]
    L1 --> L2[Embedding Layer\nMasking Activo]
    L2 --> L3[Bidirectional LSTM\n32 o 64 Unidades]
    L3 --> L4[Dropout\nEstático: 0.3]
    L4 --> L5[Dense Layer\n64 Unidades, ReLU]
    L5 --> Out([Dense Salida\n1 Unidad, Logits])
# Tecnologías y Dependencias
El proyecto está configurado para un entorno Windows nativo utilizando la versión estable de CPU de TensorFlow para evitar conflictos con librerías dinámicas de aceleración gráfica externa.

* Python 3.10 / 3.11

* TensorFlow 2.15.0 (CPU)

* Keras Tuner

* Pandas, NumPy, Scikit-learn, Matplotlib

# Instalación y Configuración
Se recomienda utilizar un entorno virtual con Python 3.10 o 3.11 para garantizar la compatibilidad de los binarios.

# Clonar el repositorio:

Bash
git clone [https://github.com/Puenayan1/NLP-Taller-5-Entrenamiento-de-RNN.git](https://github.com/Puenayan1/NLP-Taller-5-Entrenamiento-de-RNN.git)
cd NLP-Taller-5-Entrenamiento-de-RNN
Crear y activar un entorno virtual (Windows):

PowerShell
py -3.10 -m venv env
.\env\Scripts\activate
Actualizar pip e instalar dependencias:

PowerShell
python -m pip install --upgrade pip
pip install tensorflow-cpu==2.15.0 keras-tuner pandas scikit-learn matplotlib
# Datos
El modelo se alimenta del archivo Combined Data es-ES.csv. El script de preprocesamiento extrae automáticamente las columnas statement (inglés) y statement_es_es (español), las concatena en un único array bilingüe y genera las etiquetas binarias basándose en la columna status (1 para Ansiedad, 0 para el resto).

Nota: Asegúrate de que el archivo CSV se encuentre en la raíz del directorio antes de ejecutar el código.

# Uso y Ejecución
Para iniciar el flujo completo (preprocesamiento, búsqueda de hiperparámetros con Keras Tuner, entrenamiento final y evaluación), ejecuta el script principal:

PowerShell
python Taller-5.py
Fases de ejecución:

## Hyperparameter Tuning: 
Keras Tuner sondeará las mejores combinaciones de embedding_dim y lstm_units durante 4 épocas usando un factor de eliminación agresivo (Fast Search).

## Entrenamiento Final: 
Una vez identificada la arquitectura óptima, el modelo se compilará y entrenará con el conjunto de datos completo.

## Evaluación: 
Se imprimirán en consola las métricas finales y se generarán las curvas de validación.