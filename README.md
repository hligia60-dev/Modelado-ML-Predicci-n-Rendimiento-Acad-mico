# 🎓 Sistema Inteligente de Analítica & Rendimiento Académico

Proyecto de analítica predictiva y modelado prescriptivo enfocado en el aprendizaje supervisado, aplicando metodologías **CRISP-ML** sobre métricas pedagógicas del entorno universitario.

---

## 🤵 Autoría y Créditos
* **Desarrollador / Analista de IA:** Ligia Elena Herrera Frías
* **Programa:** Maestría en Inteligencia Artificial orientada a la Educación
* **Institución:** Universidad Federico Henríquez y Carvajal (UFHEC)
* **Año:** 2026

---

## 🚀 Características Principales del Dashboard

* **🔮 Predictor Inteligente en Tiempo Real:** Interfaz dual que ejecuta de forma concurrente algoritmos de regresión y clasificación supervisada.
  * **Regresión Lineal:** Estima la nota cuantitativa final esperada del estudiante (0-100 pts).
  * **Regresión Logística:** Clasifica y calcula la probabilidad de riesgo de rezago escolar o reprobación.
* **📊 Analítica Histórica:** Módulo avanzado de visualización interactiva de datos (N = 10,000 registros) que incluye matrices de correlación e histogramas de distribución poblacional tridimensionales utilizando Plotly.
* **🎯 Simulador Prescriptivo de Metas:** Motor de simulación inversa que toma una calificación meta fijada por el usuario y calcula matemáticamente las horas de estudio, asistencia o tareas requeridas en función de los coeficientes del modelo.
* **📁 Repositorio Dinámico (Assets):** Integración en la barra lateral para detectar de forma automática y desplegar recursos visuales, imágenes de soporte o diagramas alojados en el directorio local.

---

## 🛠️ Requisitos del Entorno y Arquitectura

El sistema requiere las siguientes dependencias clave para su correcto funcionamiento (asegúrese de tenerlas mapeadas en su archivo `requirements.txt`):
* `streamlit`
* `pandas`
* `numpy`
* `joblib`
* `plotly`
* `scikit-learn`

---

## 📦 Estructura del Repositorio

Para garantizar que la aplicación web se ejecute de forma óptima en **Antigravity**, la raíz de tu proyecto debe mantener la siguiente estructura de archivos:

```text
📁 tu-repositorio/
│
├── 📄 app.py                           # Archivo principal de la aplicación Streamlit
├── 📄 README.md                        # Documentación técnica del proyecto (Este archivo)
├── 📄 requirements.txt                 # Dependencias del entorno de Python
├── 📄 student_dataset_10000_rows.csv   # Base de datos histórica (Ecosistema estudiantil)
├── 📄 modelo_lineal_ufhec.pkl          # Serialización del modelo de Regresión Lineal (.joblib)
├── 📄 modelo_logistica_ufhec.pkl       # Serialización del modelo de Regresión Logística (.joblib)
└── 📁 Assets/                          # Carpeta contenedora de imágenes (.png, .jpg) para el visor dinámico
   
