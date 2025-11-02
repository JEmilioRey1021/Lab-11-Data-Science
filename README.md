# 🧭 Laboratorio 11 – Visualizaciones Interactivas y Dashboards  
**Curso:** CC3066 – Data Science  
**Universidad del Valle de Guatemala**  
**Semestre II – 2025**  

## 👩‍💻 Integrantes
- **José Emilio Reyes Paniagua – 22674**  
- **Michelle Angel de María Mejía Villela – 22596**  
- **Silvia Alejandra Illescas Fernández – 22376**

---

## 📋 Descripción general
Este proyecto desarrolla un **dashboard interactivo** en **Python con Plotly Dash** para analizar los datos de **Consumo e Importación de combustibles** en Guatemala.  
La aplicación permite explorar la evolución temporal, relaciones entre variables y modelos predictivos para estimar tendencias futuras.

Se diseñó con un **enfoque analítico y estético**, aplicando principios de **UX**, **teoría del color** y **visualización efectiva de datos**.

---

## 🧠 Objetivos
- Permitir al usuario explorar los datos de forma interactiva y flexible.  
- Mostrar resultados de **tres modelos predictivos**:  
  1. Regresión Lineal  
  2. Regresión Polinómica (grado 2)  
  3. Random Forest Regressor  
- Comparar el desempeño de los modelos mediante métricas (MAE y R²).  
- Implementar visualizaciones enlazadas y control de granularidad (Mensual, Trimestral, Anual).  
- Diseñar un tablero profesional con una **paleta pastel coherente** y un estilo limpio.

---

## 🧩 Características principales

### 🔍 Interactividad
- Filtros por **fuente** (Consumo o Importación).  
- Selector de **combustible**.  
- **Rango de fechas** dinámico con RangeSlider.  
- Selector de **nivel de agregación temporal** (M / Q / A).  
- **Checklist** para mostrar/ocultar visualizaciones.  
- Enlace entre **boxplot → serie temporal y tendencia**.

### 📈 Visualizaciones incluidas
1. Serie temporal de consumo/importación.  
2. Tendencia (media móvil 12 meses).  
3. Relación Consumo vs Importación (scatter + regresión lineal).  
4. Boxplot de distribución mensual.  
5. Importación promedio por mes.  
6. Distribución del consumo por combustible (pie chart).  
7. Predicciones de 3 modelos (líneas superpuestas).  
8. Comparativa de desempeño (barras MAE y R²).  
9. Tabla comparativa de métricas (MAE y R²).

---

## 🧮 Modelos predictivos implementados
Los modelos estiman el **consumo de cada combustible** en función de la **importación mensual**, entrenados y validados dentro del rango de fechas seleccionado.

| Modelo | Descripción | MAE ↓ | R² ↑ |
|:-------|:-------------|:------:|:----:|
| **Regresión Lineal** | Ajuste simple entre importación y consumo | Medio | 0.74 |
| **Regresión Polinómica (g2)** | Captura relaciones cuadráticas | Bajo | 0.82 |
| **Random Forest Regressor** | Ensamble no lineal robusto | **Muy bajo** | **0.91** |

El **Random Forest** presentó el mejor desempeño general.

---

## 🎨 Diseño y paleta de colores
Se aplicó una paleta **pastel cálida y armónica**, basada en fondos crema y acentos suaves:

| Color | Hex | Uso |
|:------|:----|:----|
| Crema | `#F7F3DF` | Fondo principal |
| Coral | `#ECA07D` | Acentos / líneas de tendencia |
| Amarillo | `#F6F07A` | Controles interactivos |
| Verde menta | `#B9EE93` | Indicadores / tooltips |
| Azul cielo | `#9EC1E6` | Series principales |
| Tinta | `#1F2937` | Texto principal |

La tipografía elegida fue **Inter**, optimizada para visualización en pantallas.

---

## ⚙️ Instalación y ejecución

### Requisitos
- Python ≥ 3.10  
- Dependencias:

    ```bash
  pip install dash plotly pandas scikit-learn dash-bootstrap-components

Ejecución

Coloca los archivos Consumo.xlsx, Importacion.xlsx y app.py en la misma carpeta.

Ejecuta en terminal:

python app.py


Abre el navegador en:
👉 http://127.0.0.1:8050/

📦 Estructura del repositorio
Lab-11-Data-Science/
│
├── app.py                       # Código principal del dashboard
├── Consumo.xlsx                 # Dataset de consumo
├── Importacion.xlsx             # Dataset de importación
├── Documento de Preparación.pdf # Bosquejo, paleta y planificación
├── README.md                    # (este archivo)
└── /assets                      # (opcional) estilos o recursos extra

🚀 Resultados destacados

Dashboard completamente funcional e intuitivo.

Cumplimiento de todos los requisitos del laboratorio:

≥ 8 visualizaciones interactivas.

3 modelos predictivos simples.

Visualizaciones enlazadas.

Control de granularidad.

Diseño UX profesional.

