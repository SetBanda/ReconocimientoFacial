import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Sentia - IA de Emociones",
    page_icon="🧠",
    layout="centered"
)

# --- ESTILOS PERSONALIZADOS (CSS) ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #6c63ff;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #5149e6;
        color: white;
    }
    .emotion-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LÓGICA DE LA IA ---
@st.cache_resource
def cargar_modelo():
    # Asegúrate de que el nombre del archivo coincida con el tuyo
    return tf.keras.models.load_model('c:\\Users\\yasir\\Desktop\\Diego\\Visual Studio codigos\\proyecto_emociones\\modelo_emociones.keras')

modelo = cargar_modelo()

# Diccionario de emociones con Emojis y Colores relacionados
info_emociones = {
    'angry': {'nombre': 'Enojo', 'emoji': '😡', 'color': '#ff4b4b'},
    'disgust': {'nombre': 'Asco', 'emoji': '🤢', 'color': '#4baf51'},  # ESTA EMOCION TIENE EXPRECIONES QUE SE PUEDEN IDENTIFIACAR COMO OTRAS ASI QUE QUEDA EN DESCARTADA O ERROR DE LECTURA.
    'fear': {'nombre': 'Miedo', 'emoji': '😱', 'color': '#9c27b0'},
    'happy': {'nombre': 'Alegría', 'emoji': '😊', 'color': '#ffeb3b'},
    'neutral': {'nombre': 'Neutral', 'emoji': '😐', 'color': '#9e9e9e'},
    'sad': {'nombre': 'Tristeza', 'emoji': '😔', 'color': '#2196f3'},
    'surprise': {'nombre': 'Sorpresa', 'emoji': '😲', 'color': '#ff9800'}
}

# El orden debe ser el mismo que detectó tu modelo
clases_orden = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- INTERFAZ DE USUARIO ---
st.title("Sentia: Inteligencia Emocional Artificial 🎭")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Paso 1: Sube un rostro")
    archivo_subido = st.file_uploader("", type=["jpg", "png", "jpeg"])
    if archivo_subido:
        imagen = Image.open(archivo_subido).convert('L')
        st.image(imagen, caption="Imagen para analizar", use_container_width=True)

with col2:
    st.subheader("Paso 2: Análisis")
    if archivo_subido:
        if st.button("Descifrar Estado de Ánimo"):
            # Preparar imagen
            img_redimensionada = imagen.resize((125, 125))
            img_array = np.array(img_redimensionada)
            img_array = np.expand_dims(img_array, axis=[0, -1])

            st.write(f"Rango de píxeles: {img_array.min()} - {img_array.max()}")
            
            # Predicción
            with st.spinner('Escaneando micro-expresiones...'):
                prediccion = modelo.predict(img_array)
                indice = np.argmax(prediccion)
                clase_id = clases_orden[indice]
                certeza = np.max(prediccion) * 100
                
                res = info_emociones[clase_id]
            
            # Resultado visual
            # Cambiar el fondo de la página según la emoción detectada
            st.markdown(f"""<style>
                .stApp, .main, .block-container {{
                background-color: {res['color']} !important;
                }}
                </style>
                """, unsafe_allow_html=True
            )

            st.markdown(f"""
                <div class="emotion-card" style="border-left: 10px solid {res['color']};">
                    <h1 style="font-size: 80px; margin: 0;">{res['emoji']}</h1>
                    <h2 style="color: {res['color']};">{res['nombre']}</h2>
                    <p>Confianza del análisis: <b>{certeza:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Envolvemos el cálculo de NumPy dentro de float()
            certeza = float(np.max(prediccion)) * 100
    else:
        st.info("Esperando una imagen para comenzar el análisis...")

# --- PIE DE PÁGINA ---
st.markdown("---")
st.caption("Proyecto de Inteligencia Artificial - UAdeC Saltillo | Equipo: Zzz")