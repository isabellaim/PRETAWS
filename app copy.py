# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper
import openai
from ultralytics import YOLO

# Función para detectar objetos
def detect_objects(image_path):
    # Cargar el modelo YOLOv5 preentrenado
    model = YOLO('/Users/isabella/Library/CloudStorage/OneDrive-EscuelaSuperiorPolitécnicadelLitoral/ESPOL/PAOII-2023/TAWS/avance.pt')
    # Ejecutar inferencia en la imagen de entrada
    results = model(image_path)
    # Procesar los resultados de la detección
    detected_objects = []
    for result in results:
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            conf = round(box.conf[0].item(), 2)
            detected_objects.append((class_id, conf))
    return detected_objects

# Función para buscar recetas
def buscarRecetas(ingredientes_busqueda, recetas):
    coincidencias_por_receta = []
    for receta, detalles in recetas.items():
        ingredientes_receta = detalles['ingredientes']
        coincidencias = len(set(ingredientes_busqueda) & set(ingredientes_receta))
        proporcion_coincidencias = coincidencias / len(ingredientes_busqueda) if len(ingredientes_busqueda) > 0 else 0
        coincidencias_por_receta.append((receta, proporcion_coincidencias))
    coincidencias_por_receta.sort(key=lambda x: x[1], reverse=True)
    return [receta for receta, _ in coincidencias_por_receta[:3]]

# Función para generar instrucciones
def generar_instrucciones(receta):
    complet = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Receta: {receta}. Por favor, proporciona instrucciones detalladas para completar esta receta:",
        max_tokens=100
    )
    return complet.choices[0].text.strip()


openai.api_key = "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"

# Configuración del cliente de OpenAI
#client = OpenAIApi(api_key="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")

# Diccionario de recetas
recetas = {
    'Arroz con Menestra y Carne': {'ingredientes': ['Verde', 'Carne', 'Tomate', 'Cebolla']},
    'Ensalada de Papa': {'ingredientes': ['Papa', 'Tomate', 'Cebolla']},
    'Pure con Carne': {'ingredientes': ['Papa', 'Carne']},
    'Bistec de Carne': {'ingredientes': ['Verde', 'Carne', 'Tomate', 'Cebolla']},
    'Tortilla de Papa': {'ingredientes': ['Papa', 'Carne', 'Cebolla']},
    'Empanadas de Verde': {'ingredientes': ['Verde', 'Carne', 'Cebolla']},
    'Seco de Pollo': {'ingredientes': ['Verde', 'Pollo', 'Cebolla', 'Tomate']},
    'Pollo Guisado': {'ingredientes': ['Tomate', 'Pollo', 'Cebolla']},
    'Sopa de Pollo': {'ingredientes': ['Pollo', 'Papa', 'Cebolla']},
    'Pollo Frito': {'ingredientes': ['Pollo']}
}


# Setting page layout
st.set_page_config(
    page_title="Clasificación de ingredientes utilizando YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.markdown(
    """
    <div style="text-align: center;">
        <h1>COOKFINDER</h1>
    </div>
    """,
    unsafe_allow_html=True
)
#st.title("CookFinder")

# Sidebar
#st.sidebar.header("Configuración del modelo")

# Model Options
#model_type = st.sidebar.radio(
    #"Escoja una tarea", ['Detection'])

# Configurar el color de fondo del sidebar
st.markdown(
    """
    <style>
        .sidebar {
            background-color: #E0FFFF;  /* Cambia este valor al color que desees */
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("images/logo.png", use_column_width=True)

# Selecting Detection
model_path = Path('/Users/isabella/Library/CloudStorage/OneDrive-EscuelaSuperiorPolitécnicadelLitoral/ESPOL/PAOII-2023/TAWS/avance.pt')

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"No se pudo cargae el modelo. Verifique el path indicado: {model_path}")
    st.error(ex)
# Initialize confidence (example value, adjust as needed)
confidence = 0.7

st.sidebar.header("¿Cómo desea detectar sus ingredientes?")
source_radio = st.sidebar.radio(
    "Escoja una fuente", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Escoja una imagen...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Imagen por default",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Imagen subida",
                         use_column_width=True)
        except Exception as ex:
            st.error("Ocurrió un error al abrir la imagen")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Imagen detectada',
                     use_column_width=True)
        else:
            if st.sidebar.button('Objetos detectados'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Imagen detectada',
                         use_column_width=True)
                """
                try:
                    with st.expander("Resultados de deteccción"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("¡No se ha subido ninguna imagen todavía!")
                """
    if source_img is not None:
        image = source_img.read()
        st.image(image, caption="Imagen de entrada", use_column_width=True)
        
        # Ejecutar la detección de objetos
        detected_objects = detect_objects(image)
        ingredientes_detectados = [obj[0] for obj in detected_objects]
        
        st.subheader("Objetos Detectados")
        for obj, conf in detected_objects:
            st.write(f"Tipo de Objeto: {obj}, Confianza: {conf}")

        # Buscar recetas basadas en los ingredientes detectados
        recetas_encontradas = buscarRecetas(ingredientes_detectados, recetas)
        if recetas_encontradas:
            st.subheader("Recetas Disponibles")
            for receta in recetas_encontradas:
                st.write(receta)
                instrucciones = generarInstrucciones(receta)
                st.write("Instrucciones:")
                st.write(instrucciones)
        else:
            st.subheader("No se encontraron recetas con los ingredientes detectados.")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

else:
    st.error("¡Seleccione un tipo de fuente válido!")
