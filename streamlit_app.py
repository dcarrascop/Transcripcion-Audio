import openai
import streamlit as st
from pydub import AudioSegment
import os

# Configurar la clave de API desde los secrets de Streamlit
openai.api_key = st.secrets["openai_api_key"]

# Título de la app
st.title("Transcripción de Audio a Texto con Whisper")

# Cargar el archivo de audio
audio_file = st.file_uploader("Sube un archivo de audio", type=["mp3", "wav", "m4a"])

# Limpiar el estado si no hay archivo cargado
if audio_file is None:
    st.session_state["transcription_result"] = None

# Verificar si ya existe una transcripción previa en session_state
if "transcription_result" not in st.session_state:
    st.session_state["transcription_result"] = None

# Procesar el archivo de audio si se sube uno nuevo y no hay transcripción almacenada
if audio_file is not None and st.session_state["transcription_result"] is None:
    # Convertir el archivo subido a formato pydub AudioSegment
    audio = AudioSegment.from_file(audio_file)

    # Dividir el archivo en fragmentos de 24 minutos (menos de 25 MB, aprox)
    chunk_length_ms = 24 * 60 * 1000  # 24 minutos en milisegundos
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    # Guardar los fragmentos como archivos temporales
    fragment_paths = []
    for i, chunk in enumerate(chunks):
        fragment_path = f"fragmento_{i}.mp3"
        chunk.export(fragment_path, format="mp3")
        fragment_paths.append(fragment_path)

    # Mostrar barra de progreso
    progress_bar = st.progress(0)
    total_chunks = len(fragment_paths)

    # Transcribir cada fragmento
    all_transcriptions = []
    for i, fragment_path in enumerate(fragment_paths):
        with open(fragment_path, "rb") as audio_chunk:
            # Llamar a la API de Whisper para transcribir el fragmento
            transcription = openai.Audio.transcribe(model="whisper-1", file=audio_chunk)
            all_transcriptions.append(transcription['text'])
        
        # Actualizar la barra de progreso
        progress_percentage = (i + 1) / total_chunks
        progress_bar.progress(progress_percentage)

    # Combinar todas las transcripciones en un solo texto
    st.session_state["transcription_result"] = "\n".join(all_transcriptions)

    # Limpieza de archivos temporales (opcional)
    for fragment_path in fragment_paths:
        os.remove(fragment_path)

    st.success("Transcripción completada.")

# Mostrar la transcripción solo si ya fue procesada
if st.session_state["transcription_result"] is not None:
    st.text_area("Transcripción:", st.session_state["transcription_result"], height=300)

    # Opción para descargar la transcripción como archivo .txt
    st.download_button(
        label="Descargar transcripción como .txt",
        data=st.session_state["transcription_result"],
        file_name="transcripcion.txt",
        mime="text/plain"
    )
