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

if audio_file is not None:
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

    # Mostrar mensaje de progreso
    st.write("Transcribiendo el audio...")

    # Transcribir cada fragmento
    all_transcriptions = []
    for fragment_path in fragment_paths:
        with open(fragment_path, "rb") as audio_chunk:
            # Llamar a la API de Whisper para transcribir el fragmento
            transcription = openai.Audio.transcribe(model="whisper-1", file=audio_chunk)
            all_transcriptions.append(transcription['text'])

    # Combinar todas las transcripciones en un solo texto
    complete_transcription = "\n".join(all_transcriptions)

    # Mostrar la transcripción en la aplicación
    st.text_area("Transcripción:", complete_transcription, height=300)

    # Opción para descargar la transcripción como archivo .txt
    st.download_button(
        label="Descargar transcripción como .txt",
        data=complete_transcription,
        file_name="transcripcion.txt",
        mime="text/plain"
    )

    # Limpieza de archivos temporales (opcional)
    for fragment_path in fragment_paths:
        os.remove(fragment_path)
