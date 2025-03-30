import streamlit as st
import openai
import os
import tempfile
from gtts import gTTS
from dotenv import load_dotenv
from st_audiorec import st_audiorec
import PyPDF2


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OpenAI API key not found. Please set it in the environment.")
    st.stop()


client = openai.OpenAI(api_key=api_key)

st.title("ChatGPT Voice Bot (OpenAI Whisper API)")


def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        st.error(f"PDF file '{pdf_path}' not found.")
        return "Default knowledge base text."
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text if text else "Default knowledge base text."
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return "Default knowledge base text."

pdf_path = "knowledge_base.pdf"
knowledge_base = extract_text_from_pdf(pdf_path)

system_prompt = f"""
You are an AI designed to respond like me. Use the following information to shape your tone, style, and answers:
{knowledge_base}
Answer interview-style personal questions clearly and concisely, mimicking my personality and speech patterns. DO NOT answer anything else other than the knowledge in the knowledge base, just say, Sorry, I didn't get you
"""

st.write("Press the button below to record:")
audio_data = st_audiorec()

uploaded_audio = st.file_uploader("Or upload a pre-recorded audio file (WAV format)", type=["wav"])
if uploaded_audio:
    audio_data = uploaded_audio.read()

if audio_data is not None:
    if isinstance(audio_data, bytes):
        st.write(f"Audio data size (bytes): {len(audio_data)}")
        st.audio(audio_data, format="audio/wav")
    else:
        st.error("Audio data is not in the expected binary format.")
        st.stop()

    temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    response_audio_path = None 
    try:
       
        with open(temp_audio_path, "wb") as temp_audio:
            temp_audio.write(audio_data)

        
        file_size = os.path.getsize(temp_audio_path)
        st.write(f"Saved audio file size: {file_size} bytes")

        st.info("🔄 Converting speech to text using OpenAI Whisper API...")
        try:
            with open(temp_audio_path, "rb") as audio:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    language="en"
                )
            question = response.text
            st.write(f"Whisper API response: {response}")
            st.success(f"Your question: {question}")
        except Exception as e:
            st.error(f"Whisper API transcription failed: {e}")
            raise

     
        st.info("Generating response...")
        try:
            chat_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
            )
            answer = chat_response.choices[0].message.content
            st.success(f"ChatGPT Answer: {answer}")
        except Exception as e:
            st.error(f"ChatGPT response failed: {e}")
            raise

       
        response_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        try:
            tts = gTTS(answer, lang="en")
            tts.save(response_audio_path)
            with open(response_audio_path, "rb") as response_audio:
                st.audio(response_audio.read(), format="audio/mp3")
        except Exception as e:
            st.error(f"TTS generation failed: {e}")
            raise

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        for file_path in [temp_audio_path, response_audio_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    st.warning(f"Failed to delete file {file_path}: {e}")
