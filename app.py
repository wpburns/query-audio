import os
from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
import openai
from googletrans import Translator
from pytube import YouTube
from tempfile import NamedTemporaryFile
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import ui_utils
import shutil

load_dotenv()

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI()

def save_txt_file(filename, text):
    if not os.path.exists('transcriptions'):
        os.makedirs("transcriptions")

    saved_file = os.path.join("transcriptions", filename + ".txt")
    with open(saved_file, "w") as f:
        f.write(text)

def split_transcription(text):
    out = []
    threshold = 500
    for chunk in text.split('. '):
        if out and len(chunk)+len(out[-1]) < threshold:
            out[-1] += ' '+chunk+'.'
        else:
            out.append(chunk+'.')
    return out

def download_youtube_audio(youtube_url):
    st.info("Downloading Youtube Audio")

    yt = YouTube(youtube_url)

    audio = yt.streams.filter(only_audio=True).first()

    out_file = audio.download(output_path="audio_data")

    base, ext = os.path.splitext(out_file)
    new_file = base + ".mp3"
    os.rename(out_file, new_file)

    print("Downloading Youtube:" + new_file)
    return new_file

def transcribe_audio(audio_file, source_language):
    '''
    Googles Speech-to-Text algorithm claims to be able to identify and seperate transcriptions
    for differeent speakers within an audio sample.
    https://cloud.google.com/speech-to-text/docs/multiple-voices#speech_transcribe_diarization_beta-python
    '''
    st.info("Transcribing Audio")
    with open(audio_file, "rb") as audio:
        transcript = openai.Audio.transcribe(
            file = audio,
            model = "whisper-1",
            response_format = "text",
            language = source_language
        )

    # Causing issues on Windows writing Russian charachters
    # save_txt_file("source_"+  source_language, transcript)

    return transcript

def translate(text, dest_language):
    st.info( "Translating Transcription")
    translator = Translator()

    if len(text) > 1000:
        chunked_translation = ""
        text_chunks = split_transcription(text)
        for chunk in text_chunks:
            chunked_translation = chunked_translation + ".\n" + translator.translate(chunk, dest_language).text
        
        save_txt_file("translation_"+dest_language, chunked_translation)
        return chunked_translation
    else:
        translation = translator.translate(str(text), dest_language)
        save_txt_file("translation_"+dest_language, translation.text)
        return translation.text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def embed_text(text_chunks):
    st.info( "Creating Embeddings")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # vectorstore.save_local("FAISS_audio")
    st.success("Complete")
    return vectorstore

def process(type, data, source_language, dest_language):
    if type=="youtube":
        audio_file = download_youtube_audio(data)
    else:
        audio_file = data

    transcript = transcribe_audio(audio_file, source_language)
    translation = translate(transcript, dest_language)
    embed_translated_text = get_text_chunks(translation)
    vectorstore = embed_text(embed_translated_text)

    return transcript, translation, vectorstore

data_type = "audio"

def save_uploadedfile(uploadedfile):
     with open(os.path.join("audio_data",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return os.path.join("audio_data",uploadedfile.name)

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.info(message.content)
        else:
            st.success(message.content)

def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    st.set_page_config(
        page_title="Query Audio"
    )

    # Hide Sidebar
    # ui_utils.hide_sidebar()
    # Hide Footer
    ui_utils.hide_footer()
    
    if "progress" not in st.session_state:
        st.session_state["progress"] = "Progress"
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.image("shift_logo_white.png", width=400)
    st.header("Query Audio/Video Data")

    with st.expander("Demo Description", True):
        st.write('''Uses various APIs and Tools to extract, transcribe, translate and understand information from audio files, or YouTube videos.
                 https://www.youtube.com/watch?v=lnxCZr1BySc

                 ''')
        
    st.divider()

    with st.container():
        audio_format = st.selectbox("Select Audio Source", ["Audio File", "Youtube Link"])

        if audio_format == "Audio File":
            data_type = "audio"
            audio_file = st.file_uploader(
                "Upload audio file", 
                accept_multiple_files=False, 
                type=[
                    'mp3'
                ]
            )
            if audio_file is not None:
                source_file = save_uploadedfile(audio_file)
        elif audio_format == "Youtube Link":
            data_type = "youtube"
            source_file = st.text_input("Youtube Link...")

        with st.container():
            lang_1, lang_2 = st.columns(2)
            with lang_1:
                source_language = st.selectbox("Select Audio Source Language", ["ru", "en", "fr", "de"])
            with lang_2:
                dest_language = st.selectbox("Translate to...", ["en", "ru", "fr", "de"])
        
        if st.button("Analyse"):
            with st.container():
                with st.spinner("Processing"):
                    orig_transcript, translation, vectorstore = process(data_type, source_file, source_language, dest_language)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    with st.expander("Transcriptions"):
                        with st.container():
                            # Causing issues on Windows writing Russian charachters
                            c1, c2 = st.columns(2) 
                            with c1:
                                st.text_area(label="Original Transcript", value=orig_transcript, height=600)
                            with c2:
                                st.text_area(label="Translation", value=translation, height=600)
                            shutil.rmtree('audio_data') # Remove audio data, so as not to raise any conflicts.
                            shutil.rmtree('transcriptions') # Remove transcriptions, so as not to raise any conflicts.

        st.divider()
        with st.container():
            user_question = st.text_input("Ask a question about your audio/video")
            query_btn = st.button("Query")
            if query_btn:
                with st.spinner("..."):
                    handle_userinput(user_question)


if __name__ == "__main__":
    main()