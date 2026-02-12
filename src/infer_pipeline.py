# src/infer_pipeline.py
import os
import tempfile
from TTS.api import TTS
from src.emotion_detector import detect_emotion
from src.style_mapper import StyleMapper
from src.voice_manager import get_voice_choice
from src.postprocess_audio import apply_prosody

def read_input(path_or_text: str):
    if os.path.exists(path_or_text):
        with open(path_or_text, 'r', encoding='utf-8') as f:
            return f.read()
    return path_or_text

def synthesize_text_to_emotional_speech(text_or_path: str, gender: str = 'female', accent: str = 'neutral', out_path: str = 'output/final_output.wav'):
    os.makedirs("output", exist_ok=True)
    text = read_input(text_or_path)

    print("Detecting emotion...")
    emotion_label = detect_emotion(text)
    print("Detected emotion:", emotion_label)

    mapper = StyleMapper()
    mapping = mapper.map(emotion_label, intensity=0.7)
    prosody = mapping['prosody']

    # Choose voice with gender + accent
    voice = get_voice_choice(gender, accent)
    model_name = voice['model']
    speaker_id = voice.get('speaker')

    # Initialize Coqui TTS with specific model
    print(f"Loading TTS model: {model_name}")
    tts = TTS(model_name=model_name)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_wav = os.path.join(tmp, "tts.wav")
        print(f"Synthesizing speech with {gender} {accent} voice...")
        
        # Use proper speaker ID for gender control
        tts.tts_to_file(
            text=text, 
            speaker=speaker_id, 
            file_path=tmp_wav
        )

        # Apply emotion prosody
        apply_prosody(tmp_wav, out_path, prosody)
        print(f"✅ Emotional speech saved at: {out_path}")

    return out_path
