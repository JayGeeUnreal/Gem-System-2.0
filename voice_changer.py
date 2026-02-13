import collections
import queue
import sys
import wave
import os
import io
import configparser
import requests  # To talk to your server
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from scipy.io.wavfile import read as read_wav

# --- Configuration ---
# MAKE SURE THIS MATCHES YOUR settings.ini [Server] section
TTS_SERVER_URL = "http://127.0.0.1:13300/tts" 

# Audio Settings
VAD_AGGRESSIVENESS = 3
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 20
FRAMES_PER_BUFFER = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
SILENCE_THRESHOLD_S = 1.0 
PRE_BUFFER_S = 0.5
WAV_FILE_NAME = "temp_client_input.wav"

# --- Global State ---
audio_queue = queue.Queue()
is_recording = False
pre_buffer_frames = collections.deque(maxlen=int(PRE_BUFFER_S * SAMPLE_RATE / FRAMES_PER_BUFFER))
recorded_frames = []
silent_frames_count = 0

# --- HELPER: Load Microphone Settings ---
def load_audio_settings():
    ini_path = 'mcp_settings.ini'
    settings = {'device_id': None}
    
    if not os.path.exists(ini_path):
        return settings
        
    config = configparser.ConfigParser()
    config.read(ini_path)
    try:
        if config.has_section('Audio') and config.has_option('Audio', 'selected_input'):
            device_string = config.get('Audio', 'selected_input')
            if device_string and device_string.lower() != 'none':
                if device_string.startswith('[') and ']' in device_string:
                    device_id_str = device_string.split(']')[0][1:]
                    settings['device_id'] = int(device_id_str)
    except:
        pass
    return settings

def audio_callback(indata, frames, time, status):
    if status: print(f"Audio status: {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))

def main():
    global is_recording, recorded_frames, silent_frames_count

    # 1. Setup Microphone
    settings = load_audio_settings()
    selected_device = settings['device_id']
    
    print("\n--- Initializing Whisper (Ears) ---")
    try:
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        whisper_name = "openai/whisper-base.en"
        processor = WhisperProcessor.from_pretrained(whisper_name)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_name)
        print("--- Whisper Loaded ---")
    except Exception as e:
        sys.exit(f"Error loading Whisper: {e}")

    # Check Server Connection
    print(f"Checking connection to Voice Server at {TTS_SERVER_URL}...")
    try:
        # We send a dummy request just to see if it connects
        requests.get(TTS_SERVER_URL.replace("/tts", "/"), timeout=2) 
        # Note: Your server might 404 on root, but if it connects, we are good.
        # If it fails completely, the except block catches it.
    except:
        print("WARNING: Could not verify server connection instantly. Make sure server_voice.py is running!")

    max_silent_frames = int(SILENCE_THRESHOLD_S * SAMPLE_RATE / FRAMES_PER_BUFFER)

    if selected_device is not None:
        try:
            print(f"Microphone: {sd.query_devices(selected_device)['name']}")
        except:
            print("Microphone: Default")
    else:
        print("Microphone: System Default")

    print("\n=== VOICE CHANGER READY ===")
    print("Speak naturally. The AI will repeat it in the target voice.")

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAMES_PER_BUFFER, 
                           device=selected_device, dtype="int16", channels=1, 
                           callback=audio_callback):
        while True:
            frame = audio_queue.get()
            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if is_recording:
                recorded_frames.append(frame)
                if not is_speech:
                    silent_frames_count += 1
                    if silent_frames_count > max_silent_frames:
                        print("\nProcessing...")
                        
                        # 1. Save temp input
                        all_frames = list(pre_buffer_frames) + recorded_frames
                        with wave.open(WAV_FILE_NAME, 'wb') as wf:
                            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
                            wf.writeframes(b''.join(all_frames))
                        
                        # 2. Transcribe
                        transcribed_text = ""
                        try:
                            sr_rate, audio_data = read_wav(WAV_FILE_NAME)
                            audio_data = audio_data.astype(np.float32) / 32768.0
                            input_features = processor(audio_data, sampling_rate=sr_rate, return_tensors="pt").input_features
                            predicted_ids = whisper_model.generate(input_features)
                            transcribed_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                        except Exception as e:
                            print(f"Whisper Error: {e}")
                        
                        if transcribed_text and len(transcribed_text) > 1:
                            print(f"You said: '{transcribed_text}'")
                            print("Sending to Voice Server...")
                            
                            # 3. Send to Server
                            try:
                                response = requests.post(
                                    TTS_SERVER_URL, 
                                    json={'chatmessage': transcribed_text},
                                    timeout=30 # Wait up to 30s for generation
                                )
                                
                                if response.status_code == 200:
                                    # 4. Play Response
                                    print("Playing generated voice...")
                                    # Convert bytes to audio data
                                    data, fs = sf.read(io.BytesIO(response.content))
                                    sd.play(data, samplerate=fs)
                                    sd.wait() # Block microphone while playing
                                    print("Done.")
                                else:
                                    print(f"Server Error: {response.text}")
                                    
                            except requests.exceptions.ConnectionError:
                                print("ERROR: Could not connect to Voice Server. Is it running?")
                            except Exception as e:
                                print(f"Playback Error: {e}")

                        # Reset
                        is_recording = False
                        recorded_frames.clear()
                        pre_buffer_frames.clear()
                        silent_frames_count = 0
                        print("\nListening...")
                else:
                    silent_frames_count = 0
            else:
                pre_buffer_frames.append(frame)
                if is_speech:
                    is_recording = True
                    recorded_frames.extend(pre_buffer_frames)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")