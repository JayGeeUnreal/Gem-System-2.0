# ==============================================================================
#                      Master Control Program (mcp.py)
#          - UNIFIED MULTIMODAL & PIPELINED ARCHITECTURE -
# ==============================================================================
# This script acts as the central brain for the AI system. It includes:
# - A unified multimodal mode ('ollama_vision') for models like Llava/Gemma.
# - Pipelined modes ('ollama', 'gemini') for text-only LLMs.
# - A THREAD-SAFE unified RAG( ChromaDB) memory system.
# - Location awareness, OSC command bypass, multi-platform broadcasting & more.
# - Music recognition with selectable input device and configurable triggers.
# - Music downloader with configurable max duration and immediate feedback.
# ==============================================================================

import requests
import json
import configparser
import sys
import os
import platform
import google.generativeai as genai
import re
import datetime
import pytz # For timezone-aware datetimes
import chromadb
from collections import deque
from pythonosc import udp_client, osc_message_builder
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import threading

from flask import Flask, request, jsonify
from flask_cors import CORS

import subprocess
import traceback
import queue
import yt_dlp # For checking song duration
# ------------------------------------------

import sounddevice as sd
import soundfile as sf
# ------------------------------------------

# --- 1. CONFIGURATION LOADING ---
# ------------------------------------------------------------------------------
def load_config():
    """Loads all settings from the mcp_settings.ini file."""
    global MUSIC_RECOGNITION_ENABLED, MUSIC_RECOGNITION_SETTINGS

    config_file = 'mcp_settings.ini'
    config_parser = configparser.ConfigParser(interpolation=None)
    if not os.path.exists(config_file): sys.exit(f"FATAL ERROR: Config file '{config_file}' not found.")
    config_parser.read(config_file)
    settings = {}
    try:
        settings['system_prompt'] = config_parser.get('SystemPrompt', 'prompt', fallback='').strip()
        settings['llm_choice'] = config_parser.get('MCP', 'llm_choice')
        settings['host'] = config_parser.get('MCP', 'host')
        settings['port'] = config_parser.getint('MCP', 'port')
        settings['max_response_length'] = config_parser.getint('Assistant', 'max_response_length', fallback=0)
        raw_wake_words = config_parser.get('Assistant', 'wake_words', fallback='')
        settings['wake_words'] = [word.strip().lower() for word in raw_wake_words.split(',') if word.strip()]
        raw_command_verbs = config_parser.get('Assistant', 'command_verbs', fallback='')
        settings['command_verbs'] = [verb.strip().lower() for verb in raw_command_verbs.split(',') if verb.strip()]
        settings['vision_service_scan_url'] = config_parser.get('VisionService', 'scan_url')
        settings['vision_service_get_image_url'] = config_parser.get('VisionService', 'vision_service_get_image_url', fallback='')
        raw_triggers = config_parser.get('VisionService', 'vision_trigger_words', fallback='')
        settings['vision_trigger_words'] = [word.strip().lower() for word in raw_triggers.split(',') if word.strip()]
        settings['social_stream_enabled'] = config_parser.getboolean('SocialStream', 'enabled', fallback=False)
        settings['social_stream_session_id'] = config_parser.get('SocialStream', 'session_id')
        raw_platforms = config_parser.get('SocialStream', 'target_platforms', fallback='')
        settings['social_stream_targets'] = [p.strip() for p in raw_platforms.split(',') if p.strip()]
        settings['social_stream_api_url'] = config_parser.get('SocialStream', 'api_url')
        settings['styletts_enabled'] = config_parser.getboolean('StyleTTS', 'enabled', fallback=False)
        settings['styletts_url'] = config_parser.get('StyleTTS', 'tts_url')
        settings['gemini_api_key'] = config_parser.get('Gemini', 'api_key')
        settings['gemini_model'] = config_parser.get('Gemini', 'model')
        settings['ollama_model'] = config_parser.get('Ollama', 'model')
        settings['ollama_vision_model'] = config_parser.get('Ollama', 'vision_model', fallback='')
        settings['ollama_embedding_model'] = config_parser.get('Ollama', 'embedding_model', fallback='')
        settings['ollama_api_url'] = config_parser.get('Ollama', 'api_url')
        settings['osc_enabled'] = config_parser.getboolean('OSC', 'enabled', fallback=False)
        settings['osc_ip'] = config_parser.get('OSC', 'ip')
        settings['osc_port'] = config_parser.getint('OSC', 'port')
        settings['osc_address'] = config_parser.get('OSC', 'address')
        raw_osc_verbs = config_parser.get('OSC', 'trigger_verbs', fallback='')
        settings['osc_trigger_verbs'] = [verb.strip().lower() for verb in raw_osc_verbs.split(',') if verb.strip()]
        raw_rag_triggers = config_parser.get('RAG', 'rag_trigger_words', fallback='')
        settings['rag_trigger_words'] = [trigger.strip().lower() for trigger in raw_rag_triggers.split(',') if trigger.strip()]
        
        settings['audio_selected_input_raw'] = config_parser.get('Audio', 'selected_input', fallback='').strip()

        if config_parser.has_section('MusicRecognition'):
            settings['music_recognition_enabled'] = config_parser.getboolean('MusicRecognition', 'enabled', fallback=False)
            if settings['music_recognition_enabled']:
                MUSIC_RECOGNITION_ENABLED = True
                
                MUSIC_RECOGNITION_SETTINGS['rapidapi_key'] = config_parser.get('MusicRecognition', 'rapidapi_key', fallback='').strip()
                MUSIC_RECOGNITION_SETTINGS['rapidapi_host'] = config_parser.get('MusicRecognition', 'rapidapi_host', fallback='').strip()
                MUSIC_RECOGNITION_SETTINGS['recognition_endpoint_url'] = config_parser.get('MusicRecognition', 'recognition_endpoint_url', fallback='').strip()
                MUSIC_RECOGNITION_SETTINGS['audio_duration'] = config_parser.getint('MusicRecognition', 'audio_duration', fallback=8)
                MUSIC_RECOGNITION_SETTINGS['sample_rate'] = config_parser.getint('MusicRecognition', 'sample_rate', fallback=44100)
                MUSIC_RECOGNITION_SETTINGS['channels'] = config_parser.getint('MusicRecognition', 'channels', fallback=1)
                MUSIC_RECOGNITION_SETTINGS['temp_audio_filename'] = config_parser.get('MusicRecognition', 'temp_audio_filename', fallback='temp_recognition_clip.wav')

                raw_music_triggers = config_parser.get('MusicRecognition', 'trigger_words', fallback='what song is this,identify this music,what is playing')
                settings['music_trigger_words'] = [word.strip().lower() for word in raw_music_triggers.split(',') if word.strip()]

                if not all([MUSIC_RECOGNITION_SETTINGS['rapidapi_key'],
                            MUSIC_RECOGNITION_SETTINGS['rapidapi_host'],
                            MUSIC_RECOGNITION_SETTINGS['recognition_endpoint_url']]):
                    print("MCP WARNING: MusicRecognition is enabled, but essential API settings are missing. Disabling.")
                    MUSIC_RECOGNITION_ENABLED = False
            else:
                MUSIC_RECOGNITION_ENABLED = False
        else:
            settings['music_recognition_enabled'] = False
            MUSIC_RECOGNITION_ENABLED = False
        
        # --- THIS BLOCK LOADS ALL MUSIC DOWNLOADER SETTINGS ---
        if config_parser.has_section('MusicDownloader'):
            raw_download_triggers = config_parser.get('MusicDownloader', 'trigger_words', fallback='')
            settings['download_trigger_words'] = [word.strip().lower() for word in raw_download_triggers.split(',') if word.strip()]
            settings['music_downloader_enabled'] = config_parser.getboolean('MusicDownloader', 'enabled', fallback=False)
            settings['max_download_duration_seconds'] = config_parser.getint('MusicDownloader', 'max_download_duration_seconds', fallback=600)
        else:
            settings['download_trigger_words'] = []
            settings['music_downloader_enabled'] = False
            settings['max_download_duration_seconds'] = 600
        # ----------------------------------------------------

    except Exception as e:
        sys.exit(f"FATAL ERROR: A setting is missing or invalid in '{config_file}'. Details: {e}")
    return settings
# ------------------------------------------------------------------------------


# --- Standalone function for device selection for music recognition---
def select_audio_device():
    """
    Selects an audio input device by parsing the '[Audio] selected_input'
    setting from the mcp_settings.ini file. Falls back to default if needed.
    """
    print("\n" + "-"*70)
    print("--- Configuring Audio Device for Music Recognition ---")

    configured_device_str = config.get('audio_selected_input_raw', '')

    try:
        devices = sd.query_devices()
        input_devices = {d['index']: d for d in devices if d['max_input_channels'] > 0}

        if not input_devices:
            print("MCP ERROR: No audio input devices found.")
            return None

        if configured_device_str:
            match = re.match(r'\[(\d+)\]', configured_device_str)
            if match:
                configured_index = int(match.group(1))
                if configured_index in input_devices:
                    device_name = input_devices[configured_index]['name']
                    print(f"MCP INFO: Using configured audio device from settings: Index {configured_index} ('{device_name}')")
                    return configured_index
                else:
                    print(f"MCP WARNING: Configured audio device index '{configured_index}' is not a valid input device. Falling back to default.")
            else:
                print(f"MCP WARNING: Could not parse device index from setting '{configured_device_str}'. Falling back to default.")

        default_idx = sd.default.device[0]
        if default_idx in input_devices:
            device_name = input_devices[default_idx]['name']
            print(f"MCP INFO: No valid device specified in settings. Using system default: Index {default_idx} ('{device_name}')")
            return default_idx

        first_available_idx = next(iter(input_devices))
        device_name = input_devices[first_available_idx]['name']
        print(f"MCP WARNING: System default device is not a valid input. Using first available device: Index {first_available_idx} ('{device_name}')")
        return first_available_idx

    except Exception as e:
        print(f"MCP ERROR: Could not query or configure audio devices. Details: {e}")
        return None
# ------------------------------------------------------------------------------


# --- 2. INITIALIZATION ---
# ------------------------------------------------------------------------------
MUSIC_RECOGNITION_ENABLED = False
MUSIC_RECOGNITION_SETTINGS = {}

config = load_config()
app = Flask(__name__)
CORS(app)
gemini_model = None

VISION_HISTORY = deque(maxlen=5)
CURRENT_LOCATION = "the stream room"
SELECTED_INPUT_DEVICE_INDEX = None
download_queue = queue.Queue()

from sentence_transformers import SentenceTransformer
EMBEDDING_MODEL_NAME = 'google/embedding-gemma-300m'
local_embedding_model = None
try:
    if config['llm_choice'] == 'gemini':
        local_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"MCP INFO: Successfully loaded local embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    print(f"MCP WARNING: Could not load local embedding model {EMBEDDING_MODEL_NAME}. RAG with Gemini will fail. Details: {e}")

try:
    chroma_client = chromadb.PersistentClient(path="gem_memory_db")
    chat_collection = chroma_client.get_or_create_collection(name="chat_history")
    image_collection = chroma_client.get_or_create_collection(name="images")
    print("MCP INFO: ChromaDB vector database is ready.")
except Exception as e:
    sys.exit(f"MCP FATAL ERROR: Could not initialize ChromaDB. Details: {e}")

try:
    geolocator = Nominatim(user_agent="gem_ai_assistant")
    tf = TimezoneFinder()
    print("MCP INFO: Geocoding and timezone tools initialized.")
except Exception as e:
    print(f"MCP WARNING: Could not initialize location tools. Time lookups may fail. Details: {e}")

def verify_ollama_models():
    if "ollama" not in config['llm_choice']: return
    print("MCP INFO: Verifying that required Ollama models are available...")
    try:
        tags_url = config['ollama_api_url'].replace('/api/chat', '/api/tags')
        response = requests.get(tags_url, timeout=10)
        response.raise_for_status()
        installed_models = {model['name'] for model in response.json().get('models', [])}
    except Exception as e:
        print(f"MCP WARNING: Could not get model list from Ollama. Skipping verification. Details: {e}")
        return
    required_models = set()
    if config['llm_choice'] == 'ollama': required_models.add(config.get('ollama_model'))
    elif config['llm_choice'] == 'ollama_vision': required_models.add(config.get('ollama_vision_model'))
    required_models.add(config.get('ollama_embedding_model'))
    required_models.discard(None); required_models.discard('')
    all_models_found = True
    for model_name in required_models:
        if model_name not in installed_models:
            print(f"\nFATAL ERROR: The required Ollama model '{model_name}' is not available.")
            print("Please pull the model with 'ollama pull <model_name>' or correct the name in mcp_settings.ini.\n")
            all_models_found = False
    if not all_models_found: sys.exit(1)
    print("MCP INFO: All required Ollama models were found.")

if config['llm_choice'] == "gemini":
    print("MCP INFO: Initializing Gemini...")
    try:
        if not config['gemini_api_key'] or config['gemini_api_key'] == 'YOUR_GEMINI_API_KEY_HERE':
            sys.exit("FATAL ERROR: llm_choice is 'gemini' but api_key is not set.")
        genai.configure(api_key=config['gemini_api_key'])
        gemini_model = genai.GenerativeModel(config['gemini_model'])
        print(f"MCP INFO: Gemini model '{config['gemini_model']}' loaded.")
    except Exception as e:
        sys.exit(f"MCP FATAL ERROR: Failed to configure Gemini API. Details: {e}")
elif config['llm_choice'] in ["ollama", "ollama_vision"]:
    print("MCP INFO: Verifying Ollama connection...")
    try:
        requests.get(config['ollama_api_url'].rsplit('/', 1)[0])
        print(f"MCP INFO: Ollama connection successful.")
        verify_ollama_models()
    except requests.exceptions.ConnectionError:
        sys.exit("MCP FATAL ERROR: Could not connect to Ollama. Is it running?")

osc_client = None
if config['osc_enabled']:
    try:
        osc_client = udp_client.SimpleUDPClient(config['osc_ip'], config['osc_port'])
        print(f"MCP INFO: OSC client configured to send to {config['osc_ip']}:{config['osc_port']}")
    except Exception as e:
        print(f"MCP WARNING: Could not create OSC client. Details: {e}")
# ------------------------------------------------------------------------------


# --- 3. CORE HELPER FUNCTIONS ---
# ------------------------------------------------------------------------------
def clear_screen():
    """Clears the console screen."""
    # For Windows
    if platform.system() == "Windows":
        os.system('cls')
    # For macOS and Linux
    else:
        os.system('clear')

def get_gemini_embedding(text: str = None, image_base64: str = None) -> list[float]:
    if not text: return None
    if image_base64: print("MCP WARNING: Local SentenceTransformer embedding does not support images. Using text only.")
    if local_embedding_model is None:
        print("MCP ERROR: Local embedding model failed to load at startup.")
        return None
    try:
        embedding_array = local_embedding_model.encode(text, convert_to_tensor=False)
        return embedding_array.tolist()
    except Exception as e:
        print(f"MCP ERROR: Could not get local Gemini embedding. Details: {e}")
        return None

def get_embedding(text: str = None, image_base64: str = None) -> list[float]:
    if not text and not image_base64: return None
    if config['llm_choice'] in ['ollama', 'ollama_vision']:
        model_to_use = config.get('ollama_embedding_model')
        if not model_to_use:
            print("MCP ERROR: No embedding model configured for Ollama mode.")
            return None
        payload = {"model": model_to_use, "prompt": text if text else " "}
        if image_base64: payload["images"] = [image_base64]
        try:
            embedding_url = config['ollama_api_url'].replace('/api/chat', '/api/embeddings')
            response = requests.post(embedding_url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get('embedding')
        except Exception as e:
            print(f"MCP ERROR: Could not get embedding from Ollama. Details: {e}")
            return None
    elif config['llm_choice'] == 'gemini':
        if image_base64: print("MCP WARNING: Gemini embedding path does not support images currently. Using text only.")
        return get_gemini_embedding(text=text)
    else:
        print(f"MCP ERROR: Unknown llm_choice '{config['llm_choice']}' for embedding generation.")
        return None

def add_chat_to_memory(speaker: str, text: str):
    vector = get_embedding(text=text)
    if vector:
        try:
            doc_id = datetime.datetime.now().isoformat()
            chat_collection.add(ids=[doc_id], embeddings=[vector], metadatas=[{"speaker": speaker, "text_content": text, "timestamp": doc_id}])
            print(f"MCP MEMORY: Added chat from '{speaker}' to ChromaDB.")
        except Exception as e: print(f"MCP ERROR: Failed to add chat to ChromaDB. Details: {e}")

def add_image_to_memory(image_identifier: str, image_base64: str):
    vector = get_embedding(image_base64=image_base64)
    if vector:
        try:
            image_collection.add(ids=[image_identifier], embeddings=[vector], metadatas=[{"timestamp": image_identifier}])
            print(f"MCP MEMORY: Added image '{image_identifier}' to ChromaDB.")
        except Exception as e: print(f"MCP ERROR: Failed to add image to ChromaDB. Details: {e}")

def ask_llm(user_content: str, image_data_base64: str = None) -> tuple[str, dict]:
    print(f"MCP INFO: Sending prompt to {config['llm_choice'].upper()}...")
    perf_data = {"tps": 0.0}
    try:
        response = None
        system_prompt = config.get('system_prompt', '')
        if config['llm_choice'] in ['ollama_vision', 'ollama']:
            model = config['ollama_vision_model'] if config['llm_choice'] == 'ollama_vision' else config['ollama_model']
            user_message = {"role": "user", "content": user_content}
            if image_data_base64: user_message["images"] = [image_data_base64]
            messages = [{"role": "system", "content": system_prompt}, user_message]
            payload = {"model": model, "messages": messages, "stream": False, "keep_alive": -1}
            response = requests.post(config['ollama_api_url'], json=payload, timeout=120)
        elif config['llm_choice'] == 'gemini':
            final_gemini_prompt = f"{system_prompt}\n\n---\n\n{user_content}"
            gemini_response = gemini_model.generate_content(final_gemini_prompt)
            return gemini_response.text.strip(), perf_data
        if response:
            response.raise_for_status()
            response_json = response.json()
            if 'eval_count' in response_json and 'eval_duration' in response_json:
                eval_count = response_json['eval_count']
                eval_duration_ns = response_json['eval_duration']
                if eval_duration_ns > 0:
                    eval_duration_s = eval_duration_ns / 1_000_000_000
                    tokens_per_second = eval_count / eval_duration_s
                    perf_data["tps"] = tokens_per_second
                    print(f"MCP PERF: Model generated {eval_count} tokens in {eval_duration_s:.2f}s ({tokens_per_second:.2f} T/s)")
            text_response = response_json.get('message', {}).get('content', '').strip()
            return text_response, perf_data
        return "Error: LLM choice not recognized.", perf_data
    except Exception as e:
        print(f"MCP ERROR: An exception occurred in ask_llm. Details: {e}")
        return "Sorry, I encountered an error while trying to think.", perf_data

def retrieve_from_rag(user_query: str) -> str:
    print(f"MCP INFO: RAG retrieval triggered for query: '{user_query}'")
    vector = get_embedding(text=user_query)
    if not vector: return ""
    context_str = "CONTEXT FROM LONG-TERM MEMORY:\n"
    found_context = False
    try:
        chat_results = chat_collection.query(query_embeddings=[vector], n_results=3)
        if chat_results and chat_results['ids'][0]:
            context_str += "[Relevant Chat History]\n"
            for data in chat_results['metadatas'][0]: context_str += f"- {data['speaker']} said: \"{data['text_content']}\"\n"
            found_context = True
        image_results = image_collection.query(query_embeddings=[vector], n_results=1)
        if image_results and image_results['ids'][0]:
            context_str += "\n[Relevant Image]\n"
            context_str += f"- An image was found, identified as: '{image_results['ids'][0][0]}'\n"
            found_context = True
    except Exception as e:
        print(f"MCP ERROR: Failed during RAG search with ChromaDB. Details: {e}")
        return ""
    return context_str if found_context else ""

def get_time_for_location(location_name: str) -> str:
    if not location_name: return None
    try:
        location = geolocator.geocode(location_name)
        if not location: return f"I couldn't find a location named '{location_name}'."
        timezone_name = tf.timezone_at(lng=location.longitude, lat=location.latitude)
        if not timezone_name: return f"I found '{location.address}', but couldn't determine its timezone."
        target_tz = pytz.timezone(timezone_name)
        target_time = datetime.datetime.now(target_tz)
        formatted_time = target_time.strftime("%I:%M %p on %A")
        result_string = f"The current time in {location.address.split(',')[0]} ({timezone_name}) is {formatted_time}."
        print(f"MCP TOOL USED: get_time_for_location() -> '{result_string}'")
        return result_string
    except Exception as e:
        print(f"MCP ERROR: An exception occurred in get_time_for_location. Details: {e}")
        return "I had trouble looking up the time for that location."

def send_over_osc(command_text: str):
    if not config['osc_enabled'] or not osc_client: return
    try:
        builder = osc_message_builder.OscMessageBuilder(address=config['osc_address'])
        builder.add_arg(command_text, builder.ARG_TYPE_STRING); builder.add_arg(True, builder.ARG_TYPE_TRUE)
        osc_client.send(builder.build())
        print(f"MCP INFO: Sent OSC command to {config['osc_address']} -> '{command_text}'")
    except Exception as e: print(f"MCP ERROR: Failed to send OSC message. Details: {e}")

def get_image_from_vision_service() -> str:
    url = config.get('vision_service_get_image_url')
    if not url:
        print("MCP ERROR: The 'vision_service_get_image_url' is not set.")
        return None
    print(f"MCP CORE: Requesting a fresh image from URL: {url}...")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        response_json = response.json()
        if "image_base64" not in response_json:
            print("MCP ERROR: Response from vision.py missing 'image_base64' key.")
            return None
        return response_json.get("image_base64")
    except Exception as e:
        print(f"MCP ERROR: An unexpected error occurred while getting the image: {e}")
        return None

def get_fresh_vision_context() -> str:
    url = config.get('vision_service_scan_url')
    if not url: return "Error: Vision service URL not configured."
    print(f"MCP CORE: Requesting a fresh vision scan from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json().get("vision_context", "Error: Invalid response from vision service.")
    except Exception as e: return f"Error: Could not reach vision service. Is it running? Details: {e}"

def send_to_social_stream(text_to_send: str):
    if not config.get('social_stream_enabled', False): return
    if not text_to_send or text_to_send.startswith("ACTION_GOTO:"): return
    targets = config.get('social_stream_targets', [])
    session_id = config.get('social_stream_session_id')
    api_url = config.get('social_stream_api_url')
    if not all([targets, session_id, api_url]):
        print("MCP DEBUG: Did not send to Social Stream because required settings are missing.")
        return
    def send_to_one_platform(target: str):
        url = f"{api_url}/{session_id}"
        payload = {"action": "sendChat", "value": text_to_send, "target": target}
        print(f"  -> Sending to '{target}'...")
        try:
            requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10).raise_for_status()
            print(f"  -> SUCCESS: Message accepted for '{target}'.")
        except Exception as e: print(f"  -> FAILED: Could not send to '{target}'. Details: {e}")
    print(f"MCP INFO: Broadcasting to Social Stream targets concurrently: {targets}")
    threads = [threading.Thread(target=send_to_one_platform, args=(target,)) for target in targets]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    print("MCP INFO: All social stream broadcasts have completed.")

def send_to_tts(text_to_speak: str):
    if not config.get('styletts_enabled', False): return
    if not text_to_speak or text_to_speak.startswith("ACTION_GOTO:"): return
    url = config.get('styletts_url')
    if not url: return
    clean_text = re.sub(r"[^a-zA-Z0-9\s.,?!'\"():-]", '', text_to_speak)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    if not clean_text: return
    payload = {"chatmessage": clean_text}
    print(f"MCP INFO: Sending SANITIZED text to StyleTTS Server -> '{clean_text}'")
    try:
        requests.post(url, json=payload, timeout=15).raise_for_status()
        print("MCP INFO: StyleTTS server accepted the request.")
    except Exception as e: print(f"MCP ERROR: Could not send to StyleTTS server. Details: {e}")

def get_song_info_and_check_duration(query: str) -> dict:
    """
    Finds a song on YouTube, checks its duration against the configured limit,
    and returns its details or a rejection reason.
    """
    print(f"MCP DOWNLOAD: Verifying song: '{query}'")
    MAX_DURATION_SECONDS = config.get('max_download_duration_seconds', 600)
    
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True, 'force_generic_extractor': True}) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
            if not info.get('entries'):
                return {"status": "error", "message": "I couldn't find any results for that song."}

            video_info = info['entries'][0]
            duration = video_info.get('duration', 0)
            video_url = video_info.get('url')
            title = video_info.get('title', 'Unknown Title')

        if duration and duration > MAX_DURATION_SECONDS:
            minutes, _ = divmod(int(duration), 60)
            limit_minutes, _ = divmod(int(MAX_DURATION_SECONDS), 60)
            message = f"Sorry, the song '{title}' is over {minutes} minutes long, which is past the {limit_minutes} minute limit."
            print(f"!!! MCP DOWNLOAD REJECTED: {message}")
            return {"status": "rejected", "message": message}
        
        return {"status": "ok", "url": video_url, "title": title}

    except Exception as e:
        print(f"!!! MCP DOWNLOAD ERROR: Could not verify song info. Details: {e}")
        return {"status": "error", "message": "Sorry, I ran into an error trying to look up that song."}
# ------------------------------------------------------------------------------


# --- Music Downloader Helper Functions ---
# ------------------------------------------------------------------------------
def handle_song_download_task(video_url: str):
    """Executes the download worker script for a pre-vetted video URL."""
    print(f"MCP DOWNLOAD: Starting worker for URL: '{video_url}'")
    downloaded_filename = None
    try:
        python_executable = sys.executable
        worker_script_path = os.path.join(os.path.dirname(__file__), "download_worker.py")
        if not os.path.exists(worker_script_path):
            print("!!! MCP DOWNLOAD ERROR: 'download_worker.py' not found.")
            return

        command = [python_executable, worker_script_path, video_url]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8',
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )

        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            print(f"  [Worker]: {line}")
            if '.mp3' in line:
                clean_filename = os.path.basename(line)
                downloaded_filename = clean_filename
                print(f"MCP AUTOPLAY: Captured and cleaned filename: '{downloaded_filename}'")

        process.wait()
        print("MCP DOWNLOAD: Worker process finished.")

        if downloaded_filename:
            print(f"MCP AUTOPLAY: Download successful. Creating command file for '{downloaded_filename}'")
            with open("autoplay.txt", 'w', encoding='utf-8') as f:
                f.write(downloaded_filename)
        else:
            print("MCP AUTOPLAY: Download may have failed or filename not provided by worker.")

    except Exception as e:
        print(f"!!! MCP DOWNLOAD ERROR: Failed to execute download worker: {e}")
        traceback.print_exc()

def _process_download_queue():
    """Background thread function to process song download requests one by one."""
    while True:
        video_url = download_queue.get()
        handle_song_download_task(video_url)
        download_queue.task_done()
# ------------------------------------------------------------------------------


# --- Music Recognition Helper Functions ---
# ------------------------------------------------------------------------------
def record_audio_for_music(filename):
    if not MUSIC_RECOGNITION_ENABLED: return None
    duration, samplerate, channels = MUSIC_RECOGNITION_SETTINGS['audio_duration'], MUSIC_RECOGNITION_SETTINGS['sample_rate'], MUSIC_RECOGNITION_SETTINGS['channels']
    print(f"MCP MUSIC: Recording {duration} seconds of audio for recognition...")
    try:
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16', device=SELECTED_INPUT_DEVICE_INDEX)
        sd.wait()
        sf.write(filename, recording, samplerate)
        print(f"MCP MUSIC: Audio recorded and saved to {filename}")
        return filename
    except Exception as e:
        print(f"MCP MUSIC ERROR: Error during audio recording: {e}")
        return None

def recognize_song_via_api(audio_file_path):
    if not os.path.exists(audio_file_path): return {"status": "error", "message": f"Audio file not found: {audio_file_path}"}
    api_key, api_host, api_url = MUSIC_RECOGNITION_SETTINGS['rapidapi_key'], MUSIC_RECOGNITION_SETTINGS['rapidapi_host'], MUSIC_RECOGNITION_SETTINGS['recognition_endpoint_url']
    headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": api_host}
    try:
        with open(audio_file_path, 'rb') as audio_file:
            files = {'upload_file': audio_file}
            print(f"MCP MUSIC: Sending audio file as 'upload_file' to API endpoint: {api_url}")
            response = requests.post(api_url, headers=headers, files=files, timeout=30)
        print(f"MCP MUSIC: API responded with status code: {response.status_code}")
        response.raise_for_status()
        recognition_result = response.json()
        print(f"MCP MUSIC: API response received (first 500 chars): {json.dumps(recognition_result, indent=2)[:500]}...")
        if recognition_result and recognition_result.get('track'):
            track_info = recognition_result['track']
            return {"status": "success", "title": track_info.get('title', 'N/A'), "artist": track_info.get('subtitle', 'N/A'), "url": track_info.get('url', 'N/A'), "raw_data": track_info}
        else:
            return {"status": "error", "message": "API responded but song not recognized or format unexpected.", "raw_data": recognition_result}
    except requests.exceptions.HTTPError as http_err:
        print(f"MCP MUSIC ERROR: HTTP Error occurred: {http_err}")
        print(f"MCP MUSIC ERROR: Response Body: {response.text}")
        return {"status": "error", "message": f"API returned an error: {response.status_code}", "raw_data": response.text}
    except Exception as e:
        print(f"MCP MUSIC ERROR: An unexpected error occurred during API call: {e}")
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}

def clean_up_audio_file(filepath):
    if os.path.exists(filepath):
        try: os.remove(filepath); print(f"MCP MUSIC: Cleaned up temporary file: {filepath}")
        except Exception as e: print(f"MCP MUSIC ERROR: Could not delete temporary file {filepath}. Details: {e}")

def handle_music_recognition_task():
    temp_filename = MUSIC_RECOGNITION_SETTINGS['temp_audio_filename']
    recorded_file = record_audio_for_music(temp_filename)
    if not recorded_file: return {"status": "error", "message": "Failed to record audio."}
    song_info = recognize_song_via_api(audio_file_path=recorded_file)
    clean_up_audio_file(recorded_file)
    return song_info
# ------------------------------------------------------------------------------


# --- 4. UNIVERSAL PROCESSING FUNCTION ---
# ------------------------------------------------------------------------------
def process_task(source: str, user_text: str, vision_context: str = "") -> str:
    """The central logic hub with the corrected, prioritized workflow for all tools."""
    global VISION_HISTORY, CURRENT_LOCATION

    wake_word_detected, clean_user_text = False, ""
    for word in config['wake_words']:
        if not word: continue
        pattern = re.compile(r"^(ok |so |well |hey |okay, |so, |well, |hey, )?" + re.escape(word) + r"\b", re.IGNORECASE)
        match = pattern.search(user_text)
        if match:
            wake_word_detected, start_of_clean_text = True, match.end()
            clean_user_text = user_text[start_of_clean_text:].strip()
            if clean_user_text and clean_user_text[0] in [',', '.', ':', ';']:
                clean_user_text = clean_user_text[1:].strip()
            break
    if not wake_word_detected:
        print(f"MCP: No valid wake word pattern found in '{user_text}'. Ignoring.")
        return ""
    print(f"MCP: Wake word confirmed! Processing: '{clean_user_text}'")
    add_chat_to_memory("User", clean_user_text)

    if MUSIC_RECOGNITION_ENABLED and any(keyword in clean_user_text.lower() for keyword in config['music_trigger_words']):
        print(f"MCP MUSIC: Detected music recognition request: '{clean_user_text}'")
        def run_recognition_in_thread():
            song_info = handle_music_recognition_task()
            response_text = f"I think the song is '{song_info['title']}' by {song_info['artist']}." if song_info["status"] == "success" else "Sorry, I couldn't identify the song right now."
            send_to_tts(response_text)
            send_to_social_stream(response_text)
            add_chat_to_memory("System", response_text)
        threading.Thread(target=run_recognition_in_thread, daemon=True).start()
        return "Listening..."
    
    is_music_download_request = any(trigger in clean_user_text.lower() for trigger in config.get('download_trigger_words', []))
    is_time_request = any(keyword in clean_user_text.lower() for keyword in ['time is it', 'what time', 'current time', 'date'])
    is_rag_request = any(clean_user_text.lower().startswith(trigger) for trigger in config['rag_trigger_words'])
    is_osc_request = config['osc_enabled'] and any(clean_user_text.lower().startswith(verb) for verb in config['osc_trigger_verbs'])
    is_vision_request = any(trigger in clean_user_text.lower() for trigger in config['vision_trigger_words'])

    if is_music_download_request and not config.get('music_downloader_enabled', False):
        final_response = "Sorry, the music request system is currently turned off."
        add_chat_to_memory("Gem", final_response)
        return final_response

    final_response = ""
    
    if is_music_download_request:
        trigger_found = next((trigger for trigger in config['download_trigger_words'] if trigger in clean_user_text.lower()), "")
        search_query = clean_user_text.lower().replace(trigger_found, "", 1).strip()
        
        if not search_query:
            final_response = "What song would you like me to download?"
        else:
            # Perform the duration check immediately
            song_info = get_song_info_and_check_duration(search_query)
            
            if song_info["status"] == "ok":
                # If OK, add the URL to the queue and create the success message
                download_queue.put(song_info["url"])
                final_response = f"Okay, I've added '{song_info['title']}' to the download queue."
            else:
                # If rejected or error, use the message from the helper function
                final_response = song_info["message"]
    
    elif is_osc_request:
        verb_found = next((verb for verb in config['osc_trigger_verbs'] if clean_user_text.lower().startswith(verb)), "")
        destination = clean_user_text[len(verb_found):].strip()
        if not destination: final_response = "Where do you want me to go?"
        elif destination.lower() == CURRENT_LOCATION.lower(): final_response = f"I'm already at {destination}."
        else: send_over_osc(clean_user_text); CURRENT_LOCATION = destination; final_response = f"Okay, I'm heading to {destination} now."
    elif is_time_request:
        print("MCP INFO: Time request detected. Beginning tool-use chain.")
        extraction_prompt = f"From the following text, extract the city, state, or country. Respond with only the name of the location and nothing else.\n\nText: \"{clean_user_text}\""
        location_name, _ = ask_llm(extraction_prompt)
        if not location_name or "couldn't" in location_name.lower() or len(location_name) > 25: location_name = "Pattaya"
        time_context = get_time_for_location(location_name.strip())
        prompt_for_llm = f"CONTEXT FROM A REAL-TIME CLOCK:\n- {time_context}\n\n---\nBased on this live information, answer the user's original question: \"{clean_user_text}\""
        final_response, _ = ask_llm(prompt_for_llm)
    elif is_vision_request:
        if config['llm_choice'] == 'ollama_vision':
            image_data = get_image_from_vision_service()
            if image_data:
                vision_prompt = f"In one or two short sentences, describe the main subject of the attached image. The user's original question was: '{clean_user_text}'"
                final_response, _ = ask_llm(vision_prompt, image_data_base64=image_data)
                image_id = f"image_seen_at_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                add_image_to_memory(image_id, image_data)
                VISION_HISTORY.appendleft(f"User asked about an image ({image_id}), you saw: {final_response}")
            else: final_response = "Sorry, I tried to look but couldn't get an image from the camera."
        else: description = get_fresh_vision_context(); VISION_HISTORY.appendleft(description); final_response = description
    else:
        long_term_memory = retrieve_from_rag(clean_user_text) if is_rag_request else ""
        location_context = f"Your current location is: {CURRENT_LOCATION}."
        history_context = ""
        if VISION_HISTORY: history_context = f"Short term memory of recent events:\n" + "\n".join(f"- {item}" for item in VISION_HISTORY)
        prompt_for_llm = f"{long_term_memory}{location_context}\n{history_context}\n\n---\nBased on all available context, answer the user's question:\n\"{clean_user_text}\""
        final_response, _ = ask_llm(prompt_for_llm)

    add_chat_to_memory("Gem", final_response)
    return final_response
# ------------------------------------------------------------------------------


# --- 5. API ENDPOINTS ---
# ------------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index(): return "Hello from the UNIFIED Master Control Program!"

@app.route('/update_runtime_setting', methods=['POST'])
def update_runtime_setting():
    """
    Updates a specific key in the global 'config' dictionary in real-time.
    This version is hardened to correctly handle boolean types from the control panel.
    """
    data = request.json
    key = data.get('key')
    value = data.get('value')

    print(f"\n--- MCP DEBUG: Received runtime update request ---")
    print(f"Key: {key}, Raw Value: {value}, Type of Value: {type(value)}")

    if not key:
        return jsonify({"status": "error", "message": "Missing 'key' in request."}), 400

    if isinstance(value, bool):
        actual_value = value
    elif isinstance(value, str):
        actual_value = value.lower() == 'true'
    else:
        actual_value = value
    
    if key in config:
        config[key] = actual_value
        print(f"MCP REAL-TIME UPDATE: Setting '{key}' has been successfully changed to -> {actual_value}")
        return jsonify({"status": "ok", "message": f"'{key}' updated successfully."})
    else:
        print(f"MCP REAL-TIME UPDATE FAILED: Key '{key}' not found in config.")
        return jsonify({"status": "error", "message": f"Key '{key}' not found in config."}), 404
# ------------------------------------------------------------------------------

@app.route('/chat', methods=['POST', 'PUT'])
def handle_chat_request():
    data = request.json
    chat_message = data.get('chatmessage', '')
    print(f"\nMCP: Received from [Chat]: '{chat_message}'")
    
    final_response = process_task(source='chat', user_text=chat_message)
    
    if final_response:
        send_to_tts(final_response)
        send_to_social_stream(final_response)
        add_chat_to_memory("Gem", final_response)
        
    return jsonify({"status": "ok"})

@app.route('/vision', methods=['POST'])
def handle_vision_request():
    data = request.json
    user_text = data.get('text', ''); vision_context = data.get('vision_context', '')
    print(f"\nMCP: Received from [Vision]: '{user_text}'")
    
    final_response = process_task(source='vision', user_text=user_text, vision_context=vision_context)
    
    if final_response:
        send_to_tts(final_response)
        send_to_social_stream(final_response)
        add_chat_to_memory("Gem", final_response)
        
    return jsonify({'response': final_response})
    
@app.route('/audio', methods=['POST'])
def handle_audio_request():
    data = request.json
    user_text = data.get('text', '')
    print(f"\nMCP: Received from [Audio]: '{user_text}'")
    
    final_response = process_task(source='audio', user_text=user_text)
    
    if final_response:
        send_to_tts(final_response)
        send_to_social_stream(final_response)
        add_chat_to_memory("Gem", final_response)
        
    return jsonify({'response': final_response})

@app.route('/update_vision', methods=['POST'])
def update_vision_context():
    global VISION_HISTORY
    data = request.json
    new_context = data.get('vision_context')
    if new_context:
        print(f"\nMCP MEMORY: Visual history has been UPDATED -> '{new_context[:70]}...'")
        VISION_HISTORY.appendleft(new_context)
    return jsonify({"status": "vision context updated"})
# ------------------------------------------------------------------------------


# --- 6. MAIN EXECUTION BLOCK ---
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "="*70)
    print("--- Starting UNIFIED Master Control Program (MCP) ---")

    if config.get('music_recognition_enabled', False):
        SELECTED_INPUT_DEVICE_INDEX = select_audio_device()
        if SELECTED_INPUT_DEVICE_INDEX is None:
            print("--- CRITICAL WARNING: Could not configure an audio device. Music recognition will fail. ---")

    download_thread = threading.Thread(target=_process_download_queue, daemon=True)
    download_thread.start()

    clear_screen()
    print(f"--- Using Audio Input Device Index: {SELECTED_INPUT_DEVICE_INDEX} ---")
    print(f"--- LLM Mode: {config['llm_choice'].upper()} ---")
    if config['osc_enabled']: print(f"--- OSC sending ENABLED to {config['osc_ip']}:{config['osc_port']} ---")
    else: print("--- OSC sending is DISABLED ---")
    if MUSIC_RECOGNITION_ENABLED:
        print("--- Music Recognition ENABLED ---")
    else:
        print("--- Music Recognition DISABLED ---")
    if config.get('download_trigger_words'):
        print(f"--- Music Downloader ENABLED ---")
    print(f"--- API Server listening on http://{config['host']}:{config['port']} ---")
    print("="*70 + "\n")
    app.run(host=config['host'], port=config['port'], debug=True, use_reloader=False)
# ------------------------------------------------------------------------------