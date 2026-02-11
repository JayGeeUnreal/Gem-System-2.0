# ==============================================================================
#                      Master Control Program (mcp.py)
#          - UNIFIED MULTIMODAL & PIPELINED ARCHITECTURE (ASYNC/QUART) -
# ==============================================================================

import asyncio
import httpx  # Replaces requests for async
import json
import configparser
import sys
import os
import platform
import google.generativeai as genai
import re
import datetime
import pytz 
import chromadb
from collections import deque
from pythonosc import udp_client, osc_message_builder
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import threading # Kept for the download worker queue

# --- ASYNC FRAMEWORK ---
from quart import Quart, request, jsonify
from quart_cors import cors

import subprocess
import traceback
import queue
import yt_dlp 

import sounddevice as sd
import soundfile as sf

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
            else:
                MUSIC_RECOGNITION_ENABLED = False
        else:
            settings['music_recognition_enabled'] = False
            MUSIC_RECOGNITION_ENABLED = False
        
        if config_parser.has_section('MusicDownloader'):
            raw_download_triggers = config_parser.get('MusicDownloader', 'trigger_words', fallback='')
            settings['download_trigger_words'] = [word.strip().lower() for word in raw_download_triggers.split(',') if word.strip()]
            settings['music_downloader_enabled'] = config_parser.getboolean('MusicDownloader', 'enabled', fallback=False)
            settings['max_download_duration_seconds'] = config_parser.getint('MusicDownloader', 'max_download_duration_seconds', fallback=600)
        else:
            settings['download_trigger_words'] = []
            settings['music_downloader_enabled'] = False
            settings['max_download_duration_seconds'] = 600

    except Exception as e:
        sys.exit(f"FATAL ERROR: A setting is missing or invalid in '{config_file}'. Details: {e}")
    return settings
# ------------------------------------------------------------------------------


def select_audio_device():
    """Selects an audio input device."""
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
        
        default_idx = sd.default.device[0]
        if default_idx in input_devices:
            return default_idx

        return next(iter(input_devices))

    except Exception as e:
        print(f"MCP ERROR: Could not query or configure audio devices. Details: {e}")
        return None
# ------------------------------------------------------------------------------


# --- 2. INITIALIZATION ---
# ------------------------------------------------------------------------------
MUSIC_RECOGNITION_ENABLED = False
MUSIC_RECOGNITION_SETTINGS = {}

config = load_config()

# --- QUART INIT ---
app = Quart(__name__)
app = cors(app, allow_origin="*")

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
    print(f"MCP WARNING: Could not load local embedding model. RAG with Gemini will fail. Details: {e}")

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
    print(f"MCP WARNING: Could not initialize location tools. Details: {e}")

# ASYNC: Made async to use httpx
async def verify_ollama_models():
    if "ollama" not in config['llm_choice']: return
    print("MCP INFO: Verifying that required Ollama models are available...")
    try:
        tags_url = config['ollama_api_url'].replace('/api/chat', '/api/tags')
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(tags_url)
        response.raise_for_status()
        installed_models = {model['name'] for model in response.json().get('models', [])}
    except Exception as e:
        print(f"MCP WARNING: Could not get model list from Ollama. Details: {e}")
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
            all_models_found = False
    if not all_models_found: sys.exit(1)
    print("MCP INFO: All required Ollama models were found.")

# Note: We keep synchronous logic for Gemini init as it's a library call during startup
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
    # Run async verification in the event loop at startup
    try:
        asyncio.run(verify_ollama_models())
    except Exception as e:
        sys.exit(f"MCP FATAL ERROR: Ollama Check Failed. {e}")

osc_client = None
if config['osc_enabled']:
    try:
        osc_client = udp_client.SimpleUDPClient(config['osc_ip'], config['osc_port'])
        print(f"MCP INFO: OSC client configured to send to {config['osc_ip']}:{config['osc_port']}")
    except Exception as e:
        print(f"MCP WARNING: Could not create OSC client. Details: {e}")
# ------------------------------------------------------------------------------


# --- 3. CORE HELPER FUNCTIONS (ASYNC CONVERSION) ---
# ------------------------------------------------------------------------------
def clear_screen():
    if platform.system() == "Windows": os.system('cls')
    else: os.system('clear')

def get_gemini_embedding(text: str = None, image_base64: str = None) -> list[float]:
    # CPU bound: will run in executor
    if not text: return None
    if image_base64: print("MCP WARNING: Local SentenceTransformer embedding does not support images.")
    if local_embedding_model is None: return None
    try:
        embedding_array = local_embedding_model.encode(text, convert_to_tensor=False)
        return embedding_array.tolist()
    except Exception as e:
        print(f"MCP ERROR: Local Gemini embedding failed: {e}")
        return None

async def get_embedding(text: str = None, image_base64: str = None) -> list[float]:
    if not text and not image_base64: return None
    if config['llm_choice'] in ['ollama', 'ollama_vision']:
        model_to_use = config.get('ollama_embedding_model')
        if not model_to_use: return None
        payload = {"model": model_to_use, "prompt": text if text else " "}
        if image_base64: payload["images"] = [image_base64]
        try:
            url = config['ollama_api_url'].replace('/api/chat', '/api/embeddings')
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json().get('embedding')
        except Exception as e:
            print(f"MCP ERROR: Ollama embedding failed: {e}")
            return None
    elif config['llm_choice'] == 'gemini':
        # Run CPU bound task in thread
        return await asyncio.to_thread(get_gemini_embedding, text=text)
    return None

async def add_chat_to_memory(speaker: str, text: str):
    vector = await get_embedding(text=text)
    if vector:
        try:
            doc_id = datetime.datetime.now().isoformat()
            # ChromaDB I/O in thread
            await asyncio.to_thread(chat_collection.add, ids=[doc_id], embeddings=[vector], metadatas=[{"speaker": speaker, "text_content": text, "timestamp": doc_id}])
            print(f"MCP MEMORY: Added chat from '{speaker}' to ChromaDB.")
        except Exception as e: print(f"MCP ERROR: Failed to add chat to ChromaDB: {e}")

async def add_image_to_memory(image_identifier: str, image_base64: str):
    vector = await get_embedding(image_base64=image_base64)
    if vector:
        try:
            await asyncio.to_thread(image_collection.add, ids=[image_identifier], embeddings=[vector], metadatas=[{"timestamp": image_identifier}])
            print(f"MCP MEMORY: Added image '{image_identifier}' to ChromaDB.")
        except Exception as e: print(f"MCP ERROR: Failed to add image to ChromaDB: {e}")

async def ask_llm(user_content: str, image_data_base64: str = None) -> tuple[str, dict]:
    print(f"MCP INFO: Sending prompt to {config['llm_choice'].upper()}...")
    perf_data = {"tps": 0.0}
    try:
        response_json = None
        system_prompt = config.get('system_prompt', '')
        
        if config['llm_choice'] in ['ollama_vision', 'ollama']:
            model = config['ollama_vision_model'] if config['llm_choice'] == 'ollama_vision' else config['ollama_model']
            user_message = {"role": "user", "content": user_content}
            if image_data_base64: user_message["images"] = [image_data_base64]
            messages = [{"role": "system", "content": system_prompt}, user_message]
            payload = {"model": model, "messages": messages, "stream": False, "keep_alive": -1}
            
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(config['ollama_api_url'], json=payload)
            response.raise_for_status()
            response_json = response.json()

        elif config['llm_choice'] == 'gemini':
            final_gemini_prompt = f"{system_prompt}\n\n---\n\n{user_content}"
            # ASYNC Call for Gemini
            gemini_response = await gemini_model.generate_content_async(final_gemini_prompt)
            return gemini_response.text.strip(), perf_data

        if response_json:
            if 'eval_count' in response_json and 'eval_duration' in response_json:
                eval_count = response_json['eval_count']
                eval_duration_ns = response_json['eval_duration']
                if eval_duration_ns > 0:
                    eval_duration_s = eval_duration_ns / 1_000_000_000
                    tokens_per_second = eval_count / eval_duration_s
                    perf_data["tps"] = tokens_per_second
                    print(f"MCP PERF: {tokens_per_second:.2f} T/s")
            return response_json.get('message', {}).get('content', '').strip(), perf_data
        
        return "Error: LLM choice not recognized.", perf_data
    except Exception as e:
        print(f"MCP ERROR: ask_llm failed: {e}")
        return "Sorry, I encountered an error while trying to think.", perf_data

async def retrieve_from_rag(user_query: str) -> str:
    print(f"MCP INFO: RAG retrieval for: '{user_query}'")
    vector = await get_embedding(text=user_query)
    if not vector: return ""
    context_str = "CONTEXT FROM LONG-TERM MEMORY:\n"
    found_context = False
    try:
        # ChromaDB Query in Thread
        chat_results = await asyncio.to_thread(chat_collection.query, query_embeddings=[vector], n_results=3)
        if chat_results and chat_results['ids'][0]:
            context_str += "[Relevant Chat History]\n"
            for data in chat_results['metadatas'][0]: context_str += f"- {data['speaker']} said: \"{data['text_content']}\"\n"
            found_context = True
        
        image_results = await asyncio.to_thread(image_collection.query, query_embeddings=[vector], n_results=1)
        if image_results and image_results['ids'][0]:
            context_str += "\n[Relevant Image]\n"
            context_str += f"- An image was found: '{image_results['ids'][0][0]}'\n"
            found_context = True
    except Exception as e:
        print(f"MCP ERROR: RAG search failed: {e}")
        return ""
    return context_str if found_context else ""

async def get_time_for_location(location_name: str) -> str:
    if not location_name: return None
    try:
        # Geopy in thread
        location = await asyncio.to_thread(geolocator.geocode, location_name)
        if not location: return f"I couldn't find '{location_name}'."
        timezone_name = await asyncio.to_thread(tf.timezone_at, lng=location.longitude, lat=location.latitude)
        target_tz = pytz.timezone(timezone_name)
        target_time = datetime.datetime.now(target_tz)
        formatted_time = target_time.strftime("%I:%M %p on %A")
        return f"The time in {location.address.split(',')[0]} is {formatted_time}."
    except Exception as e:
        print(f"MCP ERROR: Time lookup failed: {e}")
        return "I had trouble looking up the time."

def send_over_osc(command_text: str):
    # OSC is fast UDP, safe to be synchronous usually, but can wrap if needed. 
    # Keeping sync for simplicity as udp_client is non-blocking mostly.
    if not config['osc_enabled'] or not osc_client: return
    try:
        builder = osc_message_builder.OscMessageBuilder(address=config['osc_address'])
        builder.add_arg(command_text, builder.ARG_TYPE_STRING); builder.add_arg(True, builder.ARG_TYPE_TRUE)
        osc_client.send(builder.build())
        print(f"MCP INFO: Sent OSC -> '{command_text}'")
    except Exception as e: print(f"MCP ERROR: OSC failed: {e}")

async def get_image_from_vision_service() -> str:
    url = config.get('vision_service_get_image_url')
    if not url: return None
    print(f"MCP CORE: Requesting image from: {url}...")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url)
        response.raise_for_status()
        return response.json().get("image_base64")
    except Exception as e:
        print(f"MCP ERROR: Image fetch failed: {e}")
        return None

async def get_fresh_vision_context() -> str:
    url = config.get('vision_service_scan_url')
    if not url: return "Error: Vision URL not configured."
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
        response.raise_for_status()
        return response.json().get("vision_context", "Error: Invalid response.")
    except Exception as e: return f"Error: Vision service unreachable: {e}"

async def send_to_social_stream(text_to_send: str):
    if not config.get('social_stream_enabled', False) or not text_to_send: return
    targets = config.get('social_stream_targets', [])
    api_url = config.get('social_stream_api_url')
    session_id = config.get('social_stream_session_id')
    
    async def send_one(target):
        url = f"{api_url}/{session_id}"
        payload = {"action": "sendChat", "value": text_to_send, "target": target}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(url, json=payload)
            print(f"  -> Social sent to '{target}'")
        except Exception as e: print(f"  -> Social failed '{target}': {e}")
    
    print(f"MCP INFO: Broadcasting to Social Stream: {targets}")
    # Async gather replaces threading
    await asyncio.gather(*(send_one(t) for t in targets))
    print("MCP INFO: Social broadcast done.")

async def send_to_tts(text_to_speak: str):
    if not config.get('styletts_enabled', False) or not text_to_speak: return
    url = config.get('styletts_url')
    clean_text = re.sub(r"[^a-zA-Z0-9\s.,?!'\"():-]", '', text_to_speak)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    if not clean_text: return
    payload = {"chatmessage": clean_text}
    print(f"MCP INFO: Sending to TTS -> '{clean_text}'")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(url, json=payload)
    except Exception as e: print(f"MCP ERROR: TTS failed: {e}")

# Helper: Youtube check (Blocking, used in logic)
def get_song_info_and_check_duration(query: str) -> dict:
    print(f"MCP DOWNLOAD: Verifying song: '{query}'")
    MAX_SEC = config.get('max_download_duration_seconds', 600)
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
            if not info.get('entries'): return {"status": "error", "message": "No results found."}
            video_info = info['entries'][0]
            if video_info.get('duration', 0) > MAX_SEC:
                return {"status": "rejected", "message": f"Song is too long (>{MAX_SEC}s)."}
            return {"status": "ok", "url": video_info.get('url'), "title": video_info.get('title', 'Unknown')}
    except Exception as e:
        print(f"MCP DOWNLOAD ERROR: {e}")
        return {"status": "error", "message": "Error looking up song."}
# ------------------------------------------------------------------------------


# --- Music Downloader Helper Functions ---
# ------------------------------------------------------------------------------
def handle_song_download_task(video_url: str):
    # This remains blocking/subprocess based, run in a separate thread
    print(f"MCP DOWNLOAD: Worker starting for: '{video_url}'")
    downloaded_filename = None
    try:
        worker_path = os.path.join(os.path.dirname(__file__), "download_worker.py")
        if not os.path.exists(worker_path): return
        
        cmd = [sys.executable, worker_path, video_url]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if '.mp3' in line:
                # 1. Take only the part before and including '.mp3'
                clean_line = line.split('.mp3')[0] + '.mp3'
                # 2. If the line contains a destination path (e.g., [ExtractAudio] Destination: filename.mp3)
                if "Destination: " in clean_line:
                    clean_line = clean_line.split("Destination: ")[-1]
                # 3. Final cleanup of the basename
                downloaded_filename = os.path.basename(clean_line.strip())
        
        process.wait()
        if downloaded_filename:
            with open("autoplay.txt", 'w', encoding='utf-8') as f: f.write(downloaded_filename)
    except Exception as e: traceback.print_exc()

def _process_download_queue():
    while True:
        url = download_queue.get()
        handle_song_download_task(url)
        download_queue.task_done()
# ------------------------------------------------------------------------------


# --- Music Recognition Helper Functions ---
# ------------------------------------------------------------------------------
def record_audio_for_music(filename):
    if not MUSIC_RECOGNITION_ENABLED: return None
    s = MUSIC_RECOGNITION_SETTINGS
    print(f"MCP MUSIC: Recording {s['audio_duration']}s...")
    try:
        # Blocking record
        rec = sd.rec(int(s['audio_duration'] * s['sample_rate']), samplerate=s['sample_rate'], channels=s['channels'], dtype='int16', device=SELECTED_INPUT_DEVICE_INDEX)
        sd.wait()
        sf.write(filename, rec, s['sample_rate'])
        return filename
    except Exception as e: print(f"Music Record Error: {e}"); return None

async def recognize_song_via_api(audio_file_path):
    # Converted to Async
    if not os.path.exists(audio_file_path): return {"status": "error"}
    s = MUSIC_RECOGNITION_SETTINGS
    headers = {"X-RapidAPI-Key": s['rapidapi_key'], "X-RapidAPI-Host": s['rapidapi_host']}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            with open(audio_file_path, 'rb') as f:
                response = await client.post(s['recognition_endpoint_url'], headers=headers, files={'upload_file': f})
        response.raise_for_status()
        res = response.json()
        if res.get('track'):
            t = res['track']
            return {"status": "success", "title": t.get('title'), "artist": t.get('subtitle')}
        return {"status": "error", "message": "Not recognized."}
    except Exception as e: return {"status": "error", "message": str(e)}

def clean_up_audio_file(filepath):
    if os.path.exists(filepath):
        try: os.remove(filepath)
        except: pass

async def handle_music_recognition_task_async():
    # Wrapper to run blocking record in thread, then async upload
    temp_file = MUSIC_RECOGNITION_SETTINGS['temp_audio_filename']
    # Record (blocking) in thread
    rec_file = await asyncio.to_thread(record_audio_for_music, temp_file)
    if not rec_file: return
    # Upload (async)
    info = await recognize_song_via_api(rec_file)
    clean_up_audio_file(rec_file)
    
    resp_text = f"I think the song is '{info['title']}' by {info['artist']}." if info["status"] == "success" else "Sorry, I couldn't identify the song."
    await asyncio.gather(
        send_to_tts(resp_text),
        send_to_social_stream(resp_text),
        add_chat_to_memory("System", resp_text)
    )
# ------------------------------------------------------------------------------


# --- 4. UNIVERSAL PROCESSING FUNCTION (ASYNC) ---
# ------------------------------------------------------------------------------
async def process_task(source: str, user_text: str, vision_context: str = "") -> str:
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
        print(f"MCP: No wake word in '{user_text}'")
        return ""
    
    print(f"MCP: Wake word confirmed! Processing: '{clean_user_text}'")
    await add_chat_to_memory("User", clean_user_text)

    # Music Recognition Trigger
    if MUSIC_RECOGNITION_ENABLED and any(k in clean_user_text.lower() for k in config['music_trigger_words']):
        print(f"MCP MUSIC: Triggered.")
        # Fire and forget async background task
        asyncio.create_task(handle_music_recognition_task_async())
        return "Listening..."
    
    is_download = any(t in clean_user_text.lower() for t in config.get('download_trigger_words', []))
    is_time = any(k in clean_user_text.lower() for k in ['time is it', 'what time', 'current time'])
    is_rag = any(clean_user_text.lower().startswith(t) for t in config['rag_trigger_words'])
    is_osc = config['osc_enabled'] and any(clean_user_text.lower().startswith(v) for v in config['osc_trigger_verbs'])
    is_vision = any(t in clean_user_text.lower() for t in config['vision_trigger_words'])

    if is_download and not config.get('music_downloader_enabled', False):
        resp = "Music request system is off."
        await add_chat_to_memory("Gem", resp)
        return resp

    final_response = ""
    
    if is_download:
        trigger = next((t for t in config['download_trigger_words'] if t in clean_user_text.lower()), "")
        query = clean_user_text.lower().replace(trigger, "", 1).strip()
        if not query: final_response = "What song should I download?"
        else:
            # yt-dlp check in thread
            info = await asyncio.to_thread(get_song_info_and_check_duration, query)
            if info["status"] == "ok":
                download_queue.put(info["url"])
                final_response = f"Added '{info['title']}' to queue."
            else: final_response = info["message"]
    
    elif is_osc:
        verb = next((v for v in config['osc_trigger_verbs'] if clean_user_text.lower().startswith(v)), "")
        dest = clean_user_text[len(verb):].strip()
        if not dest: final_response = "Where to?"
        elif dest.lower() == CURRENT_LOCATION.lower(): final_response = f"I'm at {dest}."
        else: send_over_osc(clean_user_text); CURRENT_LOCATION = dest; final_response = f"Going to {dest}."
    
    elif is_time:
        loc, _ = await ask_llm(f"Extract city from: {clean_user_text}. Only location name.")
        time_ctx = await get_time_for_location(loc.strip() if len(loc) < 25 else "Pattaya")
        final_response, _ = await ask_llm(f"Time Context: {time_ctx}. Answer: {clean_user_text}")
    
    elif is_vision:
        if config['llm_choice'] == 'ollama_vision':
            img = await get_image_from_vision_service()
            if img:
                resp, _ = await ask_llm(f"Describe image. User: {clean_user_text}", image_data_base64=img)
                final_response = resp
                await add_image_to_memory(f"img_{datetime.datetime.now().timestamp()}", img)
            else: final_response = "Can't see image."
        else:
            final_response = await get_fresh_vision_context()
            VISION_HISTORY.appendleft(final_response)
    
    else:
        rag = await retrieve_from_rag(clean_user_text) if is_rag else ""
        hist = "\n".join(VISION_HISTORY) if VISION_HISTORY else ""
        prompt = f"{rag}Loc: {CURRENT_LOCATION}\n{hist}\nUser: {clean_user_text}"
        final_response, _ = await ask_llm(prompt)

    await add_chat_to_memory("Gem", final_response)
    return final_response
# ------------------------------------------------------------------------------


# --- 5. API ENDPOINTS (QUART) ---
# ------------------------------------------------------------------------------
@app.route('/', methods=['GET'])
async def index(): return "Hello from UNIFIED ASYNC MCP!"

@app.route('/update_runtime_setting', methods=['POST'])
async def update_runtime_setting():
    data = await request.get_json()
    key, value = data.get('key'), data.get('value')
    if not key: return jsonify({"status": "error"}), 400
    
    actual_val = value
    if isinstance(value, str): actual_val = value.lower() == 'true'
    
    if key in config:
        config[key] = actual_val
        return jsonify({"status": "ok"})
    return jsonify({"status": "error"}), 404

@app.route('/chat', methods=['POST', 'PUT'])
async def handle_chat_request():
    data = await request.get_json()
    chat_message = data.get('chatmessage', '')
    print(f"\nMCP: Received [Chat]: '{chat_message}'")
    
    final_response = await process_task(source='chat', user_text=chat_message)
    
    if final_response:
        await asyncio.gather(
            send_to_tts(final_response),
            send_to_social_stream(final_response),
            add_chat_to_memory("Gem", final_response)
        )
        
    return jsonify({"status": "ok"})

@app.route('/vision', methods=['POST'])
async def handle_vision_request():
    data = await request.get_json()
    final_response = await process_task(source='vision', user_text=data.get('text', ''), vision_context=data.get('vision_context', ''))
    if final_response:
        await asyncio.gather(send_to_tts(final_response), send_to_social_stream(final_response), add_chat_to_memory("Gem", final_response))
    return jsonify({'response': final_response})
    
@app.route('/audio', methods=['POST'])
async def handle_audio_request():
    data = await request.get_json()
    final_response = await process_task(source='audio', user_text=data.get('text', ''))
    if final_response:
        await asyncio.gather(send_to_tts(final_response), send_to_social_stream(final_response), add_chat_to_memory("Gem", final_response))
    return jsonify({'response': final_response})

@app.route('/update_vision', methods=['POST'])
async def update_vision_context():
    data = await request.get_json()
    if data.get('vision_context'): VISION_HISTORY.appendleft(data.get('vision_context'))
    return jsonify({"status": "updated"})
# ------------------------------------------------------------------------------


# --- 6. MAIN EXECUTION BLOCK ---
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "="*70)
    print("--- Starting UNIFIED ASYNC Master Control Program (MCP) ---")

    if config.get('music_recognition_enabled', False):
        SELECTED_INPUT_DEVICE_INDEX = select_audio_device()

    # Background threads for blocking download worker (keep as thread)
    download_thread = threading.Thread(target=_process_download_queue, daemon=True)
    download_thread.start()

    print(f"--- LLM Mode: {config['llm_choice'].upper()} ---")
    print(f"--- API Server listening on http://{config['host']}:{config['port']} ---")
    print("="*70 + "\n")
    
    # Quart run
    app.run(host=config['host'], port=config['port'], debug=True, use_reloader=False)