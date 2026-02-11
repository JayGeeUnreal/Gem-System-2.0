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

import asyncio
import httpx

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

from quart import Quart, request, jsonify
from quart_cors import cors

import subprocess
import traceback
import queue
import yt_dlp # For checking song duration
# ------------------------------------------

import sounddevice as sd
import soundfile as sf
# ------------------------------------------

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

# --- Standalone function for device selection
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

# --- 2. INITIALIZATION ---

MUSIC_RECOGNITION_ENABLED = False
MUSIC_RECOGNITION_SETTINGS = {}

config = load_config()
### ASYNC: Initialize Quart and QuartCors
app = Quart(__name__)
app = cors(app, allow_origin="*") # Basic CORS setup for development

gemini_model = None
VISION_HISTORY = deque(maxlen=5)
CURRENT_LOCATION = "the stream room"
SELECTED_INPUT_DEVICE_INDEX = None
download_queue = queue.Queue()

from sentence_transformers import SentenceTransformer
EMBEDDING_MODEL_NAME = 'google/embeddinggemma-300m'
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
        asyncio.run(verify_ollama_models())
        print(f"MCP INFO: Ollama connection successful.")
    except httpx.ConnectError:
        sys.exit("MCP FATAL ERROR: Could not connect to Ollama. Is it running?")


osc_client = None
if config['osc_enabled']:
    try:
        osc_client = udp_client.SimpleUDPClient(config['osc_ip'], config['osc_port'])
        print(f"MCP INFO: OSC client configured to send to {config['osc_ip']}:{config['osc_port']}")
    except Exception as e:
        print(f"MCP WARNING: Could not create OSC client. Details: {e}")


# --- 3. CORE HELPER FUNCTIONS (NOW ASYNC) ---
# ------------------------------------------------------------------------------

def get_gemini_embedding(text: str = None, image_base64: str = None) -> list[float]:
    #
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

async def get_embedding(text: str = None, image_base64: str = None) -> list[float]:
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
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(embedding_url, json=payload)
            response.raise_for_status()
            return response.json().get('embedding')
        except Exception as e:
            print(f"MCP ERROR: Could not get embedding from Ollama. Details: {e}")
            return None
    elif config['llm_choice'] == 'gemini':
        if image_base64: print("MCP WARNING: Gemini embedding path does not support images currently. Using text only.")

        return await asyncio.to_thread(get_gemini_embedding, text=text)
    else:
        print(f"MCP ERROR: Unknown llm_choice '{config['llm_choice']}' for embedding generation.")
        return None
#
async def add_chat_to_memory(speaker: str, text: str):
    vector = await get_embedding(text=text)
    if vector:
        try:
            doc_id = datetime.datetime.now().isoformat()

            await asyncio.to_thread(
                chat_collection.add,
                ids=[doc_id],
                embeddings=[vector],
                metadatas=[{"speaker": speaker, "text_content": text, "timestamp": doc_id}]
            )
            print(f"MCP MEMORY: Added chat from '{speaker}' to ChromaDB.")
        except Exception as e: print(f"MCP ERROR: Failed to add chat to ChromaDB. Details: {e}")

async def add_image_to_memory(image_identifier: str, image_base64: str):
    vector = await get_embedding(image_base64=image_base64)
    if vector:
        try:
            #
            await asyncio.to_thread(
                image_collection.add,
                ids=[image_identifier],
                embeddings=[vector],
                metadatas=[{"timestamp": image_identifier}]
            )
            print(f"MCP MEMORY: Added image '{image_identifier}' to ChromaDB.")
        except Exception as e: print(f"MCP ERROR: Failed to add image to ChromaDB. Details: {e}")

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
                    print(f"MCP PERF: Model generated {eval_count} tokens in {eval_duration_s:.2f}s ({tokens_per_second:.2f} T/s)")
            text_response = response_json.get('message', {}).get('content', '').strip()
            return text_response, perf_data
        return "Error: LLM choice not recognized.", perf_data
    except Exception as e:
        print(f"MCP ERROR: An exception occurred in ask_llm. Details: {e}")
        return "Sorry, I encountered an error while trying to think.", perf_data


async def retrieve_from_rag(user_query: str) -> str:
    print(f"MCP INFO: RAG retrieval triggered for query: '{user_query}'")
    vector = await get_embedding(text=user_query)
    if not vector: return ""
    context_str = "CONTEXT FROM LONG-TERM MEMORY:\n"
    found_context = False
    try:
        
        chat_results = await asyncio.to_thread(
            chat_collection.query, query_embeddings=[vector], n_results=3
        )
        if chat_results and chat_results['ids'][0]:
            context_str += "[Relevant Chat History]\n"
            for data in chat_results['metadatas'][0]: context_str += f"- {data['speaker']} said: \"{data['text_content']}\"\n"
            found_context = True

        image_results = await asyncio.to_thread(
            image_collection.query, query_embeddings=[vector], n_results=1
        )
        if image_results and image_results['ids'][0]:
            context_str += "\n[Relevant Image]\n"
            context_str += f"- An image was found, identified as: '{image_results['ids'][0][0]}'\n"
            found_context = True
    except Exception as e:
        print(f"MCP ERROR: Failed during RAG search with ChromaDB. Details: {e}")
        return ""
    return context_str if found_context else ""

# Geopy
async def get_time_for_location(location_name: str) -> str:
    if not location_name: return None
    try:
        # These are blocking I/O calls, so they must be run in a thread
        location = await asyncio.to_thread(geolocator.geocode, location_name)
        if not location: return f"I couldn't find a location named '{location_name}'."

        timezone_name = await asyncio.to_thread(tf.timezone_at, lng=location.longitude, lat=location.latitude)
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

# ... (send_over_osc, get_song_info_and_check_duration, etc., can be synchronous for now
# unless they become performance bottlenecks. yt-dlp is blocking, so it's a good candidate for `to_thread` if needed)
def send_over_osc(command_text: str):
    if not config['osc_enabled'] or not osc_client: return
    try:
        builder = osc_message_builder.OscMessageBuilder(address=config['osc_address'])
        builder.add_arg(command_text, builder.ARG_TYPE_STRING); builder.add_arg(True, builder.ARG_TYPE_TRUE)
        osc_client.send(builder.build())
        print(f"MCP INFO: Sent OSC command to {config['osc_address']} -> '{command_text}'")
    except Exception as e: print(f"MCP ERROR: Failed to send OSC message. Details: {e}")

async def get_image_from_vision_service() -> str:
    url = config.get('vision_service_get_image_url')
    if not url:
        print("MCP ERROR: The 'vision_service_get_image_url' is not set.")
        return None
    print(f"MCP CORE: Requesting a fresh image from URL: {url}...")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url)
        response.raise_for_status()
        response_json = response.json()
        if "image_base64" not in response_json:
            print("MCP ERROR: Response from vision.py missing 'image_base64' key.")
            return None
        return response_json.get("image_base64")
    except Exception as e:
        print(f"MCP ERROR: An unexpected error occurred while getting the image: {e}")
        return None

async def get_fresh_vision_context() -> str:
    url = config.get('vision_service_scan_url')
    if not url: return "Error: Vision service URL not configured."
    print(f"MCP CORE: Requesting a fresh vision scan from {url}...")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
        response.raise_for_status()
        return response.json().get("vision_context", "Error: Invalid response from vision service.")
    except Exception as e: return f"Error: Could not reach vision service. Is it running? Details: {e}"

### 
async def send_to_social_stream(text_to_send: str):
    if not config.get('social_stream_enabled', False): return
    if not text_to_send or text_to_send.startswith("ACTION_GOTO:"): return
    targets = config.get('social_stream_targets', [])
    session_id = config.get('social_stream_session_id')
    api_url = config.get('social_stream_api_url')
    if not all([targets, session_id, api_url]):
        print("MCP DEBUG: Did not send to Social Stream because required settings are missing.")
        return

    async def send_to_one_platform(target: str, client: httpx.AsyncClient):
        url = f"{api_url}/{session_id}"
        payload = {"action": "sendChat", "value": text_to_send, "target": target}
        print(f"  -> Sending to '{target}'...")
        try:
            await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            print(f"  -> SUCCESS: Message accepted for '{target}'.")
        except Exception as e:
            print(f"  -> FAILED: Could not send to '{target}'. Details: {e}")

    print(f"MCP INFO: Broadcasting to Social Stream targets concurrently: {targets}")
    async with httpx.AsyncClient(timeout=10) as client:
        # Create a list of tasks to run concurrently
        tasks = [send_to_one_platform(target, client) for target in targets]
        # Wait for all of them to complete
        await asyncio.gather(*tasks)
    print("MCP INFO: All social stream broadcasts have completed.")



async def send_to_tts(text_to_speak: str):
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
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(url, json=payload)
        response.raise_for_status()
        print("MCP INFO: StyleTTS server accepted the request.")
    except Exception as e: print(f"MCP ERROR: Could not send to StyleTTS server. Details: {e}")




# --- 4. UNIVERSAL PROCESSING FUNCTION (NOW ASYNC) ---
# ------------------------------------------------------------------------------
### ASYNC: The main logic hub is now async.
async def process_task(source: str, user_text: str, vision_context: str = "") -> str:
    """The central logic hub with the corrected, prioritized workflow for all tools."""
    global VISION_HISTORY, CURRENT_LOCATION

    #
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
    await add_chat_to_memory("User", clean_user_text)

    # ... checking triggers
    is_time_request = any(keyword in clean_user_text.lower() for keyword in ['time is it', 'what time', 'current time', 'date'])
    is_rag_request = any(clean_user_text.lower().startswith(trigger) for trigger in config['rag_trigger_words'])
    is_osc_request = config['osc_enabled'] and any(clean_user_text.lower().startswith(verb) for verb in config['osc_trigger_verbs'])
    is_vision_request = any(trigger in clean_user_text.lower() for trigger in config['vision_trigger_words'])
    
    final_response = ""

    ### NOW, USE `await` FOR ALL THE ASYNC HELPER FUNCTIONS ###

    if is_osc_request:
        # ... OSC logic is synchronous
        verb_found = next((verb for verb in config['osc_trigger_verbs'] if clean_user_text.lower().startswith(verb)), "")
        destination = clean_user_text[len(verb_found):].strip()
        if not destination: final_response = "Where do you want me to go?"
        elif destination.lower() == CURRENT_LOCATION.lower(): final_response = f"I'm already at {destination}."
        else: send_over_osc(clean_user_text); CURRENT_LOCATION = destination; final_response = f"Okay, I'm heading to {destination} now."
    elif is_time_request:
        print("MCP INFO: Time request detected. Beginning tool-use chain.")
        extraction_prompt = f"From the following text, extract the city, state, or country. Respond with only the name of the location and nothing else.\n\nText: \"{clean_user_text}\""
        location_name, _ = await ask_llm(extraction_prompt)
        if not location_name or "couldn't" in location_name.lower() or len(location_name) > 25: location_name = "Pattaya"
        time_context = await get_time_for_location(location_name.strip())
        prompt_for_llm = f"CONTEXT FROM A REAL-TIME CLOCK:\n- {time_context}\n\n---\nBased on this live information, answer the user's original question: \"{clean_user_text}\""
        final_response, _ = await ask_llm(prompt_for_llm)
    elif is_vision_request:
        if config['llm_choice'] == 'ollama_vision':
            image_data = await get_image_from_vision_service()
            if image_data:
                vision_prompt = f"In one or two short sentences, describe the main subject of the attached image. The user's original question was: '{clean_user_text}'"
                final_response, _ = await ask_llm(vision_prompt, image_data_base64=image_data)
                image_id = f"image_seen_at_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                await add_image_to_memory(image_id, image_data)
                VISION_HISTORY.appendleft(f"User asked about an image ({image_id}), you saw: {final_response}")
            else: final_response = "Sorry, I tried to look but couldn't get an image from the camera."
        else: 
            description = await get_fresh_vision_context()
            VISION_HISTORY.appendleft(description)
            final_response = description
    else:
        long_term_memory = await retrieve_from_rag(clean_user_text) if is_rag_request else ""
        location_context = f"Your current location is: {CURRENT_LOCATION}."
        history_context = ""
        if VISION_HISTORY: history_context = f"Short term memory of recent events:\n" + "\n".join(f"- {item}" for item in VISION_HISTORY)
        prompt_for_llm = f"{long_term_memory}{location_context}\n{history_context}\n\n---\nBased on all available context, answer the user's question:\n\"{clean_user_text}\""
        final_response, _ = await ask_llm(prompt_for_llm)

    ### PLACE COGNEE CALLS HERE ###
    # For example:
    # if "some cognee trigger" in clean_user_text:
    #     cognee_result = await some_async_cognee_function(final_response)
    #     final_response = cognee_result # or formatted result

    await add_chat_to_memory("Gem", final_response)
    return final_response
# ------------------------------------------------------------------------------


# --- 5. API ENDPOINTS (NOW ASYNC for QUART) ---
# ------------------------------------------------------------------------------
@app.route('/', methods=['GET'])
async def index(): 
    return "Hello from the UNIFIED Master Control Program!"

@app.route('/update_runtime_setting', methods=['POST'])
async def update_runtime_setting():
    data = await request.get_json()
    key = data.get('key')
    value = data.get('value')
    # ...
    if not key:
        return jsonify({"status": "error", "message": "Missing 'key' in request."}), 400
    if key in config:
        config[key] = value
        print(f"MCP REAL-TIME UPDATE: Setting '{key}' has been changed to -> {value}")
        return jsonify({"status": "ok", "message": f"'{key}' updated successfully."})
    else:
        print(f"MCP REAL-TIME UPDATE FAILED: Key '{key}' not found in config.")
        return jsonify({"status": "error", "message": f"Key '{key}' not found in config."}), 404

@app.route('/chat', methods=['POST', 'PUT'])
async def handle_chat_request():
    data = await request.get_json()
    chat_message = data.get('chatmessage', '')
    print(f"\nMCP: Received from [Chat]: '{chat_message}'")
    
    final_response = await process_task(source='chat', user_text=chat_message)
    
    if final_response:
        # We can run these tasks concurrently!
        await asyncio.gather(
            send_to_tts(final_response),
            send_to_social_stream(final_response),
            add_chat_to_memory("Gem", final_response)
        )
        
    return jsonify({"status": "ok"})

# ... 
@app.route('/vision', methods=['POST'])
async def handle_vision_request():
    data = await request.get_json()
    user_text = data.get('text', ''); vision_context = data.get('vision_context', '')
    print(f"\nMCP: Received from [Vision]: '{user_text}'")
    
    final_response = await process_task(source='vision', user_text=user_text, vision_context=vision_context)
    
    if final_response:
        await asyncio.gather(
            send_to_tts(final_response),
            send_to_social_stream(final_response),
            add_chat_to_memory("Gem", final_response)
        )
        
    return jsonify({'response': final_response})
    
@app.route('/audio', methods=['POST'])
async def handle_audio_request():
    data = await request.get_json()
    user_text = data.get('text', '')
    print(f"\nMCP: Received from [Audio]: '{user_text}'")
    
    final_response = await process_task(source='audio', user_text=user_text)
    
    if final_response:
        await asyncio.gather(
            send_to_tts(final_response),
            send_to_social_stream(final_response),
            add_chat_to_memory("Gem", final_response)
        )
        
    return jsonify({'response': final_response})

@app.route('/update_vision', methods=['POST'])
async def update_vision_context():
    global VISION_HISTORY
    data = await request.get_json()
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
    print("--- Starting ASYNCHRONOUS UNIFIED Master Control Program (MCP) ---")

    #
    if config.get('music_recognition_enabled', False):
        SELECTED_INPUT_DEVICE_INDEX = select_audio_device()
        if SELECTED_INPUT_DEVICE_INDEX is None:
            print("--- CRITICAL WARNING: Could not configure an audio device. Music recognition will fail. ---")

    # The download queue thread is still a good pattern
    # download_thread = threading.Thread(target=_process_download_queue, daemon=True)
    # download_thread.start()

    print(f"--- LLM Mode: {config['llm_choice'].upper()} ---")
    print(f"--- API Server listening on http://{config['host']}:{config['port']} ---")
    print("="*70 + "\n")

    ### ASYNC: Run the Quart app.
    # Note: debug=True is not recommended for production.
    app.run(host=config['host'], port=config['port'], debug=True, use_reloader=False)
