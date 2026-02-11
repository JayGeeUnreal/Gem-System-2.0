from flask import Flask, request, jsonify, render_template
import json
import aiofiles
import os
import datetime
import re
import html
import logging
import threading
import time
from collections import deque, Counter

# Note: googleapiclient import kept for compatibility
try:
    from googleapiclient.discovery import build
except ImportError:
    pass

import urllib.request
import urllib.error

app = Flask(__name__)

# --- SILENCE TERMINAL ---
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR) 

# --- CONFIGURATION ---
JSON_LOG_FILE = 'chat_log.jsonl'
TEXT_LOG_FILE = 'readable_chat_log.txt'
MASTER_LOG_FILE = 'master_chat_historly.txt'

MENTIONS_LOG_DIR = 'mention_logs'
EXTRACT_LOG_DIR = 'extracted_logs'
OLD_LOGS_DIR = 'old chat logs'
CONFIG_FILE = 'config.json'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 

# --- WHISPER AI ---
try:
    import whisper
    print("Loading Whisper Model...")
    audio_model = whisper.load_model("base") 
    print("Whisper Model Loaded!")
except ImportError:
    print("Whisper not installed.")
    audio_model = None

live_messages = deque(maxlen=50)
total_post_count = 0
PROCESSED_IDS = deque(maxlen=100)

LEADERBOARD_SCORES = Counter()
SHIT_TALKER_SCORES = Counter()

# --- DEFAULTS ---
DEFAULT_MENTIONS = [["Rabbit", "bunny"], ["Cool"], ["GG", "wp"]]
DEFAULT_TALKERS = [["trash", "bad"], ["noob"], ["lag"]]

TARGET_GROUPS = DEFAULT_MENTIONS
SHIT_TALKER_GROUPS = DEFAULT_TALKERS
IGNORED_PHRASES = [] 

# --- YOUTUBE VARS ---
YOUTUBE_API_KEY = "" 
YOUTUBE_CHANNELS = [] 
UPCOMING_STREAMS = []
HANDLE_CACHE = {} 
LAST_SCORE_TIME = 0

# --- HELPER: GET CORRECT FILE PATH ---
def get_file_path(filename):
    if filename in [JSON_LOG_FILE, TEXT_LOG_FILE, MASTER_LOG_FILE]:
        return filename
    return os.path.join(OLD_LOGS_DIR, filename)

# --- HELPER: SAVE/LOAD CONFIG ---
def load_config():
    global TARGET_GROUPS, SHIT_TALKER_GROUPS, IGNORED_PHRASES, YOUTUBE_API_KEY, YOUTUBE_CHANNELS
    if not os.path.exists(CONFIG_FILE): return
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            TARGET_GROUPS = data.get('mentions', DEFAULT_MENTIONS)
            SHIT_TALKER_GROUPS = data.get('talkers', DEFAULT_TALKERS)
            IGNORED_PHRASES = data.get('ignored', []) 
            YOUTUBE_API_KEY = data.get('yt_api_key', "")
            
            yt_data = data.get('yt_channels', [])
            if isinstance(yt_data, str):
                YOUTUBE_CHANNELS = [yt_data] if yt_data else []
            else:
                YOUTUBE_CHANNELS = yt_data
                
            print("--- Config Loaded ---")
    except Exception as e: print(f"Error loading config: {e}")

def save_config():
    try:
        data = { 
            "mentions": TARGET_GROUPS, 
            "talkers": SHIT_TALKER_GROUPS,
            "ignored": IGNORED_PHRASES, 
            "yt_api_key": YOUTUBE_API_KEY,
            "yt_channels": YOUTUBE_CHANNELS
        }
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print("--- Config Saved ---")
    except Exception as e: print(f"Error saving config: {e}")

# --- RESOLVE @HANDLE TO CHANNEL ID ---
def resolve_channel_id(entry):
    entry = entry.strip()
    if entry.startswith('UC'): return entry
    if entry in HANDLE_CACHE: return HANDLE_CACHE[entry]

    try:
        url = f"https://www.youtube.com/{entry}"
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Cookie': 'CONSENT=YES+'}
        )
        with urllib.request.urlopen(req) as response:
            html_content = response.read().decode('utf-8')
            match = re.search(r'itemprop="identifier" content="(UC[\w-]+)"', html_content)
            if not match: match = re.search(r'"externalId":"(UC[\w-]+)"', html_content)
            if not match: match = re.search(r'"channelId":"(UC[\w-]+)"', html_content)
            
            if match:
                found_id = match.group(1)
                HANDLE_CACHE[entry] = found_id
                return found_id
            return None
    except Exception as e:
        print(f"[YouTube Resolver Error] {e}")
        return None

# --- SCRAPE UPCOMING STREAMS (ROBUST for EU/GDPR) ---
def scrape_upcoming_streams(channel_id):
    streams = []
    try:
        # Check the /streams tab specifically (Live Tab)
        url = f"https://www.youtube.com/channel/{channel_id}/streams"
        req = urllib.request.Request(
            url, 
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
                'Cookie': 'CONSENT=YES+; SOCS=CAISEwgDEgk0ODE3Nzk3MjQaAmVuIAEaBgiA_LyaBg;'
            }
        )
        with urllib.request.urlopen(req) as response:
            html_content = response.read().decode('utf-8')
            
            # --- DEBUG BLOCK CHECK ---
            if "Before you continue" in html_content[:2000]:
                print(f"⚠️ BLOCKED: YouTube is showing GDPR Consent for {channel_id}.")

            # --- SEARCH STRATEGY ---
            matches = re.finditer(r'"videoId":"([\w-]+)"', html_content)
            
            for m in matches:
                vid_id = m.group(1)
                start_pos = m.start()
                # Look 6000 chars ahead of the ID for status indicators
                chunk = html_content[start_pos : start_pos + 6000]
                
                # --- LIVE INDICATORS ---
                is_live = False
                if '"text":"LIVE"' in chunk: is_live = True
                elif '"label":"LIVE"' in chunk: is_live = True
                elif 'BADGE_STYLE_TYPE_LIVE_NOW' in chunk: is_live = True
                elif 'DEFAULT_LIVE_BADGE' in chunk: is_live = True
                elif 'ow-playing-badge' in chunk: is_live = True  # Added CSS check
                
                # --- UPCOMING INDICATORS ---
                is_upcoming = False
                if '"text":"UPCOMING"' in chunk: is_upcoming = True
                elif 'upcomingEventData' in chunk: is_upcoming = True
                
                if is_live:
                    full_url = f"https://www.youtube.com/watch?v={vid_id}"
                    if not any(s['url'] == full_url for s in streams):
                        streams.append({'title': "[LIVE NOW] Live Stream", 'url': full_url})
                        
                elif is_upcoming:
                    full_url = f"https://www.youtube.com/watch?v={vid_id}"
                    if not any(s['url'] == full_url for s in streams):
                        streams.append({'title': "[SCHEDULED] Upcoming Stream", 'url': full_url})

        if streams:
            ts = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"[{ts}] 🔴 Found {len(streams)} streams for {channel_id}")
            
    except Exception as e:
        print(f"[YouTube Scrape Error {channel_id}] {e}")
        
    return streams

# --- YOUTUBE MONITOR LOOP ---
def youtube_monitor_loop():
    global UPCOMING_STREAMS
    ts_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts_start}] --- YouTube Monitor Thread Started ---")
    
    while True:
        if YOUTUBE_CHANNELS:
            all_streams = []
            for entry in YOUTUBE_CHANNELS:
                if not entry.strip(): continue
                target_id = resolve_channel_id(entry)
                if not target_id: continue
                found = scrape_upcoming_streams(target_id)
                for s in found:
                    if entry not in s['title']:
                        s['title'] = f"[{entry}] {s['title']}"
                all_streams.extend(found)
            UPCOMING_STREAMS = all_streams
            # Check every 5 minutes
            time.sleep(300)
        else:
            time.sleep(30)

def clean_message_content(raw_html):
    if not raw_html: return ""
    clean_text = re.sub(r'<img[^>]*alt=["\']([^"\']*)["\'][^>]*>', r'\1', raw_html)
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    clean_text = html.unescape(clean_text)
    return clean_text.strip()

def count_group_hits(message, group):
    words = [w.strip() for w in group if w.strip()]
    if not words: return 0
    sorted_words = sorted(words, key=len, reverse=True)
    patterns = []
    for word in sorted_words:
        start_pattern = r"\b" if re.match(r'^\w', word) else r"(?:^|\s)"
        end_pattern = r"\b" if re.match(r'\w$', word) else r"(?=$|\s)"
        safe_word = re.escape(word)
        patterns.append(f"{start_pattern}{safe_word}{end_pattern}")
    full_pattern = "|".join(patterns)
    matches = re.findall(full_pattern, message, flags=re.IGNORECASE)
    return len(matches)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/leaderboard')
def leaderboard_page():
    return render_template('leaderboard.html')

@app.route('/all_time')
def all_time_page():
    return render_template('all_time_leaderboard.html')

# --- NEW OBS OVERLAY ROUTE ---
@app.route('/obs')
def obs_overlay_page():
    return render_template('obs_overlay.html')

@app.route('/leaderboard_podium')
def leaderboard_podium_page():
    return render_template('leaderboard_podium.html')

@app.route('/chat', methods=['POST'])
async def receive_chat():
    global total_post_count, LAST_SCORE_TIME
    try:
        data = request.json
        if data:
            msg_id = data.get('id') or data.get('nid')
            if msg_id:
                if str(msg_id) in PROCESSED_IDS:
                    return jsonify({"status": "ignored"}), 200
                PROCESSED_IDS.append(str(msg_id))

            user = (data.get('chatname') or data.get('user') or 
                    data.get('username') or data.get('nickname') or 'Anon')
            raw_msg = data.get('chatmessage') or data.get('message') or ''
            platform = data.get('type', 'UNK')
            source = data.get('sourceName', '')
            
            clean_msg = clean_message_content(raw_msg)
            
            # --- IGNORED PHRASES CHECK ---
            for ignore_txt in IGNORED_PHRASES:
                if ignore_txt.lower() in user.lower() or ignore_txt.lower() in clean_msg.lower():
                     print(f"Ignored message from {user} due to filter: {ignore_txt}")
                     return jsonify({"status": "ignored_filter"}), 200

            total_post_count += 1
            
            raw_ts = data.get('timestamp')
            if raw_ts:
                dt = datetime.datetime.fromtimestamp(int(raw_ts) / 1000.0)
                time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            readable_line = f"[{time_str}] [{platform}] [{source}] {user}: {clean_msg}\n"
            
            score_updated = False

            # 1. PROCESS MENTIONS
            for group in TARGET_GROUPS:
                display_word = group[0].strip()
                hits = count_group_hits(clean_msg, group)
                if hits > 0:
                    LEADERBOARD_SCORES[display_word] += hits
                    score_updated = True
                    print(f"   >>> MENTION: {display_word} (+h {hits})")
                    try:
                        safe_name = "".join(c for c in display_word if c.isalnum() or c in (' ', '-', '_')).strip()
                        if not safe_name: safe_name = "Unknown"
                        log_path = os.path.join(MENTIONS_LOG_DIR, f"{safe_name}.txt")
                        async with aiofiles.open(log_path, mode='a', encoding='utf-8') as f_mention:
                            await f_mention.write(readable_line)
                    except Exception as e:
                        print(f"Error saving to {safe_name}.txt: {e}")

            # 2. PROCESS SHIT TALKERS
            st_score = 0
            for group in SHIT_TALKER_GROUPS:
                hits = count_group_hits(clean_msg, group)
                if hits > 0: st_score += 1
            
            if st_score > 0:
                SHIT_TALKER_SCORES[user] += st_score
                score_updated = True
                print(f"   >>> SHIT TALK: {user} (+{st_score})")

            if score_updated:
                LAST_SCORE_TIME = time.time()

            live_messages.appendleft(data)

            try:
                # Save to Session Log
                async with aiofiles.open(JSON_LOG_FILE, mode='a', encoding='utf-8') as f_json:
                    await f_json.write(json.dumps(data, ensure_ascii=False) + "\n")
                
                # Save to Readable Log
                async with aiofiles.open(TEXT_LOG_FILE, mode='a', encoding='utf-8') as f_txt:
                    await f_txt.write(readable_line)
                
                # Save to MASTER LOG (Text Format)
                async with aiofiles.open(MASTER_LOG_FILE, mode='a', encoding='utf-8') as f_master:
                    await f_master.write(readable_line)

            except Exception as e: print(f"Save Error: {e}")
            
            return jsonify({"status": "success"}), 200
        else: return jsonify({"status": "error"}), 400
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/live', methods=['GET'])
def get_live_data():
    active_mentions = []
    for group in TARGET_GROUPS:
        if not group: continue
        display_word = group[0].strip()
        score = LEADERBOARD_SCORES.get(display_word, 0)
        if score > 0: active_mentions.append((display_word, score))
    active_mentions.sort(key=lambda x: x[1], reverse=True)
    active_talkers = SHIT_TALKER_SCORES.most_common(5)

    return jsonify({
        "messages": list(live_messages),
        "post_count": total_post_count,
        "leaderboard_mentions": active_mentions[:5],
        "leaderboard_talkers": active_talkers,
        "groups_mentions": TARGET_GROUPS, 
        "groups_talkers": SHIT_TALKER_GROUPS,
        "groups_ignored": IGNORED_PHRASES, 
        "yt_key": YOUTUBE_API_KEY,
        "yt_channels": YOUTUBE_CHANNELS, 
        "upcoming_streams": UPCOMING_STREAMS,
        "last_score_time": LAST_SCORE_TIME
    })

# --- ALL TIME STATS (TEXT BASED) ---
@app.route('/api/all_time', methods=['GET'])
def get_all_time_stats():
    if not os.path.exists(MASTER_LOG_FILE):
        return jsonify({"mentions": [], "talkers": []})
    
    at_mentions = Counter()
    at_talkers = Counter()

    try:
        # Regex to parse: [Time] [Plat] [Src] User: Message
        line_pattern = re.compile(r'^\[.*?\] \[.*?\] \[.*?\] (.*?): (.*)$')

        with open(MASTER_LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                match = line_pattern.match(line)
                if match:
                    user = match.group(1).strip()
                    msg_content = match.group(2).strip()
                    
                    for group in TARGET_GROUPS:
                        hits = count_group_hits(msg_content, group)
                        if hits > 0:
                            at_mentions[group[0].strip()] += hits
                    
                    st_score = 0
                    for group in SHIT_TALKER_GROUPS:
                        hits = count_group_hits(msg_content, group)
                        if hits > 0: st_score += 1
                    
                    if st_score > 0:
                        at_talkers[user] += st_score
        
        top_mentions = []
        for group in TARGET_GROUPS:
            if not group: continue
            display_word = group[0].strip()
            score = at_mentions.get(display_word, 0)
            if score > 0: top_mentions.append((display_word, score))
        top_mentions.sort(key=lambda x: x[1], reverse=True)
        
        top_talkers = at_talkers.most_common(5)

        return jsonify({
            "mentions": top_mentions[:5],
            "talkers": top_talkers
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/all_time_top10')
def all_time_top10_page():
    return render_template('all_time_leaderboard_top10.html')

@app.route('/api/all_time_top10', methods=['GET'])
def get_all_time_stats_top10():
    if not os.path.exists(MASTER_LOG_FILE):
        return jsonify({"mentions": [], "talkers": []})
    at_mentions, at_talkers = Counter(), Counter()
    try:
        line_pattern = re.compile(r'^\[.*?\] \[.*?\] \[.*?\] (.*?): (.*)$')
        with open(MASTER_LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                match = line_pattern.match(line)
                if match:
                    user, msg_content = match.group(1).strip(), match.group(2).strip()
                    for group in TARGET_GROUPS:
                        hits = count_group_hits(msg_content, group)
                        if hits > 0: at_mentions[group[0].strip()] += hits
                    if sum(1 for group in SHIT_TALKER_GROUPS if count_group_hits(msg_content, group) > 0) > 0:
                        at_talkers[user] += 1
        top_mentions = sorted(at_mentions.items(), key=lambda x: x[1], reverse=True)
        return jsonify({"mentions": top_mentions[:10], "talkers": at_talkers.most_common(10)})
    except Exception as e: return jsonify({"error": str(e)})

@app.route('/api/reset_session', methods=['POST'])
def reset_session():
    global total_post_count, LEADERBOARD_SCORES, SHIT_TALKER_SCORES, live_messages, PROCESSED_IDS
    try:
        total_post_count = 0
        LEADERBOARD_SCORES.clear()
        SHIT_TALKER_SCORES.clear()
        live_messages.clear()
        PROCESSED_IDS.clear()
        print("[RESET] Session cleared by user.")
        return jsonify({"status": "success"})
    except Exception as e: return jsonify({"status": "error"}), 500

@app.route('/api/clear_logs', methods=['POST'])
def clear_logs():
    global total_post_count, LEADERBOARD_SCORES, SHIT_TALKER_SCORES, live_messages, PROCESSED_IDS
    try:
        open(JSON_LOG_FILE, 'w').close() 
        open(TEXT_LOG_FILE, 'w').close() 
        total_post_count = 0
        LEADERBOARD_SCORES.clear()
        SHIT_TALKER_SCORES.clear()
        live_messages.clear()
        PROCESSED_IDS.clear()
        print("[LOGS] Session files cleared. Master Log intact.")
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/set_words', methods=['POST'])
def set_target_words():
    global TARGET_GROUPS, SHIT_TALKER_GROUPS, IGNORED_PHRASES, YOUTUBE_API_KEY, YOUTUBE_CHANNELS
    try:
        data = request.json
        should_reset = data.get('reset', False)

        if 'mentions' in data:
            raw = data['mentions']
            TARGET_GROUPS = [[w.strip() for w in g if w.strip()] for g in raw if g]
            if should_reset: LEADERBOARD_SCORES.clear()

        if 'talkers' in data:
            raw = data['talkers']
            SHIT_TALKER_GROUPS = [[w.strip() for w in g if w.strip()] for g in raw if g]
            if should_reset: SHIT_TALKER_SCORES.clear()
            
        if 'ignored' in data:
            raw = data['ignored']
            IGNORED_PHRASES = [w.strip() for w in raw if w.strip()]
        
        if 'yt_key' in data: YOUTUBE_API_KEY = data['yt_key'].strip()
        if 'yt_channels' in data: 
            YOUTUBE_CHANNELS = [c.strip() for c in data['yt_channels'] if c.strip()]
            
        save_config()
        return jsonify({"status": "success"})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})

@app.route('/api/extract', methods=['POST'])
def extract_logs():
    try:
        data = request.json
        filename = data.get('filename')
        keyword = data.get('keyword', '').strip().lower()
        if not filename or not keyword: return jsonify({"status": "error", "message": "Missing info"})
        
        full_path = get_file_path(filename)
        if not os.path.exists(full_path): return jsonify({"status": "error", "message": "File not found"})

        safe_keyword = "".join(c for c in keyword if c.isalnum() or c in (' ', '-', '_')).strip()
        output_filename = f"extracted_{safe_keyword}.txt"
        output_path = os.path.join(EXTRACT_LOG_DIR, output_filename)
        
        count = 0
        with open(full_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                if keyword in line.lower():
                    f_out.write(line)
                    count += 1
        return jsonify({"status": "success", "count": count, "file": output_path})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})

@app.route('/api/transcribe', methods=['POST'])
def transcribe_video():
    if not audio_model: return jsonify({"status": "error", "message": "Whisper not installed"})
    if 'file' not in request.files: return jsonify({"status": "error", "message": "No file"})
    file = request.files['file']
    if file.filename == '': return jsonify({"status": "error", "message": "No file selected"})
    try:
        temp = "temp.mp4"
        file.save(temp)
        res = audio_model.transcribe(temp, fp16=False)
        os.remove(temp)
        return jsonify({"status": "success", "text": res['text']})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})

@app.route('/api/files', methods=['GET'])
def list_log_files():
    try:
        files = []
        if os.path.exists(JSON_LOG_FILE): files.append(JSON_LOG_FILE)
        if os.path.exists(TEXT_LOG_FILE): files.append(TEXT_LOG_FILE)
        if os.path.exists(MASTER_LOG_FILE): files.append(MASTER_LOG_FILE) 
        
        if os.path.exists(OLD_LOGS_DIR):
            for f in os.listdir(OLD_LOGS_DIR):
                if f.endswith('.jsonl') or f.endswith('.txt'):
                    files.append(f)
        
        files.sort(key=lambda x: os.path.getmtime(get_file_path(x)), reverse=True)
        return jsonify(files)
    except: return jsonify([])

@app.route('/api/logs', methods=['GET'])
async def get_log_file():
    target_file = request.args.get('filename', JSON_LOG_FILE)
    
    full_path = get_file_path(target_file)
    
    if not os.path.exists(full_path): return jsonify([])
    lines = []
    try:
        async with aiofiles.open(full_path, mode='r', encoding='utf-8') as f:
            async for line in f:
                if line.strip():
                    try: 
                        lines.append(json.loads(line))
                    except: 
                        lines.append({'message': line.strip(), 'user': 'Log', 'type': 'txt'})
    except: pass
    return jsonify(lines[::-1])

@app.route('/api/append_to_master', methods=['POST'])
async def append_to_master_log():
    try:
        data = request.json
        filename_to_append = data.get('filename')

        if not filename_to_append:
            return jsonify({"status": "error", "message": "Filename not provided."}), 400

        source_path = get_file_path(filename_to_append)

        if not os.path.exists(source_path):
            return jsonify({"status": "error", "message": f"Source file '{filename_to_append}' not found."}), 404
            
        if os.path.abspath(source_path) == os.path.abspath(MASTER_LOG_FILE):
             return jsonify({"status": "error", "message": "Cannot append the master log to itself."}), 400

        async with aiofiles.open(source_path, mode='r', encoding='utf-8') as f_source:
            content_to_append = await f_source.read()
        
        lines_appended = len(content_to_append.strip().split('\n')) if content_to_append.strip() else 0

        if lines_appended == 0:
             return jsonify({"status": "success", "message": "Source file is empty. Nothing to append.", "lines": 0})

        # --- START: CORRECTED LOGIC ---
        # Get the file size using the standard os library
        master_log_size = os.path.getsize(MASTER_LOG_FILE)

        async with aiofiles.open(MASTER_LOG_FILE, mode='a+', encoding='utf-8') as f_master:
            if master_log_size > 0:
                # Seek to the second-to-last byte from the START of the file
                await f_master.seek(master_log_size - 1)
                last_char = await f_master.read(1)
                if last_char != '\n':
                    # If the last character isn't a newline, write one before appending
                    await f_master.write('\n')
            
            # Now append the new content (the file pointer is already at the end)
            await f_master.write(content_to_append)
            
            # And ensure the newly added content ends with a newline for next time
            if not content_to_append.endswith('\n'):
                 await f_master.write('\n')
        # --- END: CORRECTED LOGIC ---

        print(f"[MASTER LOG] Appended {lines_appended} lines from '{filename_to_append}'.")
        return jsonify({"status": "success", "message": f"Successfully appended content from {filename_to_append}.", "lines": lines_appended})

    except Exception as e:
        print(f"CRITICAL Error in append_to_master_log: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(JSON_LOG_FILE): open(JSON_LOG_FILE, 'w').close()
    if not os.path.exists(TEXT_LOG_FILE): open(TEXT_LOG_FILE, 'w').close()
    if not os.path.exists(MASTER_LOG_FILE): open(MASTER_LOG_FILE, 'w').close() 
    if not os.path.exists(MENTIONS_LOG_DIR): os.makedirs(MENTIONS_LOG_DIR)
    if not os.path.exists(EXTRACT_LOG_DIR): os.makedirs(EXTRACT_LOG_DIR)
    if not os.path.exists(OLD_LOGS_DIR): os.makedirs(OLD_LOGS_DIR)
    
    load_config()
    
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        t = threading.Thread(target=youtube_monitor_loop, daemon=True)
        t.start()
        print(f"[{ts}] --- Server Starting ---")
        print(f"[{ts}] Saving mention clips to folder: ./{MENTIONS_LOG_DIR}/")
    
    app.run(host='127.0.0.1', port=5000, debug=True)