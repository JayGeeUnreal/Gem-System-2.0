import sys
import os
import subprocess
import ytmusicapi

def run_download(query):
    """
    Finds a song on YouTube Music and downloads it as an MP3
    using the yt-dlp command-line tool.
    """
    try:
        # --- 1. Search for the song ---
        print(f"Searching for: '{query}'...")
        yt = ytmusicapi.YTMusic()
        search_results = yt.search(query)

        if not search_results:
            print("ERROR: No results found for your query.")
            return

        video_id = search_results[0]["videoId"]
        video_title = search_results[0]["title"]
        print(f"Found Video: '{video_title}' (ID: {video_id})")

        # --- 2. Download the song ---
        output_folder = "requests"
        os.makedirs(output_folder, exist_ok=True)
        
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"Preparing to download from: {video_url}")
        print("-" * 50)

        # Assumes yt-dlp executable is in the same folder or in system PATH
        yt_dlp_command = "yt-dlp.exe" if sys.platform == "win32" else "yt-dlp"

        command = [
            yt_dlp_command,
            "-x",
            "--audio-format", "mp3",
            "-P", output_folder,
            "--no-playlist",
            video_url
        ]
        
        # Use Popen to stream output in real-time
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')

        # Print each line of output as it comes
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
        
        process.wait() # Wait for the download to complete
        
        if process.returncode == 0:
            print("-" * 50)
            print("SUCCESS: Download complete.")
        else:
            print("-" * 50)
            print(f"ERROR: yt-dlp exited with error code {process.returncode}")

    except FileNotFoundError:
        print("ERROR: 'yt-dlp' command not found.")
        print("Please make sure the yt-dlp executable is in your system's PATH or in the same directory as the script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Get the search query from the command-line arguments
    if len(sys.argv) > 1:
        search_query = " ".join(sys.argv[1:])
        run_download(search_query)
    else:
        print("Usage: python download_worker.py <search query>")