import yt_dlp
import os
import whisper

# Set FFmpeg path for this session
os.environ["PATH"] += os.pathsep + r"D:\ffmpeg-2025-12-31-git-38e89fe502-essentials_build\ffmpeg-2025-12-31-git-38e89fe502-essentials_build\bin"

def download_video_and_extract_audio(url):
    """Download video from URL and extract audio as MP3"""
    output_path = "temp_audio.mp3"
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ffmpeg_location': r"D:\ffmpeg-2025-12-31-git-38e89fe502-essentials_build\ffmpeg-2025-12-31-git-38e89fe502-essentials_build\bin",
        'quiet': False,
        'no_warnings': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Audio file not created: {output_path}")
            
        return output_path
    except Exception as e:
        print(f"Error downloading video: {e}")
        raise

def transcribe_audio(file_path):
    """Transcribe audio file using Whisper"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        
        print(f"Transcribing {file_path}...")
        result = model.transcribe(file_path)
        
        return result['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise

def cleanup_temp_files():
    """Remove temporary audio files"""
    temp_files = ['temp_audio.mp3', 'temp_audio.webm', 'temp_audio.m4a']
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up {file}")

# Main execution
if __name__ == "__main__":
    # Example usage
    video_url = input("Enter YouTube URL: ")
    
    try:
        # Step 1: Download and extract audio
        print("\n--- Downloading video and extracting audio ---")
        audio_file = download_video_and_extract_audio(video_url)
        print(f"✓ Audio extracted: {audio_file}")
        
        # Step 2: Transcribe audio
        print("\n--- Transcribing audio ---")
        transcript = transcribe_audio(audio_file)
        print("\n--- Transcript ---")
        print(transcript)
        
        # Step 3: Save transcript to file
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)
        print("\n✓ Transcript saved to transcript.txt")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    finally:
        # Step 4: Cleanup temporary files
        print("\n--- Cleaning up ---")
        cleanup_temp_files()