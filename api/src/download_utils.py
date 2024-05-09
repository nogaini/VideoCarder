import yt_dlp
from pathlib import Path
import os


def download_audio(url: str, save_dir: Path, save_name: str) -> str:
    """
    Downloads audio from a given YouTube video URL 
    and saves it as an mp3 file.

    Parameters:
        url (str): The URL of the YouTube video to be downloaded.
        save_dir (Path): The directory where the audio will be saved. 
                          If this directory does not exist, it is created.
        save_name (str): The name that will be given to the saved mp3 file. 
                          It's appended with '.mp3'.

    Returns:
        str: The path to the saved audio as a string.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    output_path = Path(f"{save_dir}/{save_name}.mp3")
    with yt_dlp.YoutubeDL(
        {"extract_audio": True, "format": "bestaudio", "outtmpl": f"{output_path}"}
    ) as video:
        video.download(url)
    return str(output_path)
