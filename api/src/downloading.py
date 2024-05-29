import yt_dlp
from pathlib import Path
import os
from pydantic import HttpUrl


def download_audio(url: HttpUrl, save_dir: Path, only_audio=True) -> str:
    """
    Downloads audio from a given YouTube video URL
    and saves it as an mp3 file.

    Parameters:
        url (HttpUrl): The URL of the YouTube video to be downloaded.
        save_dir (Path): The directory where the audio will be saved.
                          If this directory does not exist, it is created.
        save_name (str): The name that will be given to the saved mp3 file.
                          It's appended with '.mp3'.

    Returns:
        str: The path to the saved audio as a string.
    """

    os.makedirs(save_dir, exist_ok=True)
    ext = "mp3"
    output_template = Path(f"{save_dir}/%(id)s.{ext}")
    cfg = {
        "extract_audio": True,
        "format": "bestaudio",
        "outtmpl": f"{output_template}",
    }

    with yt_dlp.YoutubeDL(cfg) as video:
        info_dict = video.extract_info(url, download=True)
        video_id = info_dict["id"]

    output_path = f"{save_dir}/{video_id}.{ext}"
    return output_path

def download_video(url: HttpUrl, save_dir: Path) -> str:
    """
    Downloads video from a given YouTube video URL
    and saves it as an mp4 file.

    Parameters:
        url (HttpUrl): The URL of the YouTube video to be downloaded.
        save_dir (Path): The directory where the audio will be saved.
                          If this directory does not exist, it is created.
        save_name (str): The name that will be given to the saved mp3 file.
                          It's appended with '.mp3'.

    Returns:
        str: The path to the saved video as a string.
    """

    os.makedirs(save_dir, exist_ok=True)
    ext = "mp4"
    output_template = Path(f"{save_dir}/%(id)s.{ext}")
    cfg = {"format": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/mp4", "outtmpl": f"{output_template}"}

    with yt_dlp.YoutubeDL(cfg) as video:
        info_dict = video.extract_info(url, download=True)
        video_id = info_dict["id"]

    output_path = f"{save_dir}/{video_id}.{ext}"
    return output_path
