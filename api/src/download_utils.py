import yt_dlp
from pathlib import Path


def download_audio(url: str, save_dir: Path, save_name: str) -> str:
    """
    Downloads an audio from a youtube video URL.

    Parameters:
        url (str): The URL of the YouTube video to be downloaded.
        save_dir (Path): The directory path where the audio will be saved.
        save_name (str): The base name for the output file, without extension.
                          The function will append '.mp3' automatically.

    Returns:
        str: The file path of the downloaded audio as a string.

    Note that the user should ensure that save_dir exists and has write permission,
    otherwise the function may fail to create output file.
    """

    output_path = Path(f"{save_dir}/{save_name}.mp3")
    with yt_dlp.YoutubeDL(
        {"extract_audio": True, "format": "bestaudio", "outtmpl": f"{output_path}"}
    ) as video:
        video.download(url)
    return str(output_path)
