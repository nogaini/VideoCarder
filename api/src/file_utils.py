import os

def cleanup(folder_path: str) -> None:
    # Clean up temporary files
    for f in os.listdir(folder_path):
        os.remove(f'{folder_path}/{f}')
