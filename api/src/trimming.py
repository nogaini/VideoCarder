import os
import ffmpeg
from src.file_utils import cleanup

def get_video_duration(video_path: str):
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    return duration

def generate_video_trims(video_path: str, timestamps: list[tuple[float]], save_folder: str, bullet_idx: int, offset: float=0.5) -> str:
    os.makedirs(save_folder, exist_ok=True)
    trim_file_list_path = f'{save_folder}/trim_file_list.txt'
    temp_files = [f'trim_{i}_{bullet_idx}.mp4' for i in range(len(timestamps))]

    duration = get_video_duration(video_path)

    # Trim each segment
    for i, (start, end) in enumerate(timestamps):
        (
            ffmpeg
            .input(video_path, ss=max(start - offset, 0.0), to=min(end + offset, duration))
            .output(f'{save_folder}/{temp_files[i]}', vcodec='libx264', acodec='aac', strict='experimental')
            .global_args('-loglevel', 'error')
            .run(overwrite_output=True)
        )

    # Concatenate the trimmed segments
    with open(trim_file_list_path, 'a') as f:
        for temp_file in temp_files:
            f.write(f"file '{temp_file}'\n")
    return trim_file_list_path

def merge_trim_files(trim_file_list_path: str, save_folder: str, save_name: str) -> str:
    os.makedirs(save_folder, exist_ok=True)
    merged_video_path = f'{save_folder}/{save_name}'
    (
        ffmpeg
        .input(trim_file_list_path, format='concat', safe=0)
        .output(merged_video_path, vcodec='libx264', acodec='aac', strict='experimental')
        .global_args('-loglevel', 'error')
        .run(overwrite_output=True)
    )
    return merged_video_path

def generate_merged_video_for_bullet_dicts(video_path: str, bullet_dicts: list[dict], merged_file_idx: int, trims_save_folder: str, merged_files_save_folder: str) -> str:
    filename = video_path.split("/")[-1].rstrip(".mp4")
    for b_idx, bullet_dict in enumerate(bullet_dicts):
        timestamps = [] 
        for doc in bullet_dict["retrieved"]:
            timestamps.append((doc.meta["start"], doc.meta["end"]))
        trim_file_list_path = generate_video_trims(video_path, timestamps, trims_save_folder, b_idx)
    merged_video_path = merge_trim_files(trim_file_list_path, merged_files_save_folder, f'{filename}_{merged_file_idx}.mp4')
    cleanup(trims_save_folder)
    return merged_video_path
    