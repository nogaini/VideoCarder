import stable_whisper
from whisper.model import Whisper


def load_stable_whisper_model(model_name: str) -> Whisper:
    """
    Loads a stable Whisper model based on its name.

    Parameters:
        model_name (str): The name of the Whisper model to be loaded. It should match one of
                           the available models in the library, such as 'small', 'medium', etc.

    Returns:
        Whisper: An instance of the stable-whisper's `Whisper` class representing the loaded model.

    """
    model = stable_whisper.load_model(model_name)
    return model


def get_transcript_from_result_dict(result_dict: dict) -> str:
    """
    Extracts the transcript from a dictionary resulting from Whisper's transcription.

    Parameters:
        result_dict (dict): The dictionary containing the transcript information returned by
                            Whisper's transcription method. It should contain at least a key "text"
                            with the value being the transcript as string.

    Returns:
        str: A string representing the transcript of the input audio.

    """
    transcript = result_dict["text"].strip()
    return transcript


def preprocess_segments(segments: list[dict]) -> list[dict]:
    merged_segments = merge_segments(segments)
    filtered_segments = filter_segments_by_duration(merged_segments)
    return filtered_segments


def filter_segments_by_duration(
    segments: list[dict], duration_threshold: float = 2
) -> list[dict]:
    filtered_segments = []
    for segment in segments:
        if (segment["end"] - segment["start"]) < duration_threshold:
            continue
        filtered_segments.append(segment)
    return filtered_segments


def merge_segments(segments: list) -> list:
    start_idx = 0
    end_idx = 0
    merged_segments = []
    while end_idx < len(segments):
        segment = segments[end_idx]
        if segment["text"].endswith(".") or end_idx == len(segments) - 1:
            merged_segment = merge_segments_by_idx(segments, start_idx, end_idx)
            merged_segments.append(merged_segment)
            start_idx = end_idx + 1
            end_idx = start_idx
        else:
            end_idx += 1
    return merged_segments


def merge_segments_by_idx(segments: list[dict], start_idx: int, end_idx: int) -> dict:
    if start_idx == end_idx:
        return segments[start_idx]
    merged_segment = {}
    merged_text = ""
    merged_words = []
    no_speech_probs = []
    for idx in range(start_idx, end_idx + 1):
        merged_text += segments[idx]["text"]
        merged_words.append(segments[idx]["words"])
        no_speech_probs.append(segments[idx]["no_speech_prob"])
    merged_segment["text"] = merged_text
    merged_segment["start"] = segments[start_idx]["start"]
    merged_segment["end"] = segments[end_idx]["end"]
    merged_segment["words"] = merged_words
    merged_segment["no_speech_prob"] = no_speech_probs
    return merged_segment
