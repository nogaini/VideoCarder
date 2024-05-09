# [FRONTEND async] User provides YouTube video link
# [BACKEND] Download audio corresponding to video
# [BACKEND] Transcribe downloaded audio file
# [BACKEND async] Preprocess transcript results dict
# [BACKEND] Chunk the transcript
# [BACKEND] Summarize each chunk to get summary chunk
# [BACKEND] Bulletize each summary chunk to get summary JSONs
# [BACKEND async] Load the retrieval pipeline
# [BACKEND async] Enrich each summary JSON using the pipeline
# [FRONTEND] Fetch enriched summary JSONs
# [FRONTEND] Display data from each enriched summary JSON

from src.transcript_generation import load_stable_whisper_model

transcription_model = load_stable_whisper_model("tiny")

async def predict(url):
    audio_path = download_audio(url: str, save_dir: Path, save_name: str)
    
    result = transcription_model.transcribe(audio_path, word_timestamps=True)
    result_dict = result.to_dict()
    transcript = result_dict["text"].strip()