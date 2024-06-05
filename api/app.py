# User workflow:
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

from pathlib import Path
from src.speech_recognition import (
    load_stable_whisper_model,
    get_transcript_from_result_dict,
    preprocess_segments,
)
from src.downloading import download_audio
from src.chunking import load_text_splitter, postprocess_chunk
from src.summarization import load_llamacpp_llm, chunks_to_summaries
from src.retrieval import (
    load_components,
    load_pipeline,
    enrich_summary_dicts,
)
from fastapi import FastAPI

# from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware
from pydantic import HttpUrl
from models import SummaryDict

SUMMARIZER_LLM_GGUF_PATH = (
    "/home/jobin/Projects/transcript_summarizer/gguf/Meta-Llama-3-8B-Instruct.Q6_K.gguf"
)


transcription_model = load_stable_whisper_model("tiny")
text_splitter = load_text_splitter(
    "RecursiveCharacterTextSplitter",
    separators=["."],
    chunk_size=5000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)
summarizer_llm = load_llamacpp_llm(SUMMARIZER_LLM_GGUF_PATH)

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/summarize/")
async def predict(url: HttpUrl) -> list[SummaryDict]:
    audio_path = download_audio(url, save_dir=Path("api/static/"))
    result = transcription_model.transcribe(audio_path, word_timestamps=True)
    result_dict = result.to_dict()  # type: ignore
    filtered_segments = preprocess_segments(result_dict["segments"])

    transcript = get_transcript_from_result_dict(result_dict)
    transcript_chunks = text_splitter.split_text(transcript)
    transcript_chunks = [postprocess_chunk(x) for x in transcript_chunks]
    summary_json_list = chunks_to_summaries(transcript_chunks, llm=summarizer_llm)

    pipeline_components_dict = load_components(filtered_segments)
    pipeline = load_pipeline(pipeline_components_dict)
    enriched_summary_dicts = enrich_summary_dicts(
        summary_json_list, ranker=pipeline_components_dict["ranker"], pipeline=pipeline
    )
    return enriched_summary_dicts
