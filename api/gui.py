from pathlib import Path
from nicegui import ui
import asyncio
import concurrent.futures
from functools import partial
from multiprocessing import Manager, Queue

from src.speech_recognition import (
    load_stable_whisper_model,
    get_transcript_from_result_dict,
    preprocess_segments,
)
from src.downloading import download_audio, download_video
from src.chunking import load_text_splitter, postprocess_chunk
from src.summarization import load_llamacpp_llm, chunks_to_summaries
from src.retrieval import (
    load_components,
    load_pipeline,
    enrich_summary_dicts,
)
from src.trimming import generate_merged_video_for_bullet_dicts
from src.file_utils import cleanup
from models import SummaryDict

transcription_model = load_stable_whisper_model("tiny")
text_splitter = load_text_splitter(
    "RecursiveCharacterTextSplitter",
    separators=["."],
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

SUMMARIZER_LLM_GGUF_PATH = (
    "/home/jobin/Projects/transcript_summarizer/gguf/Meta-Llama-3-8B-Instruct.Q6_K.gguf"
)
INPUT_VIDEO_FILES_PATH = Path("static/input/video")
INPUT_AUDIO_FILES_PATH = Path("static/input/audio")
TRIM_FILES_PATH = Path("static/trims")
MERGED_FILES_PATH = Path("static/merged")

summarizer_llm = load_llamacpp_llm(SUMMARIZER_LLM_GGUF_PATH)


class VideoCard:
    def __init__(self, video_path: str, title: str, bullets: list[str]) -> None:
        with ui.card().tight().style("max-width: 500px;"):
            ui.video(video_path)
            with ui.card_section():
                ui.label(title).props("class=text-h6")
            with ui.card_section():
                with ui.list().props("dense separator"):
                    for bullet in bullets:
                        ui.item(bullet)
            with ui.card_actions():
                ui.button("", on_click=lambda: ui.download(video_path)).props(
                    "color=green icon=download"
                )


def predict(url: str, q: Queue) -> list[SummaryDict]:
    audio_path = download_audio(url, save_dir=INPUT_AUDIO_FILES_PATH)
    result = transcription_model.transcribe(audio_path, word_timestamps=True)
    result_dict = result.to_dict()  # type: ignore
    filtered_segments = preprocess_segments(result_dict["segments"])

    transcript = get_transcript_from_result_dict(result_dict)
    transcript_chunks = text_splitter.split_text(transcript)
    transcript_chunks = [postprocess_chunk(x) for x in transcript_chunks]
    q.put_nowait(0.25)
    summary_json_list = chunks_to_summaries(transcript_chunks, llm=summarizer_llm)
    q.put_nowait(0.5)
    pipeline_components_dict = load_components(filtered_segments)
    pipeline = load_pipeline(pipeline_components_dict)
    enriched_summary_dicts = enrich_summary_dicts(
        summary_json_list, ranker=pipeline_components_dict["ranker"], pipeline=pipeline
    )
    q.put_nowait(0.75)
    return enriched_summary_dicts


@ui.page("/")
def main_page():
    async def render_results(
        summary_dicts: list[SummaryDict], video_path: str, q: Queue
    ):
        loop = asyncio.get_running_loop()
        for merge_idx, summary_dict in enumerate(summary_dicts):
            title = summary_dict["title"]
            bullets = [d["bullet"] for d in summary_dict["bullets"]]
            with concurrent.futures.ProcessPoolExecutor() as pool:
                merged_video_path = await loop.run_in_executor(
                    pool,
                    partial(
                        generate_merged_video_for_bullet_dicts,
                        video_path,
                        summary_dict["bullets"],
                        merge_idx,
                        TRIM_FILES_PATH,
                        MERGED_FILES_PATH,
                    ),
                )
                with results_container:
                    VideoCard(merged_video_path, title, bullets)
        q.put_nowait(1.0)

    async def on_submit(url: str):
        # Clear previous results
        if results_container:
            results_container.clear()

        cleanup(MERGED_FILES_PATH)

        # Reset progress bar
        progressbar.value = 0.1
        progressbar.visible = True

        # Put request in process pool to not block UI
        loop = asyncio.get_running_loop()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            video_path = await loop.run_in_executor(
                pool, partial(download_video, url, save_dir=INPUT_VIDEO_FILES_PATH)
            )

        with concurrent.futures.ProcessPoolExecutor() as pool:
            summary_dicts = await loop.run_in_executor(
                pool, partial(predict, url, queue)
            )
            await render_results(summary_dicts, video_path, queue)
        progressbar.visible = False

    # Create a queue to communicate with the CPU-bound process
    queue = Manager().Queue()

    # Update progress bar every 0.1 seconds
    ui.timer(
        0.1,
        callback=lambda: progressbar.set_value(
            queue.get() if not queue.empty() else progressbar.value
        ),
    )
    ui.page_title("VideoCarder")

    with ui.row().classes("mx-auto"):
        ui.label("VideoCarder").props("class=text-h2").style("font-weight: bold;")
    with ui.row().classes("mx-auto"):
        ui.label("Visualize transcript-based videos as summary cards").props(
            "class=text-h3"
        )
    with ui.row().classes("mx-auto"):
        url_input_element = (
            ui.input(label="Try a YouTube URL")
            .props("dense outlined")
            .style("min-width: 500px;")
        )
        ui.button(
            text="Summarize",
            on_click=lambda: partial(on_submit, url_input_element.value)(),
        )

    progressbar = ui.linear_progress(value=0.1, show_value=False).props(
        'stripe rounded size="20px" color="cyan"'
    )

    progressbar.visible = False
    with ui.row().classes("mx-auto"):
        results_container = ui.row().classes("mx-auto")


ui.run(favicon="🚀")
