from typing import Literal
import json
from llama_cpp import Llama
import ollama


BULLETIZATION_SYSTEM_PROMPT = """You are a helpful assistant that outputs in JSON. You are an expert at converting text into short points. Convert the given text into short points and a title. Don't add escape characters. Directly list the points and the title, don't add additional text before or after it."""
BULLETIZATION_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "bullets": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "bullets"],
}


def load_llamacpp_llm(gguf_path: str) -> Llama:
    llm = Llama(model_path=gguf_path, chat_format="chatml", verbose=False)
    return llm


def llamacpp_chat(
    llm: Llama, system_prompt: str, user_prompt: str, schema: dict
) -> str:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_object",
            "schema": schema,
        },
        temperature=0.0,
    )
    return response["choices"][0]["message"]["content"]  # type: ignore


def ollama_chat(model_name: str, prompt: str, format: Literal["", "json"] = "") -> str:
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        stream=False,
        format=format,
    )
    return response["message"]["content"]  # type: ignore


def summarize_chunk(chunk: str) -> str:
    return ollama_chat("llama3_instruct_podcast_chunk_summarizer", chunk)


def bulletize_summary(summary: str, llm: Llama) -> str:
    response_text = llamacpp_chat(
        llm,
        system_prompt=BULLETIZATION_SYSTEM_PROMPT,
        user_prompt=summary,
        schema=BULLETIZATION_SCHEMA,
    )
    return json.loads(response_text)

def chunk_to_summary(chunk: str, llm: Llama):
    summary = summarize_chunk(chunk)
    summary_json = bulletize_summary(summary, llm)
    return summary_json


def chunks_to_summaries(chunks: list[str], llm: Llama) -> list[dict]:
    summary_json_list = []
    for chunk in chunks:
        summary_json = chunk_to_summary(chunk, llm)
        summary_json_list.append(summary_json)
    return summary_json_list
