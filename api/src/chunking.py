from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_text_splitter(splitter_type: str, **kwargs) -> RecursiveCharacterTextSplitter:
    if splitter_type == "RecursiveCharacterTextSplitter":
        splitter = RecursiveCharacterTextSplitter(**kwargs)
    return splitter  # type: ignore


def postprocess_chunk(chunk: str) -> str:
    chunk = chunk.lstrip(". ")
    chunk = chunk.capitalize()
    chunk += "."
    return chunk
