from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.rankers import MetaFieldRanker
from haystack.components.samplers import TopPSampler
from haystack import Pipeline
from models import SummaryDict

def load_embedder(
    embedder_type: str, **kwargs
) -> SentenceTransformersDocumentEmbedder | SentenceTransformersTextEmbedder:
    if embedder_type == "SentenceTransformersDocumentEmbedder":
        embedder = SentenceTransformersDocumentEmbedder(**kwargs)
        embedder.warm_up()
    if embedder_type == "SentenceTransformersTextEmbedder":
        embedder = SentenceTransformersTextEmbedder(**kwargs)
    return embedder  # type: ignore


def load_document_store(store_type: str) -> InMemoryDocumentStore:
    if store_type == "InMemoryDocumentStore":
        document_store = InMemoryDocumentStore()
    return document_store  # type: ignore


def load_retriever(
    retriever_type: str, document_store, **kwargs
) -> InMemoryEmbeddingRetriever:
    if retriever_type == "InMemoryEmbeddingRetriever":
        retriever = InMemoryEmbeddingRetriever(document_store, **kwargs)
    return retriever  # type: ignore


def load_ranker(ranker_type: str, **kwargs) -> MetaFieldRanker:
    if ranker_type == "MetaFieldRanker":
        ranker = MetaFieldRanker(**kwargs)
    return ranker  # type: ignore


def load_sampler(sampler_type: str, **kwargs) -> TopPSampler:
    if sampler_type == "TopPSampler":
        sampler = TopPSampler(**kwargs)
    return sampler  # type: ignore


def get_doc_embeddings(
    doc_embedder: SentenceTransformersDocumentEmbedder
    | SentenceTransformersTextEmbedder,
    docs: list[Document],
) -> dict[str, list[Document]]:
    docs_with_embeddings = doc_embedder.run(documents=docs)  # type: ignore
    return docs_with_embeddings


def write_doc_embeddings_to_inmemory_store(
    docs_with_embeddings: list[Document], store: InMemoryDocumentStore
):
    store.write_documents(docs_with_embeddings["documents"])  # type: ignore


def prepare_docs(segments: list) -> list[Document]:
    docs = []
    for segment in segments:
        doc = {}
        doc["content"] = segment["text"].strip()

        meta_dict = {}
        meta_dict["start"] = segment["start"]
        meta_dict["end"] = segment["end"]
        meta_dict["words"] = segment["words"]
        meta_dict["no_speech_prob"] = segment["no_speech_prob"]
        doc["meta"] = meta_dict

        docs.append(doc)
    docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in docs]
    return docs


def load_pipeline(query_embedder, retriever, sampler) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("query_embedder", query_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("sampler", sampler)
    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "sampler.documents")
    return pipeline


def retrieve_for_query(
    query: str, ranker: MetaFieldRanker, pipeline: Pipeline
) -> list[Document]:
    unique_ids = set()
    sampled_docs = []
    response = pipeline.run({"query_embedder": {"text": query}})
    for response_doc in response["sampler"]["documents"]:
        if response_doc.id not in unique_ids:
            unique_ids.add(response_doc.id)
            sampled_docs.append(response_doc)
    ranked_docs = ranker.run(documents=sampled_docs)
    return ranked_docs["documents"]


def enrich_summary_dict(summary_dict: dict, **kwargs) -> dict:
    enriched_dict = {}
    enriched_dict["title"] = summary_dict["title"]
    enriched_dict["bullets"] = []
    summary_bullets = summary_dict["bullets"]
    for bullet in summary_bullets:
        bullet_dict = {}
        bullet_dict["bullet"] = bullet
        bullet_dict["retrieved"] = retrieve_for_query(bullet, **kwargs)
        enriched_dict["bullets"].append(bullet_dict)
    return enriched_dict


def enrich_summary_dicts(summary_dicts: list[dict], **kwargs) -> list[SummaryDict]:
    enriched_summary_dicts = []
    for summary_dict in summary_dicts:
        enriched_dict = enrich_summary_dict(summary_dict, **kwargs)
        enriched_summary_dicts.append(enriched_dict)
    return enriched_summary_dicts
