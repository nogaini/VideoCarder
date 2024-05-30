from api.models import SummaryDict

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import (
    InMemoryEmbeddingRetriever,
    InMemoryBM25Retriever,
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import MetaFieldRanker, TransformersSimilarityRanker
from haystack.components.samplers import TopPSampler
from haystack import Pipeline


def load_components(
    merged_segments: list[dict],
) -> tuple[
    SentenceTransformersDocumentEmbedder | SentenceTransformersTextEmbedder,
    InMemoryEmbeddingRetriever,
    TopPSampler,
    MetaFieldRanker,
]:
    doc_embedder = load_embedder(
        embedder_type="SentenceTransformersDocumentEmbedder",
        model="sentence-transformers/all-mpnet-base-v2",
    )

    docs = prepare_docs(merged_segments)
    docs_with_embeddings = get_doc_embeddings(doc_embedder, docs)
    document_store = load_document_store(store_type="InMemoryDocumentStore")
    document_store.write_documents(docs_with_embeddings["documents"])

    retriever = load_retriever(
        retriever_type="InMemoryEmbeddingRetriever",
        document_store=document_store,
        top_k=1,
    )

    retriever_bm25 = load_retriever(
        retriever_type="InMemoryBM25Retriever",
        document_store=document_store,
        top_k=2,
    )

    query_embedder = load_embedder(
        embedder_type="SentenceTransformersTextEmbedder",
        model="sentence-transformers/all-mpnet-base-v2",
    )

    ranker = load_ranker(
        ranker_type="MetaFieldRanker", meta_field="start", sort_order="ascending"
    )

    ranker_similarity = load_ranker(
        ranker_type="TransformersSimilarityRanker", model="BAAI/bge-reranker-base"
    )
    sampler = load_sampler(sampler_type="TopPSampler", top_p=0.95)

    joiner = load_joiner(
        joiner_type="DocumentJoiner", join_mode="reciprocal_rank_fusion"
    )

    document_store = load_document_store(store_type="InMemoryDocumentStore")
    document_store.write_documents(docs_with_embeddings["documents"])

    pipeline_components_dict = {
        "query_embedder": query_embedder,
        "retriever": retriever,
        "sampler": sampler,
        "ranker": ranker,
        "retriever_bm25": retriever_bm25,
        "joiner": joiner,
        "ranker_similarity": ranker_similarity,
    }

    return pipeline_components_dict


def load_joiner(joiner_type: str, **kwargs) -> DocumentJoiner:
    if joiner_type == "DocumentJoiner":
        joiner = DocumentJoiner(**kwargs)
    return joiner


def load_embedder(
    embedder_type: str, **kwargs
) -> SentenceTransformersDocumentEmbedder | SentenceTransformersTextEmbedder:
    if embedder_type == "SentenceTransformersDocumentEmbedder":
        embedder = SentenceTransformersDocumentEmbedder(**kwargs)
        embedder.warm_up()
    if embedder_type == "SentenceTransformersTextEmbedder":
        embedder = SentenceTransformersTextEmbedder(**kwargs)
    return embedder


def load_document_store(store_type: str) -> InMemoryDocumentStore:
    if store_type == "InMemoryDocumentStore":
        document_store = InMemoryDocumentStore()
    return document_store


def load_retriever(
    retriever_type: str, document_store, **kwargs
) -> InMemoryEmbeddingRetriever:
    if retriever_type == "InMemoryEmbeddingRetriever":
        retriever = InMemoryEmbeddingRetriever(document_store, **kwargs)
    if retriever_type == "InMemoryBM25Retriever":
        retriever = InMemoryBM25Retriever(document_store, **kwargs)
    return retriever


def load_ranker(ranker_type: str, **kwargs) -> MetaFieldRanker:
    if ranker_type == "MetaFieldRanker":
        ranker = MetaFieldRanker(**kwargs)
    if ranker_type == "TransformersSimilarityRanker":
        ranker = TransformersSimilarityRanker(**kwargs)
    return ranker


def load_sampler(sampler_type: str, **kwargs) -> TopPSampler:
    if sampler_type == "TopPSampler":
        sampler = TopPSampler(**kwargs)
    return sampler


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
        meta_dict["duration"] = segment["end"] - segment["start"]
        meta_dict["words"] = segment["words"]
        meta_dict["no_speech_prob"] = segment["no_speech_prob"]
        doc["meta"] = meta_dict

        docs.append(doc)
    docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in docs]
    return docs


def load_pipeline(pipeline_components_dict: dict[str]) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("query_embedder", pipeline_components_dict["query_embedder"])
    pipeline.add_component("retriever", pipeline_components_dict["retriever"])
    pipeline.add_component("retriever_bm25", pipeline_components_dict["retriever_bm25"])
    pipeline.add_component("sampler", pipeline_components_dict["sampler"])
    pipeline.add_component("joiner", pipeline_components_dict["joiner"])
    pipeline.add_component(
        "ranker_similarity", pipeline_components_dict["ranker_similarity"]
    )

    pipeline.connect("query_embedder", "retriever")
    pipeline.connect("retriever", "joiner")
    pipeline.connect("retriever_bm25", "joiner")
    pipeline.connect("joiner", "ranker_similarity")
    pipeline.connect("ranker_similarity.documents", "sampler.documents")
    return pipeline


def deduplicate_summary_dicts_list(
    summary_dicts: list[SummaryDict],
) -> list[SummaryDict]:
    unique_ids = set()
    dedup_summary_dicts_list = []

    # Iterate over each potential VideoCard (each dict contains title and bullets)
    for summary_dict in summary_dicts:
        dedup_summary_dict = {}
        dedup_summary_dict["title"] = summary_dict["title"]
        dedup_summary_dict["bullets"] = []

        # Iterate over each bullet in the VideoCard (each dict contains bullet and corresponding retrieved docs)
        for bullet_dict in summary_dict["bullets"]:
            dedup_bullet_dict = {}
            dedup_bullet_dict["bullet"] = bullet_dict["bullet"]
            dedup_bullet_dict["retrieved"] = []

            retrieved_docs = bullet_dict["retrieved"]

            # Iterate over each retrieved doc for the bullet
            for doc in retrieved_docs:
                # Add the doc only if it doesn't appear anywhere else in the results
                if doc.id not in unique_ids:
                    unique_ids.add(doc.id)
                    dedup_bullet_dict["retrieved"].append(doc)
            # Add the bullet only if it has at least 1 unique retrieved doc
            if dedup_bullet_dict["retrieved"]:
                dedup_summary_dict["bullets"].append(dedup_bullet_dict)

        # Add the VideoCard only if it has at least 1 bullet
        if dedup_summary_dict["bullets"]:
            dedup_summary_dicts_list.append(dedup_summary_dict)

    return dedup_summary_dicts_list


def retrieve_for_query(
    query: str, ranker: MetaFieldRanker, pipeline: Pipeline
) -> list[Document]:
    sampled_docs = []
    response = pipeline.run(
        {
            "query_embedder": {"text": query},
            "retriever_bm25": {"query": query},
            "ranker_similarity": {"query": query},
        }
    )

    # response = pipeline.run({"query_embedder": {"text": query}})
    # response = pipeline.run({"query_embedder": {"text": query},
    #                          "retriever": {"filters": {"field": "meta.duration", "operator": ">", "value": 2}}})
    sampled_docs = response["sampler"]["documents"]
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


def enrich_summary_dicts(
    summary_dicts: list[SummaryDict], **kwargs
) -> list[SummaryDict]:
    enriched_summary_dicts = []
    for summary_dict in summary_dicts:
        enriched_dict = enrich_summary_dict(summary_dict, **kwargs)
        enriched_summary_dicts.append(enriched_dict)
    enriched_summary_dicts = deduplicate_summary_dicts_list(enriched_summary_dicts)
    return enriched_summary_dicts
