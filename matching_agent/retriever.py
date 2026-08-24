from pathlib import Path
from typing import Iterable

import faiss
import pandas as pd
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from .config import DEFAULT_EMBEDDING_MODEL, DEFAULT_VECTORSTORE_DIR
from .state import CandidateHit, RetrievalSource


def make_embeddings(model: str = DEFAULT_EMBEDDING_MODEL) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model)


def load_retriever(
    vectorstore_dir: Path = DEFAULT_VECTORSTORE_DIR,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    k: int = 5,
):
    embeddings = make_embeddings(embedding_model)
    db = FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatL2(1536),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    db = db.load_local(
        str(vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return db.as_retriever(search_kwargs={"k": k})


def _search_with_optional_scores(retriever, query: str, k: int):
    """Return ``(document, score)`` pairs for LangChain or test retrievers."""
    vectorstore = getattr(retriever, "vectorstore", None)
    if vectorstore is not None and hasattr(vectorstore, "similarity_search_with_score"):
        return vectorstore.similarity_search_with_score(query, k=k)

    results = retriever.invoke(query)
    return [(document, None) for document in results[:k]]


def collect_candidate_hits(
    queries: list[tuple[str, str]],
    username: str,
    retriever,
    *,
    k: int = 5,
) -> list[CandidateHit]:
    """Retrieve candidates while preserving query, rank, and distance provenance.

    A candidate returned by multiple queries is kept once and accumulates a source
    record for each retrieval path. Input/query order is preserved.
    """
    hits_by_key: dict[tuple[str, str], CandidateHit] = {}

    for query_kind, query in queries:
        for rank, (document, score) in enumerate(
            _search_with_optional_scores(retriever, query, k), start=1
        ):
            candidate_username = str(document.metadata.get("User", ""))
            if candidate_username == username:
                continue

            message = str(document.page_content)
            key = (candidate_username, message)
            source: RetrievalSource = {
                "query": query,
                "query_kind": query_kind,
                "rank": rank,
                "score": float(score) if score is not None else None,
            }
            if key not in hits_by_key:
                hits_by_key[key] = {
                    "username": candidate_username,
                    "message": message,
                    "sources": [source],
                }
            else:
                hits_by_key[key]["sources"].append(source)

    return list(hits_by_key.values())


def format_candidates(hits: Iterable[CandidateHit]) -> list[str]:
    return [f"{hit['username']}: {hit['message']}" for hit in hits]


def get_candidates(query: str, username: str, retriever) -> list[str]:
    """Backward-compatible single-query candidate helper."""
    if hasattr(query, "content"):
        query = query.content
    hits = collect_candidate_hits([("original", str(query))], username, retriever)
    return format_candidates(hits)


def build_vectorstore_from_excel(
    dataset_path: Path,
    output_dir: Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    message_col: str = "Message",
    user_col: str = "User",
    batch_size: int = 500,
) -> None:
    """Rebuild the FAISS vectorstore from the source Excel dataset with batching."""
    from tqdm import tqdm

    df = pd.read_excel(dataset_path)
    # Filter out empty or NaN messages
    df = df.dropna(subset=[message_col, user_col]).copy()
    df[message_col] = df[message_col].astype(str)
    df[user_col] = df[user_col].astype(str)

    embeddings = make_embeddings(embedding_model)
    texts = df[message_col].tolist()
    metadatas = [{"User": user} for user in df[user_col].tolist()]
    ids = [str(i) for i in df.index.tolist()]

    db = None
    for i in tqdm(range(0, len(texts), batch_size), desc="Building FAISS Index"):
        batch_texts = texts[i : i + batch_size]
        batch_metadatas = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        if db is None:
            db = FAISS.from_texts(
                batch_texts,
                embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
        else:
            db.add_texts(
                batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    db.save_local(str(output_dir))



def dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))
