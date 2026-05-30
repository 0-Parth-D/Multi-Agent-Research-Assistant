import arxiv
from langchain_core .documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import os
from langchain_chroma import Chroma
import hashlib
import time
import argparse
import dotenv
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")

def load_documents(query: str, max_docs: int) -> list:
  # Construct the default API client.
  client = arxiv.Client(
    delay_seconds=3,
    num_retries=5
  )

  # Search for the 10 most recent articles matching the keyword "quantum."
  search = arxiv.Search(
    query = query,
    max_results = max_docs,
    # sort_by = arxiv.SortCriterion.SubmittedDate
  )

  results = client.results(search)

  docs = []
  for res in results:
    doc = Document(
        page_content = res.summary,
        metadata={
            "title": res.title,
            "source": res.entry_id,
            "published": str(res.published) if res.published else None
        }
    )

    docs.append(doc)

  return docs

def chunk_documents(documents):
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=250,
      chunk_overlap=25
  )

  texts = splitter.split_documents(documents)
  return texts

def get_vectorstore(embeddings):
  vs = Chroma(
      collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
      embedding_function=embeddings,
      persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY")
  )

  return vs


import hashlib

def add_documents(chunks, embeddings):
    vs = get_vectorstore(embeddings)

    # 1. Deterministic IDs from content
    ids = [
        hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        for doc in chunks
    ]

    unique_ids = list(dict.fromkeys(ids))

    # 2. Find which of these IDs already exist in Chroma
    existing = vs._collection.get(ids=unique_ids, include=[])
    existing_ids = set(existing.get("ids", []))

    # 3. Filter to only new docs/ids
    new_docs = []
    new_ids = []
    for doc, _id in zip(chunks, unique_ids):
        if _id not in existing_ids:
            new_docs.append(doc)
            new_ids.append(_id)

    if not new_docs:
        print("No new documents to add; all chunks already exist.")
        return vs

    # 4. Add only new docs
    vs.add_documents(documents=new_docs, ids=new_ids)
    print(f"Added {len(new_docs)} new documents, skipped {len(chunks) - len(new_docs)} duplicates.")

    return vs

def retrieval(vectorstore, query, k):
  start = time.time()
  query_res = vectorstore.similarity_search_with_score(
      query = query,
      k = k
  )
  end = time.time()
  print("Execution time for query - " + query + ":", end - start, "seconds")

  return query_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Topic to ingest from arXiv (e.g. 'reinforcement learning'). If omitted, no new docs are added."
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=100,
        help="Maximum number of new arXiv docs to fetch for the topic."
    )
    args = parser.parse_args()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 1. Load or build base vector store
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        vs = get_vectorstore(embeddings)
        print("Loaded existing store.")
    else:
        # Initial build with a default topic
        print("No existing store found. Building a new one...")
        docs = load_documents("large language models", max_docs=args.max_docs)
        chunks = chunk_documents(docs)
        vs = add_documents(chunks, embeddings)
        print("Built new store.")

    # 2. Optionally ingest additional papers for a new topic
    if args.topic:
        print(f"Ingesting additional papers for topic: {args.topic!r}")
        docs = load_documents(args.topic, max_docs=args.max_docs)
        chunks = chunk_documents(docs)
        vs = add_documents(chunks, embeddings)  # uses your dedup logic (MD5 IDs)
        print("Finished ingesting new topic.")