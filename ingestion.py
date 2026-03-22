import io
import re
import shutil
import uuid
from pathlib import Path
from urllib.parse import urlparse

import chromadb
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from pypdf import PdfReader


class IngestionError(Exception):
    pass


class KnowledgeIngestor:
    def __init__(
        self,
        pdfs,
        urls,
        api_key,
        persist_directory="data/chroma",
        collection_name="knowledge_base",
    ):
        self.pdfs = pdfs or []
        self.urls = self._normalize_urls(urls or [])
        self.client = OpenAI(api_key=api_key)
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.warnings = []

        self.documents = self._load_documents()
        if not self.documents:
            raise IngestionError("No readable content was found in the provided PDFs or website URLs.")

        self.chunks = self._chunk_documents(self.documents)
        if not self.chunks:
            raise IngestionError("The loaded sources did not contain enough text to index.")

        self.collection = self._build_collection()

    def _load_documents(self):
        documents = []
        documents.extend(self._load_pdf_documents())
        documents.extend(self._load_web_documents())
        return documents

    def _load_pdf_documents(self):
        documents = []
        for pdf_file in self.pdfs:
            try:
                reader = PdfReader(io.BytesIO(pdf_file.getvalue()))
            except Exception as exc:
                self.warnings.append(f"Skipped PDF '{pdf_file.name}': {exc}")
                continue

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                cleaned_text = self._normalize_text(text)
                if not cleaned_text:
                    continue
                documents.append(
                    {
                        "content": cleaned_text,
                        "source_name": pdf_file.name,
                        "source_type": "pdf",
                        "location": f"Page {page_number}",
                    }
                )

            if not any(doc["source_name"] == pdf_file.name for doc in documents):
                self.warnings.append(f"PDF '{pdf_file.name}' did not contain extractable text.")

        return documents

    def _load_web_documents(self):
        if not self.urls:
            return []

        documents = []
        session = requests.Session()
        session.headers.update(
            {"User-Agent": "knowledge-assistant/1.0 (+https://platform.openai.com/)"}
        )

        for url in self.urls:
            try:
                response = session.get(url, timeout=20)
                response.raise_for_status()
            except requests.RequestException as exc:
                self.warnings.append(f"Skipped website '{url}': {exc}")
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            parts = [
                element.get_text(" ", strip=True)
                for element in soup.find_all(["title", "h1", "h2", "h3", "p", "li"])
            ]
            cleaned_text = self._normalize_text("\n".join(parts))
            if not cleaned_text:
                self.warnings.append(f"Website '{url}' did not contain readable text.")
                continue

            documents.append(
                {
                    "content": cleaned_text,
                    "source_name": url,
                    "source_type": "website",
                    "location": "Web page",
                }
            )

        return documents

    def _build_collection(self):
        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        chroma_client = chromadb.PersistentClient(path=str(self.persist_directory))
        collection = chroma_client.get_or_create_collection(name=self.collection_name)

        texts = [chunk["content"] for chunk in self.chunks]
        embeddings = self._embed_texts(texts)
        collection.add(
            ids=[chunk["id"] for chunk in self.chunks],
            documents=texts,
            metadatas=[chunk["metadata"] for chunk in self.chunks],
            embeddings=embeddings,
        )
        return collection

    def retrieve(self, query, limit=4):
        query_embedding = self._embed_texts([query])[0]
        result = self.collection.query(query_embeddings=[query_embedding], n_results=limit)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        return [
            {"content": document, "metadata": metadata}
            for document, metadata in zip(documents, metadatas)
        ]

    def get_source_overview(self):
        source_map = {}
        for document in self.documents:
            source_name = document["source_name"]
            source_map[source_name] = source_map.get(source_name, 0) + 1

        return [
            {"source": source, "pages_or_sections": count}
            for source, count in sorted(source_map.items())
        ]

    def get_summary_documents(self, limit=12):
        return self.chunks[:limit]

    def get_stats(self):
        source_types = {}
        for document in self.documents:
            source_type = document["source_type"]
            source_types[source_type] = source_types.get(source_type, 0) + 1

        return {
            "sources": len({document["source_name"] for document in self.documents}),
            "sections": len(self.documents),
            "chunks": len(self.chunks),
            "pdf_sections": source_types.get("pdf", 0),
            "website_sections": source_types.get("website", 0),
        }

    def _chunk_documents(self, documents, chunk_size=1400, chunk_overlap=220):
        chunks = []
        step = max(chunk_size - chunk_overlap, 1)

        for document in documents:
            text = document["content"]
            for chunk_index, start_index in enumerate(range(0, len(text), step), start=1):
                chunk_text = text[start_index : start_index + chunk_size].strip()
                if not chunk_text:
                    continue
                chunks.append(
                    {
                        "id": uuid.uuid4().hex,
                        "content": chunk_text,
                        "metadata": {
                            "source_name": document["source_name"],
                            "source_type": document["source_type"],
                            "location": document["location"],
                            "chunk_index": chunk_index,
                        },
                    }
                )
        return chunks

    def _embed_texts(self, texts, batch_size=64):
        embeddings = []
        for start_index in range(0, len(texts), batch_size):
            batch = texts[start_index : start_index + batch_size]
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    @staticmethod
    def _normalize_urls(urls):
        normalized = []
        for raw_url in urls:
            url = raw_url.strip()
            if not url:
                continue
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            normalized.append(url)

        seen = []
        for url in normalized:
            if url not in seen:
                seen.append(url)
        return seen

    @staticmethod
    def _normalize_text(text):
        return re.sub(r"\s+", " ", text).strip()
