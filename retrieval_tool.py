"""Инструмент подбора похожих задач для адаптивного агента.

В проекте уже есть отдельный интерактивный RAG-прототип в rag_system.py.
Этот модуль делает retrieval переиспользуемым инструментом, который можно вызвать
из учебного агента: например, когда ученик ошибся и ему нужна близкая по механике
задача для закрепления.

Для воспроизводимости в MVP есть два режима:
1. lexical — чистый Python TF-IDF/BM25-like baseline без внешних зависимостей;
2. vector — E5 + ChromaDB, совместимый с исходной RAG-идеей. Он включается только
   если в окружении установлены sentence-transformers и chromadb.

В демо и тестах используется lexical-режим, чтобы эксперименты запускались без
тяжелых библиотек и скачивания модели. В тексте курсовой это можно описать как
локальный retrieval baseline и optional vector backend.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from adaptive_agent import Task


TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


STOPWORDS = {
    "и", "в", "во", "на", "с", "со", "к", "ко", "по", "за", "из", "от", "до", "для",
    "что", "как", "какую", "какова", "какой", "каким", "какие", "если", "чтобы", "при",
    "после", "перед", "каждый", "каждого", "следующего", "следующий", "года", "год", "месяца",
    "месяцев", "день", "числа", "рублей", "руб", "млн", "тыс", "равна", "равен", "равно",
    "сумма", "сумму", "банк", "банке", "планируется", "должен", "должна", "нужно", "следует",
    "the", "a", "an", "of", "to", "in", "and", "or",
}


@dataclass(frozen=True)
class RetrievalResult:
    """Результат поиска похожей задачи."""

    task: Task
    score: float
    backend: str
    reason: str


def tokenize(text: str) -> List[str]:
    tokens = [token.lower().replace("ё", "е") for token in TOKEN_RE.findall(text)]
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def task_to_search_text(task: Task) -> str:
    """Собирает текст задачи и метаданные в одну строку для retrieval."""

    parts = [
        task.text,
        task.category_id,
        task.type_name,
        task.nuance,
        " ".join(task.tags),
        " ".join(task.prerequisites),
    ]
    return " ".join(str(part) for part in parts if part)


class LexicalTaskRetriever:
    """Детерминированный retrieval baseline на TF-IDF cosine similarity.

    Он полезен для курсовой по двум причинам:
    - запускается без внешних моделей;
    - дает контрольную точку, с которой можно сравнивать более тяжелый vector RAG.
    """

    backend_name = "lexical_tfidf"

    def __init__(self, tasks: Sequence[Task]) -> None:
        self.tasks = list(tasks)
        self.task_by_id = {task.task_id: task for task in self.tasks}
        self._doc_tokens: Dict[str, Counter[str]] = {}
        self._doc_vectors: Dict[str, Dict[str, float]] = {}
        self._doc_norms: Dict[str, float] = {}
        self._idf: Dict[str, float] = {}
        self._build_index()

    def _build_index(self) -> None:
        document_frequency: Dict[str, int] = defaultdict(int)

        for task in self.tasks:
            counts = Counter(tokenize(task_to_search_text(task)))
            self._doc_tokens[task.task_id] = counts
            for token in counts:
                document_frequency[token] += 1

        n_docs = max(1, len(self.tasks))
        self._idf = {
            token: math.log((1 + n_docs) / (1 + df)) + 1.0
            for token, df in document_frequency.items()
        }

        for task_id, counts in self._doc_tokens.items():
            vector = self._tfidf_vector(counts)
            self._doc_vectors[task_id] = vector
            self._doc_norms[task_id] = math.sqrt(sum(value * value for value in vector.values())) or 1.0

    def _tfidf_vector(self, counts: Counter[str]) -> Dict[str, float]:
        if not counts:
            return {}
        max_tf = max(counts.values())
        return {
            token: (0.5 + 0.5 * count / max_tf) * self._idf.get(token, 1.0)
            for token, count in counts.items()
        }

    @staticmethod
    def _cosine(query_vector: Dict[str, float], doc_vector: Dict[str, float], query_norm: float, doc_norm: float) -> float:
        if not query_vector or not doc_vector:
            return 0.0
        if len(query_vector) > len(doc_vector):
            query_vector, doc_vector = doc_vector, query_vector
        dot = sum(value * doc_vector.get(token, 0.0) for token, value in query_vector.items())
        return dot / (query_norm * doc_norm)

    def find_similar(
        self,
        query_task: Task,
        category_id: Optional[str] = None,
        exclude_task_ids: Optional[Iterable[str]] = None,
        top_k: int = 3,
    ) -> List[RetrievalResult]:
        exclude: Set[str] = set(exclude_task_ids or [])
        query_counts = Counter(tokenize(task_to_search_text(query_task)))
        query_vector = self._tfidf_vector(query_counts)
        query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0

        results: List[RetrievalResult] = []
        for task in self.tasks:
            if task.task_id in exclude:
                continue
            if category_id is not None and task.category_id != category_id:
                continue

            score = self._cosine(
                query_vector=query_vector,
                doc_vector=self._doc_vectors[task.task_id],
                query_norm=query_norm,
                doc_norm=self._doc_norms[task.task_id],
            )
            if score <= 0:
                continue

            reason = (
                "найдена близкая задача по условию и метаданным"
                if category_id is None
                else f"найдена близкая задача внутри темы {category_id}"
            )
            results.append(RetrievalResult(task=task, score=round(score, 4), backend=self.backend_name, reason=reason))

        results.sort(key=lambda item: (-item.score, item.task.difficulty, item.task.task_id))
        return results[:top_k]


class E5ChromaTaskRetriever:
    """Optional vector retrieval backend: multilingual-e5 + ChromaDB.

    Этот класс намеренно импортирует тяжелые зависимости внутри __init__, чтобы
    весь проект можно было запускать без них, если нужен только агентный MVP.
    """

    backend_name = "e5_chroma"

    def __init__(
        self,
        tasks: Sequence[Task],
        model_name: str = "intfloat/multilingual-e5-large",
        collection_name: str = "ege_economics_agent_retrieval",
    ) -> None:
        try:
            import chromadb  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Для vector retrieval нужны зависимости chromadb и sentence-transformers. "
                "Установи requirements.txt или используй backend='lexical'."
            ) from exc

        self.tasks = list(tasks)
        self.task_by_id = {task.task_id: task for task in self.tasks}
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.Client()

        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(name=collection_name)
        documents = [f"passage: {task_to_search_text(task)}" for task in self.tasks]
        embeddings = self.model.encode(documents)
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=[
                {
                    "task_id": task.task_id,
                    "category_id": task.category_id,
                    "nuance": task.nuance,
                    "difficulty": task.difficulty,
                }
                for task in self.tasks
            ],
            ids=[task.task_id for task in self.tasks],
        )

    def find_similar(
        self,
        query_task: Task,
        category_id: Optional[str] = None,
        exclude_task_ids: Optional[Iterable[str]] = None,
        top_k: int = 3,
    ) -> List[RetrievalResult]:
        exclude: Set[str] = set(exclude_task_ids or [])
        query = f"query: {task_to_search_text(query_task)}"
        query_vec = [self.model.encode(query).tolist()]
        raw = self.collection.query(query_embeddings=query_vec, n_results=min(len(self.tasks), max(top_k * 5, top_k)))

        results: List[RetrievalResult] = []
        ids = raw.get("ids", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]

        for task_id, distance, meta in zip(ids, distances, metadatas):
            if task_id in exclude:
                continue
            task = self.task_by_id[task_id]
            if category_id is not None and task.category_id != category_id:
                continue
            score = 1.0 / (1.0 + float(distance))
            reason = (
                "vector retrieval по E5-эмбеддингам"
                if category_id is None
                else f"vector retrieval внутри темы {category_id}"
            )
            results.append(RetrievalResult(task=task, score=round(score, 4), backend=self.backend_name, reason=reason))
            if len(results) >= top_k:
                break

        return results


def build_retriever(tasks: Sequence[Task], backend: str = "lexical") -> Any:
    """Фабрика retrieval-инструмента.

    backend:
    - lexical: всегда доступный TF-IDF baseline;
    - vector: E5 + ChromaDB, если установлены зависимости;
    - auto: пробует vector, при ошибке откатывается к lexical.
    """

    if backend == "lexical":
        return LexicalTaskRetriever(tasks)
    if backend == "vector":
        return E5ChromaTaskRetriever(tasks)
    if backend == "auto":
        try:
            return E5ChromaTaskRetriever(tasks)
        except Exception:
            return LexicalTaskRetriever(tasks)
    raise ValueError("backend должен быть одним из: lexical, vector, auto")
