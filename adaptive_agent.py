"""Адаптивный подбор задач для прототипа ассистента репетитора.

Модуль не заменяет RAG-поиск из rag_system.py. Он добавляет над ним учебную
логику: профиль ученика, граф переходов между темами и выбор следующей задачи
на основе результата предыдущей попытки.

В текущем MVP проверка ответа передается как флаг is_correct. Это осознанное
ограничение: в исходном датасете нет эталонных ответов и полных решений для всех
задач. Позже на это место можно подключить валидатор ответа: regex, structured
output от LLM или ручную проверку репетитором.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_DATASET_PATH = Path("ege_economics_dataset.json")
DEFAULT_GRAPH_PATH = Path("learning_graph.json")
DEFAULT_PROFILES_PATH = Path("student_profiles.json")


@dataclass(frozen=True)
class Task:
    task_id: str
    text: str
    category_id: str
    type_name: str
    nuance: str
    difficulty: int
    tags: List[str]
    prerequisites: List[str]


@dataclass
class StudentProfile:
    user_id: str
    mastery: Dict[str, float]
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def solved_task_ids(self) -> set[str]:
        return {event["task_id"] for event in self.history if event.get("is_correct") is True}

    @property
    def attempted_task_ids(self) -> set[str]:
        return {event["task_id"] for event in self.history}


CATEGORY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "credit_annuity": {
        "difficulty": 1,
        "tags": ["кредит", "проценты", "равные платежи", "аннуитет"],
        "prerequisites": ["проценты", "геометрическая прогрессия", "составление уравнения"],
    },
    "credit_diff": {
        "difficulty": 2,
        "tags": ["кредит", "проценты", "дифференцированные платежи", "остаток долга"],
        "prerequisites": ["проценты", "арифметическая прогрессия", "таблица платежей"],
    },
    "credit_mixed": {
        "difficulty": 3,
        "tags": ["кредит", "таблица", "смешанная схема", "несколько этапов"],
        "prerequisites": ["аннуитетные платежи", "дифференцированные платежи", "анализ таблиц"],
    },
    "deposit": {
        "difficulty": 2,
        "tags": ["вклад", "проценты", "пополнение", "снятие", "сложные проценты"],
        "prerequisites": ["проценты", "формула сложных процентов", "уравнения"],
    },
    "optimization": {
        "difficulty": 4,
        "tags": ["оптимизация", "максимум", "минимум", "производство", "прибыль"],
        "prerequisites": ["алгебраическая модель", "квадратичная функция", "оценка экстремума"],
    },
}


SUBTYPE_DIFFICULTY_SHIFT: Dict[str, int] = {
    "classic": 0,
    "classic_find_total": 0,
    "simple_compound": 0,
    "find_rate": 1,
    "unknown_term_n": 1,
    "with_actions": 1,
    "payment_ratio": 1,
    "two_plans": 2,
    "two_stage": 2,
    "table_given": 1,
    "hybrid_diff_annuity": 2,
    "irregular_schedule": 2,
    "optimization_max": 2,
    "production_allocation": 1,
    "profit_maximization": 2,
    "investment_strategy": 2,
}


HINTS: Dict[str, List[str]] = {
    "credit_annuity": [
        "Обозначь сумму кредита и выпиши долг после каждого начисления процентов.",
        "Для равных платежей удобно составлять рекуррентную формулу по годам.",
    ],
    "credit_diff": [
        "Сначала определи, на какую одинаковую величину уменьшается основной долг.",
        "Раздели платеж на две части: погашение долга и проценты на текущий остаток.",
    ],
    "credit_mixed": [
        "Перенеси условие в таблицу: период, долг до процентов, долг после процентов, платеж.",
        "Не смешивай разные правила начисления: для каждого этапа отдельно запиши формулу долга.",
    ],
    "deposit": [
        "Отслеживай сумму на счете после каждого начисления процентов и каждой операции.",
        "Если есть пополнения или снятия, важно соблюдать порядок действий в условии.",
    ],
    "optimization": [
        "Сначала введи переменную и запиши целевую функцию, которую нужно максимизировать или минимизировать.",
        "Проверь ограничения на переменные: в задачах ЕГЭ они часто задают допустимый диапазон.",
    ],
}


def clamp(value: float, left: float = 0.0, right: float = 1.0) -> float:
    return max(left, min(right, value))


def load_json(path: Path | str) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def flatten_tasks(dataset_path: Path = DEFAULT_DATASET_PATH) -> List[Task]:
    raw = load_json(dataset_path)
    tasks: List[Task] = []

    for category in raw["dataset"]:
        category_id = category["category_id"]
        type_name = category.get("type_name", category_id)
        defaults = CATEGORY_DEFAULTS.get(
            category_id,
            {"difficulty": 2, "tags": [category_id], "prerequisites": []},
        )

        blocks: Iterable[Dict[str, Any]]
        if "subtypes" in category:
            blocks = category["subtypes"]
        else:
            blocks = [{"nuance": "unknown", "examples": category.get("examples", [])}]

        for subtype in blocks:
            nuance = subtype.get("nuance", "unknown")
            base_difficulty = int(defaults["difficulty"])
            shift = SUBTYPE_DIFFICULTY_SHIFT.get(nuance, 0)
            difficulty = clamp(base_difficulty + shift, 1, 5)

            for example in subtype.get("examples", []):
                tasks.append(
                    Task(
                        task_id=example["id"],
                        text=example["text"],
                        category_id=category_id,
                        type_name=type_name,
                        nuance=nuance,
                        difficulty=int(difficulty),
                        tags=list(defaults["tags"]),
                        prerequisites=list(defaults["prerequisites"]),
                    )
                )

    return tasks


def load_profiles(path: Path = DEFAULT_PROFILES_PATH) -> Dict[str, StudentProfile]:
    raw_profiles = load_json(path)
    return {
        user_id: StudentProfile(
            user_id=profile["user_id"],
            mastery={k: float(v) for k, v in profile.get("mastery", {}).items()},
            history=list(profile.get("history", [])),
        )
        for user_id, profile in raw_profiles.items()
    }


def save_profiles(profiles: Dict[str, StudentProfile], path: Path = DEFAULT_PROFILES_PATH) -> None:
    dump_json(
        path,
        {
            user_id: {
                "user_id": profile.user_id,
                "mastery": profile.mastery,
                "history": profile.history,
            }
            for user_id, profile in profiles.items()
        },
    )


class AdaptiveTutorAgent:
    def __init__(
        self,
        tasks: Sequence[Task],
        graph: Dict[str, Any],
        random_seed: int = 42,
        retriever: Optional[Any] = None,
    ) -> None:
        self.tasks = list(tasks)
        self.graph = graph
        self.random = random.Random(random_seed)
        self.retriever = retriever
        self.tasks_by_id = {task.task_id: task for task in self.tasks}
        self.tasks_by_category: Dict[str, List[Task]] = {}

        for task in self.tasks:
            self.tasks_by_category.setdefault(task.category_id, []).append(task)

        for category_tasks in self.tasks_by_category.values():
            category_tasks.sort(key=lambda task: (task.difficulty, task.task_id))

    @classmethod
    def from_files(
        cls,
        dataset_path: Path = DEFAULT_DATASET_PATH,
        graph_path: Path = DEFAULT_GRAPH_PATH,
        random_seed: int = 42,
        retriever: Optional[Any] = None,
    ) -> "AdaptiveTutorAgent":
        return cls(
            tasks=flatten_tasks(dataset_path),
            graph=load_json(graph_path),
            random_seed=random_seed,
            retriever=retriever,
        )

    def ensure_profile(self, user_id: str, profiles: Dict[str, StudentProfile]) -> StudentProfile:
        if user_id not in profiles:
            profiles[user_id] = StudentProfile(
                user_id=user_id,
                mastery={category_id: 0.0 for category_id in self.tasks_by_category},
                history=[],
            )
        return profiles[user_id]

    def select_start_task(self, profile: StudentProfile) -> Task:
        category = self._weakest_category(profile)
        return self.select_task(profile, category)

    def register_attempt(
        self,
        profile: StudentProfile,
        task_id: str,
        is_correct: bool,
        attempts: int = 1,
        hint_used: bool = False,
    ) -> Dict[str, Any]:
        if task_id not in self.tasks_by_id:
            raise KeyError(f"Неизвестный task_id: {task_id}")

        task = self.tasks_by_id[task_id]
        previous_mastery = profile.mastery.get(task.category_id, 0.0)

        if is_correct:
            delta = 0.15
            if attempts > 1:
                delta -= 0.03 * (attempts - 1)
            if hint_used:
                delta -= 0.05
            delta = max(0.04, delta)
        else:
            delta = -0.10
            if attempts > 1:
                delta -= 0.02 * (attempts - 1)
            if hint_used:
                delta -= 0.03

        new_mastery = round(clamp(previous_mastery + delta), 3)
        profile.mastery[task.category_id] = new_mastery

        event = {
            "task_id": task.task_id,
            "category_id": task.category_id,
            "nuance": task.nuance,
            "difficulty": task.difficulty,
            "is_correct": bool(is_correct),
            "attempts": int(attempts),
            "hint_used": bool(hint_used),
            "mastery_before": round(previous_mastery, 3),
            "mastery_after": new_mastery,
        }
        profile.history.append(event)
        return event

    def choose_next_step(self, profile: StudentProfile, last_task_id: str, is_correct: bool) -> Dict[str, Any]:
        last_task = self.tasks_by_id[last_task_id]
        retrieval_info: Optional[Dict[str, Any]] = None
        selection_method = "graph_and_mastery"

        if not is_correct:
            retrieved_task, retrieval_info = self._select_similar_task_for_remediation(profile, last_task)
            if retrieved_task is not None:
                next_task = retrieved_task
                next_category = next_task.category_id
                action = "hint_and_retrieve_similar"
                selection_method = "retrieval"
            else:
                next_category = self._select_next_category(profile, last_task.category_id, is_correct)
                next_task = self.select_task(profile, next_category)
                action = "hint_and_repeat"
        else:
            next_category = self._select_next_category(profile, last_task.category_id, is_correct)
            next_task = self.select_task(profile, next_category)
            action = "advance" if next_category != last_task.category_id else "repeat"

        return {
            "action": action,
            "selection_method": selection_method,
            "last_category": last_task.category_id,
            "next_category": next_category,
            "next_task_id": next_task.task_id,
            "next_task_text": next_task.text,
            "hint": self.suggest_hint(last_task.category_id) if not is_correct else None,
            "retrieval": retrieval_info,
            "mastery": dict(profile.mastery),
        }

    def _select_similar_task_for_remediation(
        self,
        profile: StudentProfile,
        last_task: Task,
    ) -> tuple[Optional[Task], Optional[Dict[str, Any]]]:
        if self.retriever is None:
            return None, None
        exclude_ids = set(profile.attempted_task_ids)
        exclude_ids.add(last_task.task_id)

        try:
            results = self.retriever.find_similar(
                query_task=last_task,
                category_id=last_task.category_id,
                exclude_task_ids=exclude_ids,
                top_k=5,
            )
        except Exception as exc:
            return None, {
                "status": "failed",
                "error": str(exc),
                "query_task_id": last_task.task_id,
            }

        if not results:
            return None, {
                "status": "empty",
                "query_task_id": last_task.task_id,
                "category_filter": last_task.category_id,
            }
        best = results[0]
        task = getattr(best, "task", None) or self.tasks_by_id[getattr(best, "task_id")]
        top = []
        for item in results[:3]:
            item_task = getattr(item, "task", None)
            if item_task is None:
                item_task = self.tasks_by_id[getattr(item, "task_id")]
            top.append(
                {
                    "task_id": item_task.task_id,
                    "category_id": item_task.category_id,
                    "nuance": item_task.nuance,
                    "difficulty": item_task.difficulty,
                    "score": getattr(item, "score", None),
                    "backend": getattr(item, "backend", "unknown"),
                    "reason": getattr(item, "reason", ""),
                }
            )

        return task, {
            "status": "ok",
            "query_task_id": last_task.task_id,
            "category_filter": last_task.category_id,
            "chosen_task_id": task.task_id,
            "top_results": top,
        }

    def select_task(self, profile: StudentProfile, category_id: str) -> Task:
        if category_id not in self.tasks_by_category:
            category_id = self._weakest_category(profile)

        solved = profile.solved_task_ids
        category_tasks = self.tasks_by_category[category_id]
        mastery = profile.mastery.get(category_id, 0.0)
        max_difficulty = self._allowed_difficulty(mastery)

        candidates = [
            task
            for task in category_tasks
            if task.task_id not in solved and task.difficulty <= max_difficulty
        ]
        if not candidates:
            candidates = [task for task in category_tasks if task.task_id not in solved]
        if not candidates:
            candidates = category_tasks

        candidates.sort(key=lambda task: (abs(task.difficulty - max_difficulty), task.task_id))
        top_pool = candidates[: min(3, len(candidates))]
        return self.random.choice(top_pool)

    def suggest_hint(self, category_id: str) -> str:
        hints = HINTS.get(category_id, ["Разбей условие на известные величины, неизвестные величины и целевое уравнение."])
        return self.random.choice(hints)

    def _weakest_category(self, profile: StudentProfile) -> str:
        categories = list(self.tasks_by_category)
        not_mastered = [
            category
            for category in categories
            if profile.mastery.get(category, 0.0) < 0.75
        ]
        pool = not_mastered or categories
        return min(
            pool,
            key=lambda category: (
                self.graph[category].get("level", 99),
                profile.mastery.get(category, 0.0),
            ),
        )

    def _select_next_category(self, profile: StudentProfile, current_category: str, is_correct: bool) -> str:
        branch = "if_correct" if is_correct else "if_wrong"
        candidates = self.graph.get(current_category, {}).get(branch, [current_category])
        candidates = [category for category in candidates if category in self.tasks_by_category]
        if not candidates:
            return current_category
        if is_correct:
            return min(candidates, key=lambda category: (profile.mastery.get(category, 0.0), self.graph[category].get("level", 99)))
        return min(candidates, key=lambda category: (self.graph[category].get("level", 99), profile.mastery.get(category, 0.0)))

    @staticmethod
    def _allowed_difficulty(mastery: float) -> int:
        if mastery < 0.25:
            return 1
        if mastery < 0.50:
            return 2
        if mastery < 0.75:
            return 3
        return 5
