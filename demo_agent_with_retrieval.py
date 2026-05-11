"""Демо: адаптивный агент вызывает retrieval после ошибки ученика.

Сценарий имитирует одну важную механику MVP:
1. ученик получает задачу;
2. ученик ошибается;
3. агент обновляет профиль;
4. агент дает подсказку;
5. агент находит похожую задачу внутри той же темы через retrieval-инструмент.

Этот файл не требует LLM, ChromaDB и sentence-transformers: используется
локальный lexical retrieval backend. Vector backend можно подключить в
retrieval_tool.build_retriever(..., backend="vector"), если установлены зависимости.
"""

from __future__ import annotations

from adaptive_agent import AdaptiveTutorAgent, StudentProfile, flatten_tasks, load_json
from retrieval_tool import build_retriever


def print_task_header(prefix: str, task_id: str, category_id: str, text: str) -> None:
    print(f"{prefix}")
    print(f"  task_id: {task_id}")
    print(f"  category: {category_id}")
    print(f"  text: {text[:220].replace(chr(10), ' ')}...")


def main() -> None:
    tasks = flatten_tasks()
    graph = load_json("learning_graph.json")
    retriever = build_retriever(tasks, backend="lexical")
    agent = AdaptiveTutorAgent(tasks=tasks, graph=graph, random_seed=7, retriever=retriever)

    profile = StudentProfile(
        user_id="student_demo_retrieval",
        mastery={category_id: 0.35 for category_id in agent.tasks_by_category},
        history=[],
    )
    current_task = agent.tasks_by_id["annuity_rate_02"]
    print_task_header("Стартовая задача", current_task.task_id, current_task.category_id, current_task.text)

    event = agent.register_attempt(
        profile=profile,
        task_id=current_task.task_id,
        is_correct=False,
        attempts=1,
        hint_used=False,
    )
    next_step = agent.choose_next_step(profile, current_task.task_id, is_correct=False)

    print("\nРезультат попытки")
    print(f"  is_correct: {event['is_correct']}")
    print(f"  mastery: {event['mastery_before']} -> {event['mastery_after']}")

    print("\nРешение агента")
    print(f"  action: {next_step['action']}")
    print(f"  selection_method: {next_step['selection_method']}")
    print(f"  hint: {next_step['hint']}")
    print(f"  next_category: {next_step['next_category']}")
    print_task_header("  Подобранная следующая задача", next_step["next_task_id"], next_step["next_category"], next_step["next_task_text"])

    retrieval = next_step.get("retrieval") or {}
    print("\nRetrieval log")
    print(f"  status: {retrieval.get('status')}")
    print(f"  query_task_id: {retrieval.get('query_task_id')}")
    print(f"  category_filter: {retrieval.get('category_filter')}")
    for i, item in enumerate(retrieval.get("top_results", []), start=1):
        print(
            f"  top-{i}: {item['task_id']} | {item['category_id']} | "
            f"{item['nuance']} | difficulty={item['difficulty']} | "
            f"score={item['score']} | backend={item['backend']}"
        )


if __name__ == "__main__":
    main()
