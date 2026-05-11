"""Мини-проверки связи адаптивного агента с retrieval-инструментом."""

from __future__ import annotations

from adaptive_agent import AdaptiveTutorAgent, StudentProfile, flatten_tasks, load_json
from retrieval_tool import build_retriever


def make_agent() -> AdaptiveTutorAgent:
    tasks = flatten_tasks()
    graph = load_json("learning_graph.json")
    retriever = build_retriever(tasks, backend="lexical")
    return AdaptiveTutorAgent(tasks=tasks, graph=graph, random_seed=11, retriever=retriever)


def test_retrieval_is_used_after_wrong_answer() -> None:
    agent = make_agent()
    profile = StudentProfile(
        user_id="student_test_retrieval",
        mastery={category_id: 0.4 for category_id in agent.tasks_by_category},
        history=[],
    )

    task_id = "annuity_rate_02"
    task = agent.tasks_by_id[task_id]
    event = agent.register_attempt(profile, task_id=task_id, is_correct=False)
    step = agent.choose_next_step(profile, task_id, is_correct=False)

    assert event["mastery_after"] < event["mastery_before"]
    assert step["selection_method"] == "retrieval"
    assert step["action"] == "hint_and_retrieve_similar"
    assert step["retrieval"]["status"] == "ok"
    assert step["next_task_id"] != task_id
    assert step["next_category"] == task.category_id
    assert step["retrieval"]["top_results"]


def test_graph_is_used_after_correct_answer() -> None:
    agent = make_agent()
    profile = StudentProfile(
        user_id="student_test_graph",
        mastery={category_id: 0.2 for category_id in agent.tasks_by_category},
        history=[],
    )

    task_id = "annuity_classic_01"
    event = agent.register_attempt(profile, task_id=task_id, is_correct=True)
    step = agent.choose_next_step(profile, task_id, is_correct=True)

    assert event["mastery_after"] > event["mastery_before"]
    assert step["selection_method"] == "graph_and_mastery"
    assert step["retrieval"] is None
    assert step["next_category"] in agent.graph["credit_annuity"]["if_correct"]


def test_retriever_respects_category_filter() -> None:
    agent = make_agent()
    query_task = agent.tasks_by_id["diff_total_01"]
    results = agent.retriever.find_similar(
        query_task=query_task,
        category_id="credit_diff",
        exclude_task_ids={query_task.task_id},
        top_k=5,
    )

    assert results
    assert all(item.task.category_id == "credit_diff" for item in results)
    assert all(item.task.task_id != query_task.task_id for item in results)


def main() -> None:
    test_retrieval_is_used_after_wrong_answer()
    test_graph_is_used_after_correct_answer()
    test_retriever_respects_category_filter()
    print("Связка агент + retrieval: OK")


if __name__ == "__main__":
    main()
