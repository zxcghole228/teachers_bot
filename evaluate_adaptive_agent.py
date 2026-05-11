"""Мини-оценка агентной логики без LLM и без ChromaDB.

Цель этого файла — показать, что новая часть проекта не является только
описанием в README. Здесь проверяются базовые свойства адаптивного сценария:
1. mastery растет после правильного ответа и падает после ошибки;
2. следующая тема выбирается из learning_graph.json;
3. агент не повторяет уже решенные задачи, пока в теме есть альтернативы;
4. демонстрационные сценарии завершаются без ошибок.
"""

from adaptive_agent import AdaptiveTutorAgent, StudentProfile


def make_profile() -> StudentProfile:
    return StudentProfile(
        user_id="test_student",
        mastery={
            "credit_annuity": 0.40,
            "credit_diff": 0.20,
            "credit_mixed": 0.05,
            "deposit": 0.30,
            "optimization": 0.00,
        },
        history=[],
    )


def check_mastery_update(agent: AdaptiveTutorAgent) -> None:
    profile = make_profile()
    task = agent.select_task(profile, "credit_annuity")
    before = profile.mastery[task.category_id]
    agent.register_attempt(profile, task.task_id, is_correct=True)
    after_correct = profile.mastery[task.category_id]
    assert after_correct > before, "mastery должен расти после правильного ответа"

    agent.register_attempt(profile, task.task_id, is_correct=False, attempts=2, hint_used=True)
    after_wrong = profile.mastery[task.category_id]
    assert after_wrong < after_correct, "mastery должен снижаться после ошибки"


def check_graph_transition(agent: AdaptiveTutorAgent) -> None:
    profile = make_profile()
    task = agent.select_task(profile, "credit_annuity")
    agent.register_attempt(profile, task.task_id, is_correct=True)
    decision = agent.choose_next_step(profile, task.task_id, is_correct=True)
    allowed = agent.graph[task.category_id]["if_correct"]
    assert decision["next_category"] in allowed, "следующая тема должна быть взята из графа переходов"


def check_no_repeat_when_possible(agent: AdaptiveTutorAgent) -> None:
    profile = make_profile()
    first = agent.select_task(profile, "credit_annuity")
    agent.register_attempt(profile, first.task_id, is_correct=True)
    second = agent.select_task(profile, "credit_annuity")
    assert first.task_id != second.task_id, "агент не должен повторять уже решенную задачу, если есть альтернативы"


def run_scenario(agent: AdaptiveTutorAgent, outcomes: list[bool]) -> dict:
    profile = make_profile()
    task = agent.select_start_task(profile)
    actions = []

    for outcome in outcomes:
        agent.register_attempt(profile, task.task_id, is_correct=outcome, hint_used=not outcome)
        decision = agent.choose_next_step(profile, task.task_id, outcome)
        actions.append(decision["action"])
        task = agent.tasks_by_id[decision["next_task_id"]]

    return {
        "steps": len(outcomes),
        "actions": actions,
        "final_mastery": profile.mastery,
        "history_len": len(profile.history),
    }


def main() -> None:
    agent = AdaptiveTutorAgent.from_files(random_seed=42)

    check_mastery_update(agent)
    check_graph_transition(agent)
    check_no_repeat_when_possible(agent)

    scenarios = {
        "weak_student": [False, False, True, False],
        "growing_student": [False, True, True, True],
        "strong_student": [True, True, True, True],
    }

    print("Агентные проверки: OK")
    print("\nСценарные прогоны:")
    for name, outcomes in scenarios.items():
        result = run_scenario(agent, outcomes)
        print(f"\n{name}")
        print(f"  steps: {result['steps']}")
        print(f"  actions: {', '.join(result['actions'])}")
        print("  final_mastery:")
        for category, score in result["final_mastery"].items():
            print(f"    {category:<15} {score:.3f}")


if __name__ == "__main__":
    main()
