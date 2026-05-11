"""Демонстрация адаптивной сессии ученика.

Запуск:
    python demo_adaptive_session.py

Сценарий фиксированный, чтобы его можно было вставить в курсовую как
воспроизводимый эксперимент. В реальном приложении вместо outcome_sequence
будут ответы ученика и отдельный валидатор решения.
"""

from adaptive_agent import AdaptiveTutorAgent, StudentProfile


def short(text: str, limit: int = 130) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def print_step(step_no: int, task_id: str, category: str, result: bool, event: dict, decision: dict) -> None:
    status = "верно" if result else "ошибка"
    print(f"\nШаг {step_no}")
    print(f"  Задача: {task_id} [{category}]")
    print(f"  Результат ученика: {status}")
    print(
        "  Mastery по теме: "
        f"{event['mastery_before']:.3f} -> {event['mastery_after']:.3f}"
    )
    print(f"  Действие агента: {decision['action']}")
    if decision["hint"]:
        print(f"  Подсказка: {decision['hint']}")
    print(f"  Следующая тема: {decision['next_category']}")
    print(f"  Следующая задача: {decision['next_task_id']}")
    print(f"  Условие: {short(decision['next_task_text'])}")


def main() -> None:
    agent = AdaptiveTutorAgent.from_files(random_seed=7)

    profile = StudentProfile(
        user_id="student_demo",
        mastery={
            "credit_annuity": 0.45,
            "credit_diff": 0.25,
            "credit_mixed": 0.10,
            "deposit": 0.30,
            "optimization": 0.05,
        },
        history=[],
    )

    print("Адаптивная сессия ученика student_demo")
    print("Начальный профиль mastery:")
    for category, score in profile.mastery.items():
        print(f"  {category:<15} {score:.3f}")

    current_task = agent.select_start_task(profile)
    print("\nСтартовая задача:")
    print(f"  {current_task.task_id} [{current_task.category_id}] — {short(current_task.text)}")
    outcome_sequence = [False, True, True, False, True]

    for step_no, result in enumerate(outcome_sequence, start=1):
        event = agent.register_attempt(
            profile=profile,
            task_id=current_task.task_id,
            is_correct=result,
            attempts=1 if result else 2,
            hint_used=not result,
        )
        decision = agent.choose_next_step(profile, current_task.task_id, result)
        print_step(step_no, current_task.task_id, current_task.category_id, result, event, decision)
        current_task = agent.tasks_by_id[decision["next_task_id"]]

    print("\nФинальный профиль mastery:")
    for category, score in profile.mastery.items():
        print(f"  {category:<15} {score:.3f}")

    print("\nИстория попыток:")
    for item in profile.history:
        print(
            f"  {item['task_id']:<24} {item['category_id']:<15} "
            f"correct={str(item['is_correct']):<5} "
            f"mastery={item['mastery_before']:.3f}->{item['mastery_after']:.3f}"
        )


if __name__ == "__main__":
    main()
