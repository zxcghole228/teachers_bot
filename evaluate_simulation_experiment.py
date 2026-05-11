"""Минимальные проверки результатов сценарного эксперимента."""

from __future__ import annotations

import csv
from pathlib import Path

from adaptive_agent import AdaptiveTutorAgent

LOG_PATH = Path("adaptive_experiment_log.csv")
SUMMARY_PATH = Path("adaptive_experiment_summary.csv")
EXPECTED_STEPS = 36
EXPECTED_USERS = {"weak_student", "growing_student", "strong_student"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise AssertionError(f"Не найден файл: {path}. Сначала запусти simulate_adaptive_students.py")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = read_csv(LOG_PATH)
    summary = read_csv(SUMMARY_PATH)
    agent = AdaptiveTutorAgent.from_files(random_seed=17)
    known_categories = set(agent.tasks_by_category)
    known_task_ids = set(agent.tasks_by_id)

    assert len(rows) == EXPECTED_STEPS, f"Ожидалось {EXPECTED_STEPS} строк лога, получено {len(rows)}"
    assert {row["user_id"] for row in summary} == EXPECTED_USERS, "Неверный набор симулированных учеников"
    assert {row["user_id"] for row in rows} == EXPECTED_USERS, "В логе есть не все ученики"

    selection_methods = {row["selection_method"] for row in rows}
    assert "retrieval" in selection_methods, "В эксперименте ни разу не был вызван retrieval"
    assert "graph_and_mastery" in selection_methods, "В эксперименте ни разу не был использован граф"

    actions = {row["action"] for row in rows}
    assert any(action.startswith("hint") for action in actions), "Нет шагов с подсказкой"
    assert {"advance", "repeat"}.intersection(actions), "Нет шагов продвижения или повторения по графу"

    for row in rows:
        assert row["category_id"] in known_categories, f"Неизвестная категория: {row['category_id']}"
        assert row["next_category"] in known_categories, f"Неизвестная следующая категория: {row['next_category']}"
        assert row["task_id"] in known_task_ids, f"Неизвестная задача: {row['task_id']}"
        assert row["next_task_id"] in known_task_ids, f"Неизвестная следующая задача: {row['next_task_id']}"

        before = float(row["mastery_before"])
        after = float(row["mastery_after"])
        assert 0.0 <= before <= 1.0, f"mastery_before вне диапазона: {before}"
        assert 0.0 <= after <= 1.0, f"mastery_after вне диапазона: {after}"

        is_correct = int(row["is_correct"])
        if is_correct:
            assert after >= before, "После правильного ответа mastery не должен уменьшаться"
            assert row["selection_method"] == "graph_and_mastery", "После правильного ответа должен использоваться граф"
        else:
            assert after <= before, "После ошибки mastery не должен увеличиваться"
            assert row["action"].startswith("hint"), "После ошибки ожидается подсказка"

    print("Сценарный эксперимент: OK")
    print(f"Проверено строк лога: {len(rows)}")
    print(f"Ученики: {', '.join(sorted(EXPECTED_USERS))}")


if __name__ == "__main__":
    main()
