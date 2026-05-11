"""Сценарный эксперимент с несколькими симулированными учениками.

Цель скрипта — не имитировать реальных учеников идеально, а проверить, что
агентная механика ведет себя ожидаемо на разных профилях:
- слабый ученик чаще получает подсказки и похожие задачи через retrieval;
- растущий ученик постепенно переходит к более сложным темам;
- сильный ученик чаще продвигается по графу тем.

Скрипт сохраняет два CSV-файла и текстовый отчет, которые затем можно использовать
в курсовой как экспериментальные логи MVP.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from adaptive_agent import AdaptiveTutorAgent, StudentProfile
from retrieval_tool import build_retriever


LOG_PATH = Path("adaptive_experiment_log.csv")
SUMMARY_PATH = Path("adaptive_experiment_summary.csv")
REPORT_PATH = Path("simulate_adaptive_students_result.txt")


@dataclass(frozen=True)
class SimulatedStudentConfig:
    user_id: str
    initial_mastery: Dict[str, float]
    ability: Dict[str, float]
    steps: int = 12


STUDENT_CONFIGS: List[SimulatedStudentConfig] = [
    SimulatedStudentConfig(
        user_id="weak_student",
        initial_mastery={
            "credit_annuity": 0.20,
            "credit_diff": 0.10,
            "credit_mixed": 0.00,
            "deposit": 0.15,
            "optimization": 0.00,
        },
        ability={
            "credit_annuity": 0.46,
            "credit_diff": 0.34,
            "credit_mixed": 0.24,
            "deposit": 0.38,
            "optimization": 0.18,
        },
    ),
    SimulatedStudentConfig(
        user_id="growing_student",
        initial_mastery={
            "credit_annuity": 0.35,
            "credit_diff": 0.20,
            "credit_mixed": 0.10,
            "deposit": 0.25,
            "optimization": 0.05,
        },
        ability={
            "credit_annuity": 0.72,
            "credit_diff": 0.62,
            "credit_mixed": 0.58,
            "deposit": 0.66,
            "optimization": 0.50,
        },
    ),
    SimulatedStudentConfig(
        user_id="strong_student",
        initial_mastery={
            "credit_annuity": 0.60,
            "credit_diff": 0.45,
            "credit_mixed": 0.30,
            "deposit": 0.45,
            "optimization": 0.25,
        },
        ability={
            "credit_annuity": 0.92,
            "credit_diff": 0.86,
            "credit_mixed": 0.80,
            "deposit": 0.88,
            "optimization": 0.72,
        },
    ),
]


def clamp(value: float, left: float = 0.05, right: float = 0.95) -> float:
    return max(left, min(right, value))


def estimate_correct_probability(
    config: SimulatedStudentConfig,
    category_id: str,
    difficulty: int,
    current_mastery: float,
) -> float:
    """Оценивает вероятность правильного решения для симуляции.

    Формула намеренно простая и интерпретируемая:
    - базовая способность ученика зависит от темы;
    - более высокая сложность задачи снижает вероятность успеха;
    - текущий mastery частично повышает вероятность успеха.
    """

    base = config.ability.get(category_id, 0.40)
    difficulty_penalty = 0.08 * max(0, difficulty - 1)
    mastery_bonus = 0.20 * current_mastery
    return round(clamp(base - difficulty_penalty + mastery_bonus), 3)


def simulate_one_student(
    config: SimulatedStudentConfig,
    agent: AdaptiveTutorAgent,
    rng: random.Random,
) -> List[Dict[str, object]]:
    profile = StudentProfile(
        user_id=config.user_id,
        mastery={key: float(value) for key, value in config.initial_mastery.items()},
        history=[],
    )

    rows: List[Dict[str, object]] = []
    current_task = agent.select_start_task(profile)

    for step in range(1, config.steps + 1):
        mastery_before = profile.mastery.get(current_task.category_id, 0.0)
        probability = estimate_correct_probability(
            config=config,
            category_id=current_task.category_id,
            difficulty=current_task.difficulty,
            current_mastery=mastery_before,
        )
        is_correct = rng.random() < probability

        event = agent.register_attempt(profile, current_task.task_id, is_correct=is_correct)
        decision = agent.choose_next_step(profile, current_task.task_id, is_correct=is_correct)

        rows.append(
            {
                "user_id": profile.user_id,
                "step": step,
                "task_id": current_task.task_id,
                "category_id": current_task.category_id,
                "nuance": current_task.nuance,
                "difficulty": current_task.difficulty,
                "correct_probability": probability,
                "is_correct": int(is_correct),
                "mastery_before": event["mastery_before"],
                "mastery_after": event["mastery_after"],
                "action": decision["action"],
                "selection_method": decision["selection_method"],
                "next_category": decision["next_category"],
                "next_task_id": decision["next_task_id"],
                "retrieval_status": (decision.get("retrieval") or {}).get("status", ""),
            }
        )

        current_task = agent.tasks_by_id[decision["next_task_id"]]

    return rows


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    user_ids = sorted({str(row["user_id"]) for row in rows})

    for user_id in user_ids:
        user_rows = [row for row in rows if row["user_id"] == user_id]
        correct_count = sum(int(row["is_correct"]) for row in user_rows)
        retrieval_count = sum(1 for row in user_rows if row["selection_method"] == "retrieval")
        graph_count = sum(1 for row in user_rows if row["selection_method"] == "graph_and_mastery")
        categories = sorted({str(row["category_id"]) for row in user_rows})
        final_by_category: Dict[str, float] = {}
        for row in user_rows:
            final_by_category[str(row["category_id"])] = float(row["mastery_after"])

        result.append(
            {
                "user_id": user_id,
                "steps": len(user_rows),
                "correct_count": correct_count,
                "correct_rate": round(correct_count / max(1, len(user_rows)), 3),
                "retrieval_count": retrieval_count,
                "graph_count": graph_count,
                "visited_categories": ", ".join(categories),
                "final_mastery_by_visited_category": "; ".join(
                    f"{category}={final_by_category[category]:.3f}"
                    for category in sorted(final_by_category)
                ),
            }
        )

    return result


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(rows: List[Dict[str, object]], summary_rows: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("Эксперимент: адаптивный агент на симулированных учениках")
    lines.append("")
    lines.append(f"Всего учеников: {len(summary_rows)}")
    lines.append(f"Всего шагов: {len(rows)}")
    lines.append("")
    lines.append("Сводка по ученикам:")

    for row in summary_rows:
        lines.append(f"\n{row['user_id']}")
        lines.append(f"  steps: {row['steps']}")
        lines.append(f"  correct_rate: {row['correct_rate']}")
        lines.append(f"  retrieval_count: {row['retrieval_count']}")
        lines.append(f"  graph_count: {row['graph_count']}")
        lines.append(f"  visited_categories: {row['visited_categories']}")
        lines.append(f"  final_mastery: {row['final_mastery_by_visited_category']}")

    lines.append("")
    lines.append("Первые 10 шагов полного лога:")
    for row in rows[:10]:
        lines.append(
            "  "
            f"{row['user_id']} step={row['step']} "
            f"task={row['task_id']} category={row['category_id']} "
            f"correct={row['is_correct']} "
            f"mastery={row['mastery_before']}->{row['mastery_after']} "
            f"action={row['action']} method={row['selection_method']} "
            f"next={row['next_task_id']}"
        )

    lines.append("")
    lines.append(f"CSV-лог сохранен в {LOG_PATH}")
    lines.append(f"CSV-сводка сохранена в {SUMMARY_PATH}")
    return "\n".join(lines) + "\n"


def main() -> None:
    base_agent = AdaptiveTutorAgent.from_files(random_seed=17)
    retriever = build_retriever(base_agent.tasks, backend="lexical")
    agent = AdaptiveTutorAgent.from_files(random_seed=17, retriever=retriever)
    rng = random.Random(2026)

    all_rows: List[Dict[str, object]] = []
    for config in STUDENT_CONFIGS:
        all_rows.extend(simulate_one_student(config=config, agent=agent, rng=rng))

    summary_rows = summarize(all_rows)
    write_csv(LOG_PATH, all_rows)
    write_csv(SUMMARY_PATH, summary_rows)
    report = build_report(all_rows, summary_rows)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
