"""Аудит датасета задач для раздела курсовой о данных.

Скрипт считает категории, подтипы, длины условий и проверяет уникальность id.
Он нужен не для модели, а для воспроизводимого описания текущего состояния базы
задач.
"""

from collections import Counter

from adaptive_agent import flatten_tasks


def main() -> None:
    tasks = flatten_tasks()
    category_counts = Counter(task.category_id for task in tasks)
    subtype_counts = Counter((task.category_id, task.nuance) for task in tasks)
    ids = [task.task_id for task in tasks]
    lengths = [len(task.text.split()) for task in tasks]

    print("Аудит датасета")
    print(f"Всего задач: {len(tasks)}")
    print(f"Уникальных id: {len(set(ids))}")
    print(f"Дубликатов id: {len(ids) - len(set(ids))}")
    print(f"Минимальная длина условия, слов: {min(lengths)}")
    print(f"Средняя длина условия, слов: {sum(lengths) / len(lengths):.1f}")
    print(f"Максимальная длина условия, слов: {max(lengths)}")

    print("\nРаспределение по категориям:")
    for category, count in category_counts.items():
        print(f"  {category:<15} {count}")

    print("\nРаспределение по подтипам:")
    for (category, subtype), count in sorted(subtype_counts.items()):
        print(f"  {category:<15} {subtype:<24} {count}")

    print("\nПример плоской записи задачи:")
    task = tasks[0]
    print(f"  id: {task.task_id}")
    print(f"  category_id: {task.category_id}")
    print(f"  type_name: {task.type_name}")
    print(f"  nuance: {task.nuance}")
    print(f"  difficulty: {task.difficulty}")
    print(f"  tags: {', '.join(task.tags)}")
    print(f"  prerequisites: {', '.join(task.prerequisites)}")


if __name__ == "__main__":
    main()
