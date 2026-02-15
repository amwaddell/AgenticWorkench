"""
Evaluation datasets: question sets and loaders.

Ships with a built-in set of 20 history questions designed to exercise
timeline extraction, multi-source citation, and factual grounding.

Usage:
    from workbench.evaluation.datasets import load_history_questions
    questions = load_history_questions()
    for q in questions:
        print(q["id"], q["question"])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# -------------------------------------------------------------------- #
#  Built-in history question set                                        #
# -------------------------------------------------------------------- #

HISTORY_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "hist_001",
        "question": "When did the Roman Republic end and what caused its fall?",
        "topic": "Ancient Rome",
        "expected_date_range": "133 BC – 27 BC",
    },
    {
        "id": "hist_002",
        "question": "What were the main events of the French Revolution?",
        "topic": "French Revolution",
        "expected_date_range": "1789 – 1799",
    },
    {
        "id": "hist_003",
        "question": "How did World War I start and what were its major turning points?",
        "topic": "World War I",
        "expected_date_range": "1914 – 1918",
    },
    {
        "id": "hist_004",
        "question": "What was the timeline of the American Civil War?",
        "topic": "American Civil War",
        "expected_date_range": "1861 – 1865",
    },
    {
        "id": "hist_005",
        "question": "How did the Byzantine Empire rise and fall?",
        "topic": "Byzantine Empire",
        "expected_date_range": "330 – 1453",
    },
    {
        "id": "hist_006",
        "question": "What were the key events of the Industrial Revolution in Britain?",
        "topic": "Industrial Revolution",
        "expected_date_range": "1760 – 1840",
    },
    {
        "id": "hist_007",
        "question": "What was the timeline of the Cold War?",
        "topic": "Cold War",
        "expected_date_range": "1947 – 1991",
    },
    {
        "id": "hist_008",
        "question": "How did ancient Egypt's Old Kingdom rise and decline?",
        "topic": "Ancient Egypt",
        "expected_date_range": "2686 BC – 2181 BC",
    },
    {
        "id": "hist_009",
        "question": "What were the major events leading to the fall of the Berlin Wall?",
        "topic": "Berlin Wall",
        "expected_date_range": "1961 – 1989",
    },
    {
        "id": "hist_010",
        "question": "How did the Mongol Empire expand under Genghis Khan?",
        "topic": "Mongol Empire",
        "expected_date_range": "1206 – 1227",
    },
    {
        "id": "hist_011",
        "question": "What were the causes and consequences of the Black Death in Europe?",
        "topic": "Black Death",
        "expected_date_range": "1347 – 1353",
    },
    {
        "id": "hist_012",
        "question": "How did the Renaissance begin and spread across Europe?",
        "topic": "Renaissance",
        "expected_date_range": "1300 – 1600",
    },
    {
        "id": "hist_013",
        "question": "What was the timeline of the American Revolution?",
        "topic": "American Revolution",
        "expected_date_range": "1765 – 1783",
    },
    {
        "id": "hist_014",
        "question": "How did the Ottoman Empire rise to power?",
        "topic": "Ottoman Empire",
        "expected_date_range": "1299 – 1453",
    },
    {
        "id": "hist_015",
        "question": "What were the key events of the Russian Revolution?",
        "topic": "Russian Revolution",
        "expected_date_range": "1905 – 1922",
    },
    {
        "id": "hist_016",
        "question": "How did the Silk Road develop and what was its impact?",
        "topic": "Silk Road",
        "expected_date_range": "130 BC – 1453",
    },
    {
        "id": "hist_017",
        "question": "What were the main events of the Reformation?",
        "topic": "Reformation",
        "expected_date_range": "1517 – 1648",
    },
    {
        "id": "hist_018",
        "question": "How did Japan modernise during the Meiji Restoration?",
        "topic": "Meiji Restoration",
        "expected_date_range": "1868 – 1912",
    },
    {
        "id": "hist_019",
        "question": "What were the causes and key battles of World War II in Europe?",
        "topic": "World War II",
        "expected_date_range": "1939 – 1945",
    },
    {
        "id": "hist_020",
        "question": "How did the ancient Greek city-states develop and interact?",
        "topic": "Ancient Greece",
        "expected_date_range": "800 BC – 146 BC",
    },
]


def load_history_questions() -> list[dict[str, Any]]:
    """
    Return the built-in set of 20 history evaluation questions.

    Returns:
        List of question dicts with keys: id, question, topic,
        expected_date_range.
    """
    return list(HISTORY_QUESTIONS)


def load_questions_from_file(path: Path) -> list[dict[str, Any]]:
    """
    Load a custom question set from a JSON file.

    Expected format — a JSON array of objects, each with at least
    ``id`` and ``question`` keys.

    Args:
        path: Path to the JSON file.

    Returns:
        List of question dicts.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If JSON is malformed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    for i, item in enumerate(data):
        if "id" not in item or "question" not in item:
            raise ValueError(
                f"Item {i} missing required keys 'id' and/or 'question': {item}"
            )

    return data


def save_questions_to_file(questions: list[dict[str, Any]], path: Path) -> None:
    """
    Save a question set to a JSON file.

    Args:
        questions: List of question dicts.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
