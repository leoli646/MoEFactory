from __future__ import annotations

import json


def first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == chr(92):
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def post_process_intent(text: str) -> str:
    prediction = text.strip()
    if not prediction:
        return "None"
    fence = chr(96) * 3
    json_fence = fence + "json"
    if json_fence in prediction:
        fenced = prediction.split(json_fence, 1)[1]
        prediction = fenced.split(fence, 1)[0]
    json_object = first_json_object(prediction)
    if json_object is not None:
        return json_object
    first_line = next((line for line in prediction.splitlines() if line), prediction)
    return first_line.strip(".。")


def score_intent(prediction: str, ground_truth: str) -> float:
    try:
        predicted = json.loads(prediction)
    except Exception:
        return 0.0
    expected = json.loads(ground_truth)
    if not isinstance(predicted, dict) or not isinstance(expected, dict):
        return 0.0

    intent_em = predicted.get("intent", "") == expected.get("intent", "")
    try:
        expected_slots = {
            (key, str(tuple(sorted([str(item) for item in value]))) if isinstance(value, list) else value)
            for key, value in expected.get("slots", {}).items()
        }
        predicted_slots = {
            (
                key,
                str(tuple(sorted([str(item).replace(" ", "") for item in value])))
                if isinstance(value, list)
                else value.replace(" ", ""),
            )
            for key, value in predicted.get("slots", {}).items()
        }
    except Exception:
        return 0.0

    correct = predicted_slots.intersection(expected_slots)
    slots_em = len(correct) == len(predicted_slots) and len(correct) == len(expected_slots)
    return float(intent_em and slots_em)
