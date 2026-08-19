import json
import re


def extract_json_from_markdown(text):
    """Extract JSON from a markdown text that may contain ```json ... ``` blocks."""
    if not isinstance(text, str):
        return None

    if text.strip().startswith("{") and text.strip().endswith("}"):
        try:
            json.loads(text.strip())
            return text.strip()
        except json.JSONDecodeError:
            pass

    if "```json" in text and "```" in text[text.find("```json") + 7 :]:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            json_str = text[start:end].strip()
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass

    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        json_str = match.group(1).strip()
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass

    try:
        json_obj = json.loads(text.strip())
        return text.strip()
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start != -1:
        level = 0
        for i, char in enumerate(text[start:]):
            if char == "{":
                level += 1
            elif char == "}":
                level -= 1
                if level == 0:
                    end = start + i + 1
                    potential_json = text[start:end]
                    try:
                        json.loads(potential_json)
                        return potential_json
                    except json.JSONDecodeError:
                        pass
                    break

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        potential_json = text[start : end + 1]
        try:
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            pass

    if "comprehensiveness" in text and "article_1_score" in text and "article_2_score" in text:
        try:
            dimensions = ["comprehensiveness", "insight", "instruction_following", "readability"]
            result = {}

            for dim in dimensions:
                if dim in text:
                    result[dim] = []

                    dim_start = text.find(f'"{dim}"')
                    if dim_start == -1:
                        dim_start = text.find(f"'{dim}'")
                    if dim_start == -1:
                        dim_start = text.find(dim)

                    if dim_start != -1:
                        next_dim_start = len(text)
                        for next_dim in dimensions:
                            if next_dim != dim:
                                pos = text.find(f'"{next_dim}"', dim_start)
                                if pos == -1:
                                    pos = text.find(f"'{next_dim}'", dim_start)
                                if pos == -1:
                                    pos = text.find(next_dim, dim_start + len(dim))
                                if pos != -1 and pos < next_dim_start:
                                    next_dim_start = pos

                        dim_content = text[dim_start:next_dim_start]

                        criterion_matches = re.finditer(r'"criterion"\s*:\s*"([^"]+)"', dim_content)
                        score1_matches = re.finditer(r'"article_1_score"\s*:\s*(\d+\.?\d*)', dim_content)
                        score2_matches = re.finditer(r'"article_2_score"\s*:\s*(\d+\.?\d*)', dim_content)

                        criteria = [m.group(1) for m in criterion_matches]
                        scores1 = [float(m.group(1)) for m in score1_matches]
                        scores2 = [float(m.group(1)) for m in score2_matches]

                        for i in range(min(len(criteria), len(scores1), len(scores2))):
                            result[dim].append(
                                {
                                    "criterion": criteria[i],
                                    "article_1_score": scores1[i],
                                    "article_2_score": scores2[i],
                                }
                            )

            if any(len(scores) > 0 for scores in result.values()):
                return json.dumps(result)
        except Exception:
            pass

    return None
