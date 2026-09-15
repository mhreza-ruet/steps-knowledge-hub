# utils_json.py
import json, re

def extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def repair_json_loose(s: str) -> str:
    if not s:
        return s
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)

    out, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                out.append(ch); esc = False
            else:
                if ch == "\\":
                    out.append(ch); esc = True
                elif ch == '"':
                    out.append(ch); in_str = False
                elif ch == "\n":
                    out.append("\\n")
                else:
                    out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_str = True
    return "".join(out)

def json_from_llm(text: str) -> dict:
    block = extract_json_object(text)
    if not block:
        return {}
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        fixed = repair_json_loose(block)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return {}