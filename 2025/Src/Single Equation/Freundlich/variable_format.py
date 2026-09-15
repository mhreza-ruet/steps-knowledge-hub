# variable_format.py
import re

def parse_col_key(key: str):
    m = re.match(r"^(.+?)_\((.+)\)$", key)
    if m:
        return m.group(1), m.group(2)
    return key, ""

def symbol_to_math(symbol: str) -> str:
    s = symbol.strip()
    if "_" in s:
        base, sub = s.split("_", 1)
        sub = sub.strip("{}")
        return rf"$\mathit{{{base}}}_{{{sub}}}$"
    if re.fullmatch(r"[A-Za-z]", s):
        return rf"$\mathit{{{s}}}$"
    m = re.fullmatch(r"([A-Za-z]+)([0-9]+)", s)
    if m:
        return rf"$\mathit{{{m.group(1)}}}_{{{m.group(2)}}}$"
    m = re.fullmatch(r"([A-Za-z])([A-Za-z0-9]+)", s)
    if m:
        return rf"$\mathit{{{m.group(1)}}}_{{{m.group(2)}}}$"
    return rf"$\mathit{{{s}}}$"

def build_var_map_from_df_cols(cols: list[str]) -> dict:
    varmap = {}
    for k in cols:
        sym, _unit = parse_col_key(k)
        varmap[sym] = {"math": symbol_to_math(sym)}
    return varmap

def normalize_broken_variable_spellings(text: str) -> str:
    return re.sub(r"\b([A-Za-z])\s+([A-Za-z0-9])\b", r"\1\2", text)

def canonicalize_emph_symbol(s: str) -> str:
    if not s:
        return s
    x = s.strip()
    x = re.sub(r"[.,;:]+$", "", x)
    x = re.sub(r"\s+", "", x)
    x = re.sub(r"\\+_", "_", x)
    m = re.fullmatch(r"([A-Za-z]+)_\{?([A-Za-z0-9]+)\}?", x)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return x

def convert_emph_variable_to_math(inner_raw: str) -> str | None:
    if not inner_raw:
        return None
    x = inner_raw.strip()
    if len(x) > 30 or re.search(r"\s", x):
        return None
    x = re.sub(r"[.,;:]+$", "", x)
    x = re.sub(r"\\+_", "_", x)
    x = x.replace("{", "").replace("}", "")

    if "_" in x:
        base, sub = x.split("_", 1)
        if base and sub:
            return rf"$\mathit{{{base}}}_{{{sub}}}$"

    m = re.fullmatch(r"([A-Za-z]+)([0-9]+)", x)
    if m:
        return rf"$\mathit{{{m.group(1)}}}_{{{m.group(2)}}}$"

    m = re.fullmatch(r"([A-Za-z])([A-Za-z0-9]+)", x)
    if m:
        return rf"$\mathit{{{m.group(1)}}}_{{{m.group(2)}}}$"

    if re.fullmatch(r"[A-Za-z]", x):
        return rf"$\mathit{{{x}}}$"
    return None

_MATH_INLINE_RE = re.compile(r"(\$.*?\$)", re.S)

def apply_variable_math(text: str, VARMAP: dict, VARS: list[str]) -> str:
    if not text:
        return text
    text = normalize_broken_variable_spellings(text)

    def repl_emph(m):
        inner_raw = m.group(1)
        math = convert_emph_variable_to_math(inner_raw)
        if math:
            canon = canonicalize_emph_symbol(inner_raw)
            if canon in VARMAP:
                return VARMAP[canon]["math"]
            return math
        return r"\emph{" + inner_raw + "}"

    text = re.sub(r"\\emph\{([^{}]*)\}", repl_emph, text)

    parts = _MATH_INLINE_RE.split(text)
    for i, chunk in enumerate(parts):
        if not chunk:
            continue
        if chunk.startswith("$") and chunk.endswith("$"):
            continue
        for sym in VARS:
            rep = VARMAP[sym]["math"]
            chunk = re.sub(rf"\b{re.escape(sym)}\b", lambda _m, r=rep: r, chunk)
        chunk = re.sub(r"(?<!\\)_", r"\\_", chunk)
        parts[i] = chunk
    return "".join(parts)

def remove_variables_from_first_paragraph(body: str, VARS: list[str]) -> str:
    parts = re.split(r"\n\s*\n", body, maxsplit=1)
    first = parts[0]
    rest  = parts[1] if len(parts) > 1 else ""
    for sym in VARS:
        first = re.sub(rf"\b{re.escape(sym)}\b", "the measured quantity", first)
        first = re.sub(rf"\\emph\{{\s*{re.escape(sym)}\s*\}}", "the measured quantity", first)
    return first + ("\n\n" + rest if rest else "")