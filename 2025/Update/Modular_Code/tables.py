# tables.py
import re
import pandas as pd

def parse_col_key(key: str):
    """
    Splits 'qt_(mg/g)' -> ('qt', 'mg/g')
    If no unit pattern is present, returns (key, '').
    """
    m = re.match(r"^(.+?)_\((.+)\)$", key)
    if m:
        return m.group(1), m.group(2)
    return key, ""

def _header_cell(sym: str, unit: str, VARMAP: dict) -> str:
    """
    Builds a compact two-line header using \shortstack:
      q_t
      (mg/g)
    """
    unit = (unit or "").strip()

    # Use math version from VARMAP if available (gives q_t, C_e, etc.)
    math_sym = VARMAP.get(sym, {}).get("math", sym)

    if unit:
        # IMPORTANT: no spaces inside parentheses
        return rf"\shortstack{{{math_sym}\\({unit})}}"
    return rf"\shortstack{{{math_sym}}}"

def df_to_latex_summary_table(df: pd.DataFrame, caption: str, label: str, VARMAP: dict) -> str:
    """
    Table 1: mean/SD/min/max (excluding run_id)
    Uses LaTeX table float + caption/label for Table~\\ref{...}.
    Header units are on a new line to reduce width.
    """
    df2 = df.copy()
    if "run_id" in df2.columns:
        df2 = df2.drop(columns=["run_id"])

    desc = df2.describe().loc[["mean", "std", "min", "max"]].copy()
    desc.index = ["Mean", "SD", "Min", "Max"]

    cols = list(desc.columns)
    header_cells = []
    for c in cols:
        sym, unit = parse_col_key(c)
        header_cells.append(_header_cell(sym, unit, VARMAP))
    header = " & ".join(header_cells)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c"*len(cols) + r"}",
        r"\hline",
        r" & " + header + r" \\",
        r"\hline",
    ]

    for idx, row in desc.iterrows():
        vals = " & ".join(f"{float(v):.2f}" for v in row.values)
        lines.append(f"{idx} & {vals} \\\\")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)

def df_to_latex_full_table(df: pd.DataFrame, caption: str, label: str, VARMAP: dict, max_rows: int | None = None) -> str:
    """
    Table 2: full dataset (optionally truncated)
    """
    df2 = df.copy()
    if max_rows is not None:
        df2 = df2.head(max_rows)

    cols = list(df2.columns)

    header_cells = []
    for c in cols:
        if c == "run_id":
            header_cells.append(r"\shortstack{Run}")
        else:
            sym, unit = parse_col_key(c)
            header_cells.append(_header_cell(sym, unit, VARMAP))
    header = " & ".join(header_cells)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\renewcommand{\arraystretch}{1.02}",
        r"\tiny",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + "c"*len(cols) + r"}",
        r"\hline",
        header + r" \\",
        r"\hline",
    ]

    for _, row in df2.iterrows():
        vals = []
        for c in cols:
            if c == "run_id":
                vals.append(str(int(row[c])))
            else:
                vals.append(f"{float(row[c]):.4f}")
        lines.append(" & ".join(vals) + r" \\\\")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


import re
def insert_after_first_paragraph(body: str, insertion: str) -> str:
    # paragraphs are separated by blank lines
    parts = re.split(r"\n\s*\n", body, maxsplit=1)
    if len(parts) == 2:
        return parts[0] + "\n\n" + insertion + "\n\n" + parts[1]
    # fallback: if there is only one paragraph
    return body + "\n\n" + insertion