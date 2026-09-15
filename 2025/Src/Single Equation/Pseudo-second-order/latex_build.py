# latex_build.py
import os, subprocess
from pathlib import Path

def build_tex_doc(title_text: str, body: str) -> str:
    safe_title = title_text.replace("&", r"\&")
    return r"""\documentclass[10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{microtype}
\usepackage{float}
\usepackage{graphicx}
\usepackage{longtable}
\setlength{\parskip}{6pt}
\setlength{\parindent}{0pt}

\begin{document}

\begin{center}
{\large\bfseries """ + safe_title + r"""}
\end{center}

\begin{center}
\textbf{Extended Abstract}
\end{center}

""" + body + r"""

\end{document}
"""

def ensure_tex_on_path():
    candidates = ["/Library/TeX/texbin", "/usr/texbin", "/usr/local/bin", "/usr/bin"]
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for p in candidates:
        if p and os.path.isdir(p) and p not in path_parts:
            path_parts.insert(0, p)
    os.environ["PATH"] = os.pathsep.join(path_parts)


def compile_tex(tex_path: Path, workdir: Path):
    def run_once():
        return subprocess.run( ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name], cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True )

    p1 = run_once()
    if p1.returncode != 0:
        return False, p1.stdout
    p2 = run_once()
    return (p2.returncode == 0), p2.stdout


def show_first_latex_error(log_text: str, max_lines: int = 60) -> str:
    lines = (log_text or "").splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("!"):
            start = max(0, i - 5)
            end = min(len(lines), i + max_lines)
            return "\n".join(lines[start:end])
    return "\n".join(lines[:max_lines])