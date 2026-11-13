"""Free task pre-analysis: type, rough cost, feasibility signals, one model per task. No LLM calls.

Cost unit = output-token equivalent (input tokens / 10 + output tokens), the same unit as BudgetManager.
Model priors are placeholders until measured (tools/measure.py); keep them in MODEL_PRIOR.
"""
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

NOREASON = "qwen3.8-27b-noreason"
REASON = "qwen3.8-27b"

# success prior and output tokens per model class; replace with measured values
MODEL_PRIOR = {
    "math":          {"noreason": (0.80, 40),  "reason": (0.95, 400)},
    "data_analysis": {"noreason": (0.80, 40),  "reason": (0.93, 400)},
    "search":        {"noreason": (0.75, 20),  "reason": (0.90, 300)},
    "coding":        {"noreason": (0.35, 150), "reason": (0.70, 600)},
}
AGGREGATE = re.compile(r"\b(how many|total|sum|average|mean|largest|smallest|longest|shortest|earliest|latest|"
                       r"most|least|highest|lowest|combined|count|which .* (has|have) the)\b", re.I)
FILE_REF = re.compile(r"(?<![\w/.-])([\w-]+\.(?:py|csv|md|json|txt))\b")


@dataclass
class Analysis:
    task_id: str
    type: str
    reward: float
    model: str
    est_units: float          # expected cost of the chosen model
    p_success: float
    priority: float           # reward * p / est : what to run first
    flags: list = field(default_factory=list)   # weak feasibility signals, never a reason to skip alone


def _text(ws, pattern):
    return sum(p.stat().st_size for p in ws.rglob(pattern) if p.is_file())


def analyze(task_dir, meta):
    ws = task_dir / "workspace" if (task_dir / "workspace").is_dir() else task_dir
    typ = str(meta.get("task_type", "")).lower() or "unknown"
    text = str(meta.get("instruction", ""))
    reward = float(meta.get("reward", 1) or 1)
    flags = []

    # feasibility signals (weak): referenced files that do not exist, entities missing from the kb
    for name in set(FILE_REF.findall(text)):
        if name != "answer.txt" and not (ws / name).exists() and not list(ws.rglob(name)):
            flags.append(f"missing file {name}")
    kb_chars = _text(ws / "kb", "*.md") if (ws / "kb").is_dir() else 0
    if kb_chars:
        kb = " ".join(p.read_text(encoding="utf-8", errors="replace") for p in (ws / "kb").glob("*.md")).lower()
        for name in re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b", text):
            if name.lower() not in kb and not name.startswith("In"):
                flags.append(f"name not in kb: {name}")

    # size of inputs: what the prompt will have to carry (in input chars)
    csv_chars = _text(ws, "*.csv")
    code_chars = sum(p.stat().st_size for p in ws.glob("*.py") if not p.name.startswith("test_"))
    test_chars = sum(p.stat().st_size for p in ws.glob("test_*.py"))
    in_chars = len(text) + min(csv_chars, 600) + kb_chars + code_chars + test_chars + 400
    in_units = in_chars / 40.0                     # about 4 chars per token, billed at 1/10

    # model choice: one model per task, decided here
    if typ == "coding":
        model_class = "reason" if code_chars + test_chars > 600 or len(flags) else "noreason"
    elif typ == "search" and AGGREGATE.search(text):
        model_class = "reason"
    elif typ == "data_analysis" and AGGREGATE.search(text) and csv_chars > 5000:
        model_class = "reason"
    else:
        model_class = "noreason"
    model = NOREASON if model_class == "noreason" else REASON

    prior = MODEL_PRIOR.get(typ, {}).get(model_class, (0.5, 300))
    p_success, out_units = prior
    if typ == "coding":
        out_units *= 3                              # up to three repair rounds
        in_units *= 3
    est = in_units + out_units
    # a missing-entity or missing-file signal lowers the chance, but does not skip the task
    if flags:
        p_success *= 0.5
    priority = reward * p_success / max(est, 1.0)
    return Analysis(task_dir.name, typ, reward, model, est, p_success, priority, flags)


if __name__ == "__main__":
    import json, sys
    root = Path(sys.argv[1])
    rows = []
    for d in sorted(p for p in root.iterdir() if (p / "task.json").exists()):
        a = analyze(d, json.loads((d / "task.json").read_text(encoding="utf-8")))
        rows.append(a)
    for a in sorted(rows, key=lambda r: -r.priority):
        print(f"{a.task_id:22} {a.type:14} r={a.reward:<4} est={a.est_units:6.0f} p={a.p_success:.2f} "
              f"prio={a.priority:.4f} {a.model:22} {'; '.join(a.flags)}")
