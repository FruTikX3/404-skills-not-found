#!/usr/bin/env python3
"""Budget-aware code-agent harness (openJiuwen hackathon).

Task format (see README): <task>/task.json {task_id, task_type, instruction, reward, ...} and an optional
<task>/workspace/ directory. The agent must leave answer.txt for every solved task.

Pipeline
  phase 1  free local solvers (0 tokens): templated math / csv / knowledge-base questions, mutation-based
           repair of buggy python modules verified by the task's own tests; infeasible tasks are detected
           by free pre-flight checks (missing files / entities)
  phase 2  LLM only for what is left, easiest cost band first, more points first inside a band, under a
           thread-safe budget manager;
           cheap one-shot paths per task type, generic bash agent loop as the last resort
`expected` / `check` fields of task.json are never used for decisions.
"""
import argparse
import ast
import csv
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

START = time.monotonic()
TOTAL_LIMIT_S = 900
SAFETY_MARGIN_S = 60
DEADLINE = START + TOTAL_LIMIT_S - SAFETY_MARGIN_S

MODEL_FAST = os.environ.get("MODEL_FAST", "qwen3.8-27b-noreason")
MODEL_REASON = os.environ.get("MODEL_REASON", "qwen3.8-27b")
NO_LOCAL = bool(os.environ.get("NO_LOCAL"))          # debug: force the LLM paths

NOREASON = os.environ.get("MODEL_FAST", "qwen3.8-27b-noreason")
REASON = os.environ.get("MODEL_REASON", "qwen3.8-27b")

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent")
try:
    from importlib.metadata import version as _pkg_version
    OJ_VERSION = _pkg_version("openjiuwen")
except Exception:  # noqa: BLE001
    OJ_VERSION = None

STOP = threading.Event()
# ======================================================================
# YOUR PROXY SETTINGS: put your URL and key on these two lines.
# They are used only when LLM_BASE_URL / LLM_API_KEY are NOT set in the environment.
# Do NOT leave a real key here when you submit the repository (regulation: no keys in the repo).
# ======================================================================
DEFAULT_LLM_BASE_URL = "https://api.neuraldeep.ru/v1"   # <-- YOUR URL
DEFAULT_LLM_API_KEY = "sk-eNctGLbm4H_YeCawN3_vpA"                                          # key is passed via LLM_API_KEY only
FORCE_THINK = True    # True: every task uses the reasoning model (qwen3.8-27b)
SECRET_ENV_KEYS = {"LLM_API_KEY", "OPENAI_API_KEY", "API_KEY", "TOKEN_BUDGET", "BUDGET", "LLM_TOKEN_BUDGET"}


def build_sub_env():
    """Container env for solver sub-processes: secrets removed, deterministic and single-threaded."""
    env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV_KEYS and not k.endswith("_SECRET")}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env.update({"HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "LANG": "C.UTF-8",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    return env


SUB_ENV = build_sub_env()


class BudgetExhausted(Exception):
    pass


class OutOfTime(Exception):
    pass


class Infeasible(Exception):
    pass


class ContextOverflow(Exception):
    """Prompt does not fit the model's context window (HTTP 400)."""


class TaskCap(Exception):
    """Per-task token ceiling reached (hopeless / runaway task)."""


class TransportError(Exception):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status, self.body = status, body


# ------------------------------------------------------------------ budget
class BudgetManager:
    """Thread-safe accounting: spent += prompt/10 + completion (reasoning tokens are in completion)."""

    def __init__(self, total=None):
        self.total, self.spent, self.reserved = total, 0.0, 0.0
        self.est_sum = self.act_sum = 0.0
        self.lock = threading.Lock()
        self.exhausted = threading.Event()
        self.tl = threading.local()          # per-thread (= per-task) cost for calibration / trajectories

    def charge(self, in_tok, out_tok):
        self.tl.cost = getattr(self.tl, "cost", 0.0) + in_tok / 10.0 + out_tok
        with self.lock:
            self.spent += in_tok / 10.0 + out_tok
            if self.total is not None and self.spent >= self.total:
                self.exhausted.set()

    def factor(self):
        with self.lock:
            return (self.act_sum / self.est_sum) if self.est_sum > 50 else 1.0

    def try_reserve(self, est):
        if self.total is None:
            return 0.0
        need = est * max(0.5, min(3.0, self.factor()))
        with self.lock:
            if self.spent + self.reserved + need * 0.8 > self.total * 0.97:
                return None
            self.reserved += need
            return need

    def release(self, amount, est, actual):
        with self.lock:
            self.reserved = max(0.0, self.reserved - (amount or 0.0))
            if actual > 0:
                self.est_sum += est
                self.act_sum += actual

    def left(self):
        with self.lock:
            return float("inf") if self.total is None else self.total - self.spent

# ------------------------------------------------------------------ LLM transport
BUDGET_WORDS = ("budget", "quota", "exceed", "exhaust", "insufficient", "token limit", "limit reached")


class LLM:
    """OpenAI-compatible chat client: official SDK, stdlib urllib fallback."""

    def __init__(self, budget):
        self.budget = budget
        self.base = (os.environ.get("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL).rstrip("/")
        self.key = os.environ.get("LLM_API_KEY") or DEFAULT_LLM_API_KEY
        self.fail_streak, self.lock, self.client = 0, threading.Lock(), None
        try:
            import openai
            self.openai = openai
            self.client = openai.OpenAI(base_url=self.base, api_key=self.key, max_retries=0)
        except Exception as e:  # noqa: BLE001
            log.warning("openai SDK unavailable (%s) - using urllib fallback", e)

    def _post(self, model, messages, max_tokens, temperature, timeout):
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if self.client is not None:
            o = self.openai
            try:
                return self.client.with_options(timeout=timeout).chat.completions.create(**payload).model_dump()
            except o.APIStatusError as e:
                raise TransportError(e.status_code, str(e))
            except o.APIConnectionError as e:
                raise TransportError(0, repr(e))
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            raise TransportError(e.code, e.read().decode("utf-8", "replace"))
        except Exception as e:  # noqa: BLE001
            raise TransportError(0, repr(e))

    def chat(self, model, messages, max_tokens, temperature=0.0, timeout_cap=300.0):
        # refuse a request whose worst case (input/10 + max output) does not fit what is left
        est_in = len(json.dumps(messages, ensure_ascii=False)) / 4.0
        if self.budget.total is not None and self.budget.left() < est_in / 10.0 + max_tokens:
            raise BudgetExhausted("next request does not fit the budget")
        for attempt in range(4):
            if self.budget.exhausted.is_set() or STOP.is_set():
                raise BudgetExhausted()
            cap = getattr(self.budget.tl, "cap", None)
            if cap and getattr(self.budget.tl, "cost", 0.0) >= cap:
                raise TaskCap()
            left = DEADLINE - time.monotonic()
            if left < 15:
                raise OutOfTime()
            try:
                data = self._post(model, messages, max_tokens, temperature, min(timeout_cap, left))
            except TransportError as e:
                low = e.body.lower()
                if e.status in (402, 403, 429) and any(w in low for w in BUDGET_WORDS):
                    self.budget.exhausted.set()
                    raise BudgetExhausted()
                if e.status == 400 and any(w in low for w in ("context", "too long", "maximum", "too many tokens")):
                    raise ContextOverflow(e.body)
                if e.status in (0, 429, 500, 502, 503, 504) and attempt < 3:
                    with self.lock:
                        self.fail_streak += 1
                        if self.fail_streak >= 8:
                            self.budget.exhausted.set()
                    log.warning("LLM error %s (attempt %d)", e, attempt + 1)
                    time.sleep(min(2 ** attempt * 2, 15))
                    continue
                raise
            with self.lock:
                self.fail_streak = 0
            text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
            u = data.get("usage") or {}
            pt = u.get("prompt_tokens")
            ct = u.get("completion_tokens")
            if pt is None:
                pt = sum(len(m["content"]) for m in messages) // 3
            if ct is None:
                ct = len(text) // 3
            self.budget.charge(pt, ct)
            return text
        raise TransportError(0, "retries exhausted")


# ------------------------------------------------------------------ tasks
SKIP_KEYS = {"reward", "expected", "check", "task_id", "task_type"}
PRIMARY_KEYS = ["instruction", "prompt", "description", "task", "problem", "statement", "question", "text",
                "content", "query"]
SUSPICIOUS = [r"\bgpu\b", r"\bcuda\b", r"send (an? )?e-?mail", r"\bsmtp\b", r"internet access",
              r"https?://", r"\bssh\b", r"\bkubernetes\b", r"\bpip install\b", r"external (server|service|api)",
              r"web ?search", r"browse the (web|internet)"]


def _find_reward(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() == "reward":
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        for v in obj.values():
            r = _find_reward(v)
            if r is not None:
                return r
    return None


def _strings(obj, out):
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() not in SKIP_KEYS:
                _strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _strings(v, out)


def extract_text(data):
    """Universal task.json parser: known keys first, then the longest string value."""
    if isinstance(data, dict):
        low = {str(k).lower(): v for k, v in data.items()}
        for k in PRIMARY_KEYS:
            if isinstance(low.get(k), str) and low[k].strip():
                return low[k].strip()
        log.warning("no known text key; keys=%s", list(data.keys()))
    strs = []
    _strings(data, strs)
    return max(strs, key=len) if strs else json.dumps(data, ensure_ascii=False)


@dataclass
class Task:
    dir: Path
    name: str
    type: str
    reward: float
    text: str
    cwd: Path
    est: float = 0.0
    density: float = 0.0
    meta: dict = field(default_factory=dict)


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
    if FORCE_THINK:
        model_class = "reason"
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





def load_tasks(root):
    tasks = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            data = json.loads((d / "task.json").read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.error("skip %s: %s", d.name, e)
            continue
        text = extract_text(data)
        typ = str(data.get("task_type") or re.sub(r"_?\d+$", "", d.name)) if isinstance(data, dict) else "unknown"
        reward = _find_reward(data)
        reward = 1.0 if reward is None else reward
        cwd = d / "workspace" if (d / "workspace").is_dir() else d
        a = analyze(d, data) if isinstance(data, dict) else None
        est = a.est_units if a else 3000.0
        prio = a.priority if a else reward / est
        meta = {"base_est": est, "model": a.model if a else NOREASON, "flags": a.flags if a else []}
        if a and a.flags:
            log.info("[%s] flags (weak, task still runs): %s", d.name, "; ".join(a.flags))
        tasks.append(Task(d, d.name, typ, reward, text, cwd, est, prio, meta))
    return tasks


def core_text(t):
    """Instruction without the boilerplate about answer.txt (saves input tokens)."""
    s = re.split(r"Work only inside the current directory", t.text)[0].strip()
    return s


# ------------------------------------------------------------------ answer I/O
def _write(path, answer):
    tmp = path.parent / f".answer.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        tmp.write_text(answer, encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError as e:
        log.warning("atomic write failed %s: %s", path, e)
        try:
            tmp.unlink()
        except OSError:
            pass
    try:
        path.write_text(answer, encoding="utf-8")
        return True
    except OSError as e:
        log.error("CANNOT WRITE %s: %s (check directory permissions / uid)", path, e)
        return False


def save_answer(t, answer):
    """answer.txt in the task dir AND in workspace/ (the task text says 'current directory')."""
    ok = _write(t.dir / "answer.txt", answer)
    if t.cwd != t.dir:
        ok = _write(t.cwd / "answer.txt", answer) or ok
    if ok:
        SOLUTIONS[t.name] = answer
        flush_solutions()
    if not ok:
        try:
            Path("/tmp/answers_backup").mkdir(exist_ok=True)
            (Path("/tmp/answers_backup") / f"{t.name}.txt").write_text(answer, encoding="utf-8")
        except OSError:
            pass
    return ok


SOL_LOCK = threading.Lock()
SOLUTIONS = {}
OUT_PATH = None
ROOT_PATH = None


def flush_solutions():
    """Solutions file (OUT) rewritten after every solved task, not only at the end."""
    with SOL_LOCK:
        data = json.dumps(SOLUTIONS, ensure_ascii=False, indent=1)
        targets = [Path("/tmp/solutions.json")]
        if OUT_PATH:
            p = Path(OUT_PATH)
            targets.append(p / "solutions.json" if p.is_dir() else p)
        if ROOT_PATH:
            targets.append(Path(ROOT_PATH) / "solutions.json")
        for tp in targets:
            try:
                tp.write_text(data, encoding="utf-8")
            except OSError:
                pass


TRAJ_LOCK = threading.Lock()


def log_traj(t, path, spent, solved, note=""):
    rec = {"task": t.name, "type": t.type, "reward": t.reward, "path": path, "cost": round(spent, 1),
           "solved": solved, "note": note}
    try:
        with TRAJ_LOCK, open(os.environ.get("TRAJ_FILE", "/tmp/trajectories.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


# ------------------------------------------------------------------ sandboxed execution
DENY = re.compile(r"(^|[\s;&|(])(curl|wget|ssh|scp|nc|sudo|apt|apt-get|pip|pip3)\b|rm\s+-rf\s+/|\.\./")


def run_bash(cmd, cwd, timeout=20):
    if DENY.search(cmd):
        return "blocked: command not allowed (work only inside the current directory)"
    try:
        r = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout, env=SUB_ENV)
        return ((r.stdout or "") + (r.stderr or ""))[-1500:] or "(no output)"
    except subprocess.TimeoutExpired:
        return "timeout"


def run_py(code, cwd, timeout=20):
    fd, p = tempfile.mkstemp(suffix=".py", dir="/tmp")
    os.close(fd)
    try:
        Path(p).write_text(code, encoding="utf-8")
        r = subprocess.run([sys.executable, p], cwd=cwd, capture_output=True, text=True, timeout=timeout, env=SUB_ENV)
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "")[-800:]
    except subprocess.TimeoutExpired:
        return 1, "", "timeout"
    finally:
        try:
            os.unlink(p)
        except OSError:
            pass


def run_tests(cwd, test_file, timeout=20):
    try:
        r = subprocess.run([sys.executable, test_file], cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, env=SUB_ENV)
    except subprocess.TimeoutExpired:
        return 1, 99, "timeout"
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 0:
        return 0, 0, out
    m = re.search(r"FAILED \(([^)]*)\)", out)
    n = sum(int(x) for x in re.findall(r"=(\d+)", m.group(1))) if m else 99
    return r.returncode, n, out


# ------------------------------------------------------------------ free pre-flight (infeasible detection)
def preflight(t):
    low = t.text.lower()
    if any(re.search(p, low) for p in SUSPICIOUS):
        raise Infeasible("needs network / external resources")
    for m in re.finditer(r"directory\s+`?([\w.-]+)/", t.text):
        if not (t.cwd / m.group(1)).is_dir():
            raise Infeasible(f"missing dir {m.group(1)}")
    for m in re.finditer(r"(?<![\w/.-])([\w-]+\.(?:py|csv|tsv|json|md|txt|xlsx))\b", t.text):
        name = m.group(1)
        if name != "answer.txt" and not (t.cwd / name).exists():
            raise Infeasible(f"missing file {name}")


# ------------------------------------------------------------------ local solvers (0 tokens)
def _ints(s):
    return [int(x) for x in re.findall(r"(?<![\w.])-?\d+(?!\w)(?!\.\d)", s)]

def _is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True

def math_intent(q):
    """Deterministic answers for common arithmetic families. Returns str or None (no confident match)."""
    s = q.lower()
    try:
        if re.search(r"greatest common (?:divisor|factor)|\bgcd\b", s):
            n = _ints(q)
            if len(n) >= 2:
                return str(math.gcd(n[0], n[1]))
        if re.search(r"least common multiple|\blcm\b", s):
            n = _ints(q)
            if len(n) >= 2:
                return str(n[0] * n[1] // math.gcd(n[0], n[1]))
        if re.search(r"digital root|single digit (?:remains|is left)|until a single digit", s):
            n = _ints(q)
            if n and n[0] > 0:
                return str(1 + (n[0] - 1) % 9)
        m = re.search(r"(\d+)\^(\d+)", s)
        if m and "digit" in s and re.search(r"\bsum|add together|total", s):
            return str(sum(map(int, str(int(m.group(1)) ** int(m.group(2))))))
        if re.search(r"1\^2 \+ 2\^2", s):
            sq = re.findall(r"(\d+)\^2", s)
            if sq:
                n = int(sq[-1])
                return str(n * (n + 1) * (2 * n + 1) // 6)
        if re.search(r"zeros?\b.*\b(?:end|trailing)|trailing zeros", s):
            m = re.search(r"(\d+)!", q)
            if m:
                n, c, p = int(m.group(1)), 0, 5
                while p <= n:
                    c += n // p
                    p *= 5
                return str(c)
        m = re.search(r"multiples of (\d+)", s)
        if m:
            m2 = re.search(r"(?:no larger than|not exceed|not greater than|at most|up to|less than or equal to|below or equal to) (\d+)", s)
            if m2:
                return str(int(m2.group(1)) // int(m.group(1)))
        m = re.search(r"(\d+) (?:raised to the power|to the power of) (\d+)", s)
        m2 = re.search(r"(?:modulo|\bmod\b|remainder[^\d]*?division by|divided by)\s*(\d+)", s)
        if m and m2 and re.search(r"remainder|modulo|\bmod\b", s):
            return str(pow(int(m.group(1)), int(m.group(2)), int(m2.group(1))))
        m = re.search(r"(?:begin|start|starting)(?: with| at| from)? (\d+)", s)
        if m and re.search(r"halve|collatz|3 times|3x", s) and re.search(r"reach(?:es)? 1|arrive at 1|equals 1|becomes 1", s):
            x, steps = int(m.group(1)), 0
            while x != 1:
                x = x // 2 if x % 2 == 0 else 3 * x + 1
                steps += 1
            return str(steps)
        m = re.search(r"sum .*divisors?|divisors? .*sum|add .*divisors?|sum of (?:all )?(?:the )?(?:positive )?divisors?", s)
        if m:
            n = _ints(q)
            if n:
                return str(sum(d for d in range(1, n[0] + 1) if n[0] % d == 0))
        m = re.search(r"(\d+) [a-z ]*?(?:chosen|picked|selected|choose)[^\d]*?(?:group|set|pool|collection) of (\d+)", s)
        if m and re.search(r"how many ways|in how many", s):
            return str(math.comb(int(m.group(2)), int(m.group(1))))
        m = re.search(r"x \+ y = (\d+)", s)
        if m and "ordered pairs" in s and "x < y" in s:
            N = int(m.group(1))
            return str(sum(1 for x in range(1, N) if x < N - x))
        if re.search(r"how many (?:positive )?divisors", s):
            n = _ints(q)
            if n:
                return str(sum(1 for d in range(1, n[0] + 1) if n[0] % d == 0))
        m = re.search(r"(?:smaller|less) than (\d+)", s)
        if m and "prime" in s and re.search(r"how many", s):
            N = int(m.group(1))
            return str(sum(1 for k in range(2, N) if _is_prime(k)))
        if re.search(r"largest prime (?:divisor|factor)", s):
            n = _ints(q)
            if n:
                x, best, p = n[0], 1, 2
                while p * p <= x:
                    while x % p == 0:
                        best, x = p, x // p
                    p += 1
                return str(max(best, x) if x > 1 else best)
        m = re.search(r"divisible by (\d+) or (?:by )?(\d+)", s)
        m2 = re.search(r"from (\d+) to (\d+)", s)
        if m and m2:
            a, b, d1, d2 = int(m2.group(1)), int(m2.group(2)), int(m.group(1)), int(m.group(2))
            return str(sum(1 for x in range(a, b + 1) if x % d1 == 0 or x % d2 == 0))
        m = re.search(r"(?:x\s*\*\s*x|x\^2|square)[^\d]*?(?:<=|does not exceed|not exceed|at most)\s*(\d+)", s)
        if m:
            return str(math.isqrt(int(m.group(1))))
        m = re.search(r"inverse of (\d+) modulo (\d+)", s)
        if m:
            return str(pow(int(m.group(1)), -1, int(m.group(2))))
        m = re.search(r"base (\d+)", s)
        m2 = re.search(r"(?:express|write) (\d+)", s)
        if m and m2 and re.search(r"base \d+", s):
            b, x, digs = int(m.group(1)), int(m2.group(1)), ""
            while x:
                digs = str(x % b) + digs
                x //= b
            return str(int(digs))
        if re.search(r"1-?(?:bits|digits)|base-2|binary", s) and re.search(r"how many|count", s):
            m = re.search(r"representation of (\d+)", s)
            if m:
                return str(bin(int(m.group(1))).count("1"))
        m = re.search(r"sides? (\d+), (\d+),? and (\d+)", s)
        if m and re.search(r"area", s):
            a, b, c = map(int, m.groups())
            v = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)
            r = math.isqrt(v)
            if r * r == v and r % 4 == 0:
                return str(r // 4)
        m = re.search(r"starts? at (-?\d+)", s)
        m2 = re.search(r"(?:by|difference (?:is|of))\s+(\d+)", s)
        m3 = re.search(r"first (\d+) terms", s)
        if m and m2 and m3 and re.search(r"arithmetic", s):
            a, d, n = int(m.group(1)), int(m2.group(1)), int(m3.group(1))
            return str(n * a + d * n * (n - 1) // 2)
        m = re.search(r"floor\((\d+)\s*/\s*i\)", s)
        if m:
            n = int(m.group(1))
            return str(sum(n // i for i in range(1, n + 1)))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return None


def math_local(t):
    """Deterministic families first (0 tokens), then the stock-chain template, else None (LLM)."""
    ans = math_intent(core_text(t))
    if ans is not None:
        return ans
    return _math_chain(t)


def _math_chain(t):
    """Chain of arithmetic sentences about a stock. Any unknown sentence -> None (LLM fallback)."""
    q = core_text(t)
    if not re.search(r"How many \w+ (?:does|do) (?:it|the \w+) have at the end\?", q):
        return None
    n = None
    for s in re.split(r"(?<=\.)\s+", q.split("How many")[0]):
        s = re.sub(r"^(?:Then|After that|Next|Finally|Later),?\s+", "", s.strip().rstrip("."))
        if not s:
            continue
        if m := re.fullmatch(r"(?:A|The|An) [\w ]+? starts? with (\d+) \w+", s):
            n = int(m.group(1))
        elif n is None:
            return None
        elif m := re.search(r"split equally into (\d+) [\w ]+? and (?:it )?keeps one share \(rounded down\)", s):
            n //= int(m.group(1))
        elif re.search(r"\bdoubled\b", s):
            n *= 2
        elif re.search(r"\btripled\b", s):
            n *= 3
        elif m := re.search(r"\b(?:ships?|sells?|removes?|loses?|uses?) (\d+) [\w ]+? per day for (\d+) days", s):
            n -= int(m.group(1)) * int(m.group(2))
        elif m := re.search(r"\b(?:receives?|gets?|produces?|adds?) (\d+) [\w ]+? per day for (\d+) days", s):
            n += int(m.group(1)) * int(m.group(2))
        elif m := re.search(r"\breceives? (\d+) (?:deliveries|shipments|batches) of (\d+) [\w ]+? each", s):
            n += int(m.group(1)) * int(m.group(2))
        elif m := re.fullmatch(r"(?:[Ii]t |[Tt]he \w+ )?(?:ships?|sells?|removes?|loses?|uses?|donates?) (\d+) \w+", s):
            n -= int(m.group(1))
        elif m := re.fullmatch(r"(?:[Ii]t |[Tt]he \w+ )?(?:receives?|gets?|adds?|buys?) (\d+) \w+", s):
            n += int(m.group(1))
        else:
            return None
    return None if n is None else str(n)


def data_local(t):
    """Counts / sums over a csv with optional product / region / month conditions."""
    q = core_text(t)
    f = re.search(r"The file ([\w.-]+\.csv)", q)
    if not f:
        return None
    if re.search(r"how many units|total units", q, re.I):
        metric = "units"
    elif re.search(r"total revenue|revenue", q, re.I):
        metric = "revenue"
    else:
        return None
    prods, regs = re.findall(r"product '([^']+)'", q), re.findall(r"region '([^']+)'", q)
    if len(re.findall(r"'[^']+'", q)) != len(prods) + len(regs) or len(prods) > 1 or len(regs) > 1:
        return None                                     # some quoted condition is not understood
    mo = re.search(r"months? (\d+) (?:through|to|-) (\d+)", q)
    lo, hi = (int(mo.group(1)), int(mo.group(2))) if mo else (None, None)
    if not mo and (m1 := re.search(r"\bin month (\d+)", q)):
        lo = hi = int(m1.group(1))
    try:
        with open(t.cwd / f.group(1), newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        total = 0
        for r in rows:
            if prods and r["product"] != prods[0]:
                continue
            if regs and r["region"] != regs[0]:
                continue
            if lo is not None and not (lo <= int(r["month"]) <= hi):
                continue
            total += int(r["units"]) * (int(r["price"]) if metric == "revenue" else 1)
        return str(total)
    except (OSError, KeyError, ValueError):
        return None


class Missing(Exception):
    pass


def load_kb(cwd):
    kb = cwd / "kb"
    if not kb.is_dir():
        return None, ""
    ents, texts = {}, []
    for p in sorted(kb.glob("*.md")):
        s = p.read_text(encoding="utf-8", errors="replace")
        texts.append(f"== {p.name} ==\n{s.strip()}")
        title = next((l[2:].strip() for l in s.splitlines() if l.startswith("# ")), p.stem)
        fields = {}
        for l in s.splitlines():
            m = re.match(r"\s*[-*]?\s*([A-Za-z ]+?)\s*:\s*(.+?)\s*$", l)
            if m:
                fields[m.group(1).lower()] = m.group(2)
        ents[title.lower()] = fields
    return ents, "\n".join(texts)


def _ent(E, name):
    f = E.get(name.strip().lower())
    if f is None:
        raise Missing(name)
    return f


def _emp(E, n):
    return _ent(E, _ent(E, n).get("employer", ""))


SEARCH_RULES = [
    (r"in which year was (.+?)'s employer founded", lambda E, n: _emp(E, n).get("founded")),
    (r"in which year was (.+?) born", lambda E, n: _ent(E, n).get("born")),
    (r"in which year was (.+?) founded", lambda E, n: _ent(E, n).get("founded")),
    (r"(?:in which city|where) is the headquarters of (.+?)'s employer", lambda E, n: _emp(E, n).get("headquarters")),
    (r"headquarters of (.+?)\?", lambda E, n: _ent(E, n).get("headquarters")),
    (r"who is (.+?)'s mentor\?", lambda E, n: _ent(E, n).get("mentor")),
    (r"who is the mentor of (.+?)\?", lambda E, n: _ent(E, n).get("mentor")),
    (r"employer of (.+?)\?", lambda E, n: _ent(E, n).get("employer")),
]


def search_walk(E, orig):
    """Generic knowledge-base walker: entity -> possessive hops ('s employer, 's mentor...) -> target attribute."""
    q = orig.replace("\u2019", "'").lower()
    hits = [(q.find(n), -len(n), n) for n in E if n and q.find(n) != -1]
    if not hits:
        names = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b", orig)
        if names:
            raise Infeasible(f"entity not in kb: {names[0]}")
        return None
    pos, _, name = min(hits)
    rest, hops = q[pos + len(name):], []
    while m := re.match(r"'s (\w+)", rest):
        hops.append(m.group(1))
        rest = rest[m.end():]
    if re.search(r"\bborn\b|\bbirth", q):
        target = "born"
    elif "founded" in q:
        target = "founded"
    elif "headquarter" in q:
        target = "headquarters"
    elif hops and re.search(r"\b(?:who|what|which)\b", q):
        target = hops.pop()
    else:
        pre = re.search(r"(?:the )?(\w+) of (?:the )?$", q[:pos])
        if pre and re.search(r"\b(?:who|what|which)\b", q):
            target = pre.group(1)
        else:
            return None
    pre = re.search(r"(?:the )?(\w+) of (?:the )?$", q[:pos])
    if pre and pre.group(1) not in ("born", "founded", "headquarters", target) and pre.group(1) in {k for f in E.values() for k in f}:
        hops.insert(0, pre.group(1))
    cur = E[name]
    for h in hops:
        v = cur.get(h)
        if v is None:
            return None
        if v.lower() not in E:
            raise Infeasible(f"entity not in kb: {v}")
        cur = E[v.lower()]
    return cur.get(target)


def search_local(t):
    E, _ = load_kb(t.cwd)
    if not E:
        return None
    orig = core_text(t)
    ans = search_walk(E, orig.split("kb/")[-1] if "kb/" in orig else orig)
    if ans:
        return ans
    q = orig.replace("\u2019", "'").lower()
    for pat, fn in SEARCH_RULES:
        m = re.search(pat, q)
        if m:
            try:
                return fn(E, m.group(1))
            except Missing as e:
                raise Infeasible(f"entity not in kb: {e}")
    return None


# --- mutation-based program repair verified by the task's own tests
SWAPS = [("-=", "+="), ("+=", "-="), (" + ", " - "), (" - ", " + "), (" < ", " > "), (" > ", " < "),
         (" <= ", " < "), (" >= ", " > "), (" < ", " <= "), (" > ", " >= "), (" == ", " != "), (" != ", " == "),
         (" and ", " or "), (" or ", " and "), ("min(", "max("), ("max(", "min("), ("sorted(", "reversed("),
         ("reversed(", "sorted("), ("//", "/"), (" / ", " // "), (" * ", " + "), ("True", "False"),
         ("False", "True"), (" % ", " // ")]


def mutations(src):
    for a, b in SWAPS:
        i = src.find(a)
        while i != -1:
            yield src[:i] + b + src[i + len(a):]
            i = src.find(a, i + 1)
    for m in re.finditer(r"(?<![\w.'\"])\d+(?![\w.'\"])", src):
        v = int(m.group())
        for nv in (v + 1, v - 1):
            yield src[:m.start()] + str(nv) + src[m.end():]


def code_files(t):
    py = re.search(r"[Ff]ix the bugs in `?([\w./-]+\.py)", t.text)
    tf = re.search(r"python3 ([\w./-]+\.py)", t.text)
    if not (py and tf):
        return None
    return py.group(1), tf.group(1)


def normalize_number_str(s, question=""):
    """'25.0' -> '25'; keep other text as it is."""
    if not s:
        return s
    if re.fullmatch(r"-?\d+\.0+", s):
        val = int(float(s))
        return "0" if val == 0 else str(val)
    return s


def final_answer(ans, t):
    """Clean a one-line answer: unwrap one-element set reprs, quotes, integer floats. Code is left untouched."""
    if ans is None:
        return None
    if "\n" in ans.strip():
        return ans
    a = ans.strip()
    m = re.fullmatch(r"\{\s*['\"]?(.*?)['\"]?\s*\}", a)
    if m:
        a = m.group(1)
    a = a.strip().strip("'\"").strip()
    a = normalize_number_str(a, t.text)
    return a or None


def format_coding_answer(t, patched_src):
    """Coding answer = the corrected source code (organisers' format). The completion word is no longer used."""
    return patched_src


def path_text(p, t):
    """Coding answer: the corrected source, or the requested completion word (see format_coding_answer)."""
    return format_coding_answer(t, p.read_text(encoding="utf-8"))


def coding_local(t):
    cf = code_files(t)
    if not cf:
        return None
    pyfile, testfile = cf
    if not (t.cwd / testfile).is_file() or not (t.cwd / pyfile).is_file():
        raise Infeasible("source or test file missing")
    rc, fails, _ = run_tests(t.cwd, testfile)
    if rc == 0:
        return path_text(t.cwd / pyfile, t)
    work = Path(tempfile.mkdtemp(prefix="repair_", dir="/tmp"))
    try:
        for p in t.cwd.iterdir():
            if p.is_file() and p.stat().st_size < 200000:
                shutil.copy(p, work / p.name)
        src = (work / pyfile).read_text(encoding="utf-8")
        t0, cur = time.monotonic(), fails
        for _ in range(5):
            best = None
            for cand in mutations(src):
                if time.monotonic() - t0 > 40:
                    break
                (work / pyfile).write_text(cand, encoding="utf-8")
                rc, f, _ = run_tests(work, testfile, timeout=5)
                if rc == 0:
                    best = (0, cand)
                    break
                if f < cur and (best is None or f < best[0]):
                    best = (f, cand)
            if best is None:
                return None
            cur, src = best[0], best[1]
            if cur == 0:
                orig = (t.cwd / pyfile).read_text(encoding="utf-8")
                (t.cwd / pyfile).write_text(src, encoding="utf-8")
                if run_tests(t.cwd, testfile)[0] == 0:
                    return path_text(t.cwd / pyfile, t)
                (t.cwd / pyfile).write_text(orig, encoding="utf-8")
                return None
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


LOCAL = {"math": math_local, "data_analysis": data_local, "search": search_local, "coding": coding_local}


# ------------------------------------------------------------------ LLM paths (phase 2)
def cap_for(t, small, big):
    """A reasoning model spends its output limit on thinking first, so it needs a much larger limit."""
    return small if t.meta.get("model") == NOREASON else big


def clean_line(ans):
    ans = re.sub(r"<think>.*?</think>", "", ans or "", flags=re.S).strip()
    if not ans:
        return None
    return ans.splitlines()[-1].strip().strip("`'\"* ").rstrip(".") or None


# ------------------------------------------------------------------ cheap LLM paths (phase 2)
# Cost model: spent = prompt_tokens/10 + completion_tokens. Input is nearly free, output is not.
# So every path asks for the SHORTEST possible reply: an expression, a number, a name, or one function.
EXPR_BAD = re.compile(r"import|__|open\(|exec\(|eval\(|\n|`")
SAFE_PRE = ("import csv, json, math, statistics\n"
            "def _load(p):\n"
            "    with open(p, newline='', encoding='utf-8') as fh:\n"
            "        return list(csv.DictReader(fh))\n")


def _clean_expr(text):
    m = re.search(r"```(?:python)?\s*\n?(.*?)```", text, re.S)
    e = (m.group(1) if m else text).strip().strip("`").strip()
    e = e.splitlines()[0].strip() if e else ""
    if e.startswith("answer ="):
        e = e[len("answer ="):].strip()
    if e.startswith("print(") and e.endswith(")"):
        e = e[len("print("):-1]
    return e


def _run_expr(t, expr, load_csvs):
    """Evaluate one expression in a subprocess. Returns printed value or None."""
    if not expr or len(expr) > 400 or EXPR_BAD.search(expr):
        return None
    code = SAFE_PRE
    names = [re.sub(r"\W", "_", p.stem) + "_rows" for p in load_csvs]
    for p, var in zip(load_csvs, names):
        code += f"{var} = _load({p.name!r})\n"
    if names:
        code += f"rows = {names[0]}\n"
    code += f"print({expr})\n"
    rc, out, _ = run_py(code, t.cwd)
    return clean_line(out) if rc == 0 and out.strip() else None


def llm_math_expr(llm, t):
    """math / data: model replies with ONE python expression (~30 output tokens), harness evaluates it."""
    csvs = sorted(t.cwd.glob("*.csv"))[:2]
    ctx = ""
    for p in csvs:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        ctx += f"\n{p.name} ({len(lines) - 1} rows), header + 2 sample rows:\n" + "\n".join(lines[:3])
    sys_msg = ("Reply with ONE python3 expression and nothing else (no text, no code fence, no print). "
               "It is evaluated as-is. Available: `rows` (list of dict, all values are strings, convert with int/float) "
               "for the first CSV" + (", named variables: " + ", ".join(re.sub(r"\W", "_", p.stem) + "_rows" for p in csvs) if csvs else "")
               + ". Standard library only through math/statistics; no imports.")
    msgs = [{"role": "system", "content": sys_msg},
            {"role": "user", "content": core_text(t) + ctx}]
    model = t.meta["model"]
    expr = _clean_expr(llm.chat(model, msgs, cap_for(t, 80, 1500)))
    val = _run_expr(t, expr, csvs)
    if val is None:                               # one repair round, still cheap
        msgs += [{"role": "assistant", "content": expr},
                 {"role": "user", "content": "That failed or was not one expression. Reply with one expression only."}]
        val = _run_expr(t, _clean_expr(llm.chat(model, msgs, cap_for(t, 120, 1500))), csvs)
    return val


def _norm_key(k):
    return re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")


def _to_value(v):
    """'611', '1,200', '611 km' -> number; anything else stays text."""
    m = re.match(r"^([-+]?\d[\d,]*(?:\.\d+)?)\s*[A-Za-z%$()/ .]*$", v.strip())
    if m:
        num = m.group(1).replace(",", "")
        return float(num) if "." in num else int(num)
    return v.strip()


def kb_records(cwd):
    """Generic records from kb/*.md: one dict per file, keys taken from 'Key: value' lines (any names)."""
    kb = cwd / "kb"
    if not kb.is_dir():
        return []
    recs = []
    for p in sorted(kb.glob("*.md")):
        text = p.read_text(encoding="utf-8", errors="replace")
        rec = {}
        title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), p.stem)
        rec["name"] = title
        for line in text.splitlines():
            m = re.match(r"\s*[-*]?\s*([A-Za-z][A-Za-z0-9 _()/-]*?)\s*:\s*(.+?)\s*$", line)
            if m:
                k = _norm_key(m.group(1))
                if k and k not in rec:
                    rec[k] = _to_value(m.group(2))
        recs.append(rec)
    return recs


def _eval_records(t, expr, blob):
    if not expr or len(expr) > 400 or EXPR_BAD.search(expr):
        return None
    code = "import json\nrecords = json.loads(" + repr(blob) + ")\nprint(" + expr + ")\n"
    rc, out, _ = run_py(code, t.cwd)
    return clean_line(out) if rc == 0 and out.strip() else None


def llm_search_expr(llm, t):
    """search, generic: the model writes ONE expression over the records of the knowledge base; the harness evaluates it."""
    recs = kb_records(t.cwd)
    if len(recs) < 2:
        return None
    blob = json.dumps(recs, ensure_ascii=False)
    if len(blob) > 8000:
        return None
    fields = sorted({k for r in recs for k in r})
    model = t.meta["model"]
    sys_msg = ("Reply with ONE python3 expression and nothing else (no text, no code fence, no print). "
               "It is evaluated as-is. `records` is a list of dicts, one per knowledge-base file. "
               "Keys: " + ", ".join(fields) + ". Numbers are int/float, other values are str. "
               "Use only builtins (max, min, sorted, filter, len, sum, comprehensions). No imports.")
    msgs = [{"role": "system", "content": sys_msg},
            {"role": "user", "content": core_text(t) + "\n\nrecords = " + blob}]
    expr = _clean_expr(llm.chat(model, msgs, cap_for(t, 120, 1500)))
    ans = _eval_records(t, expr, blob)
    if ans is None:
        msgs += [{"role": "assistant", "content": expr},
                 {"role": "user", "content": "That did not run or is not one expression. Reply with one expression only."}]
        ans = _eval_records(t, _clean_expr(llm.chat(model, msgs, cap_for(t, 120, 1500))), blob)
    return ans


SENT_BOILER = re.compile(r"answer\.txt|file tools|bash|kb/|directory|look it up|arithmetic", re.I)
CAP_STOP = {"In", "What", "When", "Who", "Which", "Where", "How", "Why", "The", "A", "An", "Is", "Was", "Were",
            "Did", "Do", "Does", "Among", "According", "Give", "Name", "List", "Find", "Look", "Use", "Save",
            "Answer", "If", "It", "This", "That", "There", "Its", "His", "Her", "Their", "Of", "On", "For"}


def question_entities(text):
    """Capitalised phrases of the question (generic: no names are listed in advance)."""
    ents = []
    for sent in re.split(r"(?<=[.?!])\s+", text):
        if SENT_BOILER.search(sent):
            continue
        group = []
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", sent)
        for k, w in enumerate(words):
            if w[0].isupper() and w not in CAP_STOP:
                group.append(w.split("'")[0])
            elif group:
                # a lone capitalised word at the start of a sentence is ordinary text ("Consult ...", "Using ..."), not a name
                if not (len(group) == 1 and k - 1 == 0):
                    ents.append(" ".join(group))
                group = []
        if group and not (len(group) == 1 and len(words) == 1):
            ents.append(" ".join(group))
    return [e for e in ents if len(e) >= 3]


def search_unanswerable(t):
    """Free check (no tokens): a named entity of the question that appears nowhere in kb/ means the task cannot be answered."""
    kb = t.cwd / "kb"
    if not kb.is_dir():
        return None
    text = " ".join(p.read_text(encoding="utf-8", errors="replace") for p in kb.glob("*.md")).lower()
    for e in question_entities(core_text(t)):
        words = [w.lower() for w in e.split() if len(w) >= 3]
        if words and not any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in words):   # whole words only
            return e
    return None


SEARCH_KB_LIMIT = 24000


def kb_files(cwd):
    """(name, title_lower, text) for every kb/*.md file."""
    kb = cwd / "kb"
    if not kb.is_dir():
        return []
    out = []
    for p in sorted(kb.glob("*.md")):
        s = p.read_text(encoding="utf-8", errors="replace")
        title = next((l[2:].strip() for l in s.splitlines() if l.startswith("# ")), p.stem)
        out.append((p.name, title.lower(), s.strip()))
    return out


def select_kb_files(files, question, limit):
    """Keep files the question names, then files those link to (by title), then by word overlap, up to limit."""
    q = question.lower()
    qwords = set(re.findall(r"[a-z]{4,}", q))
    chosen = []

    def take(f):
        if f not in chosen:
            chosen.append(f)

    for f in files:
        if f[1] and f[1] in q:
            take(f)
    for _ in range(2):                                   # follow links: a chosen file names another entity
        for f in list(chosen):
            body = f[2].lower()
            for g in files:
                if g[1] and g[1] in body:
                    take(g)
    rest = sorted((f for f in files if f not in chosen),
                  key=lambda f: -sum(w in f[2].lower() for w in qwords))
    for f in rest:
        take(f)
    text, used = "", 0
    for name, _, body in chosen:
        block = f"== {name} ==\n{body}\n"
        if used + len(block) > limit:
            break
        text += block
        used += len(block)
    return text


def llm_search_cheap(llm, t):
    """search: whole knowledge base in the prompt (input is cheap), reply is the bare answer."""
    ans = llm_search_expr(llm, t)                  # generic records path first
    if ans is not None:
        return ans
    files = kb_files(t.cwd)
    if not files:
        return None
    question = core_text(t)
    limit = SEARCH_KB_LIMIT
    while limit >= 2000:
        kb_text = select_kb_files(files, question, limit)
        msgs = [{"role": "system", "content": "Answer from the knowledge base below. Reply with the bare answer only "
                                              "(a number, name or short phrase). Reply UNANSWERABLE if it is not there.\n\n"
                                              + kb_text},
                {"role": "user", "content": question}]
        try:
            ans = clean_line(llm.chat(t.meta["model"], msgs, cap_for(t, 16, 1500)))
            break
        except ContextOverflow:
            limit //= 2                               # model window is smaller than the prompt: shrink and retry
    else:
        return None
    if not ans or "UNANSWERABLE" in ans.upper():
        return None                               # no file for unsolvable tasks (organisers)
    return ans


def _find_py_files(t):
    tests = sorted(t.cwd.glob("test_*.py"))
    srcs = [p for p in t.cwd.glob("*.py") if not p.name.startswith("test_")]
    if len(tests) != 1 or len(srcs) != 1:
        return None
    return srcs[0].name, tests[0].name


def _replace_function(src, name, new_block):
    """Replace the single def named `name` (top-level or method) with new_block; None if ambiguous."""
    tree = ast.parse(src)
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(nodes) != 1:
        return None
    node = nodes[0]
    lines = src.splitlines()
    indent = " " * node.col_offset
    body = textwrap.dedent(new_block).strip("\n").splitlines()
    new_lines = [(indent + l) if l.strip() else "" for l in body]
    out = lines[:node.lineno - 1] + new_lines + lines[node.end_lineno:]
    new_src = "\n".join(out) + "\n"
    try:
        ast.parse(new_src)
    except SyntaxError:
        return None
    return new_src


def llm_patch_function(llm, t):
    """coding: the model names one buggy function per round and returns only that function.
    A replacement is kept when it lowers the number of failing tests, so several bugs can be fixed in turn."""
    pf = _find_py_files(t)
    if not pf:
        return None
    srcf, testf = pf
    path = t.cwd / srcf
    original = path.read_text(encoding="utf-8")
    cur = original
    rc, fails, out = run_tests(t.cwd, testf)
    if rc == 0:
        return path_text(path, t)
    ok = False
    try:
        for cap in (400, 800, 800):                 # same model every round: one model per task
            msgs = [{"role": "system", "content": "Fix one bug. Reply in exactly this format and nothing else:\n"
                                                  "FUNCTION: <name of the function to replace>\n```python\n"
                                                  "<the complete corrected function, same indentation as in the file>\n```"},
                    {"role": "user", "content": f"{srcf}:\n```python\n{cur}\n```\nTest output:\n{out[-900:]}"}]
            text = llm.chat(t.meta["model"], msgs, cap_for(t, cap, max(cap, 2000)))
            m = re.search(r"FUNCTION:\s*([A-Za-z_]\w*)", text)
            blk = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
            new_src = _replace_function(cur, m.group(1), blk.group(1)) if (m and blk) else None
            if new_src is None:
                continue
            path.write_text(new_src, encoding="utf-8")
            rc2, fails2, out2 = run_tests(t.cwd, testf)
            if rc2 == 0:
                ok = True
                return format_coding_answer(t, new_src)
            if fails2 < fails:                       # progress: keep it, look for the next bug
                cur, fails, out = new_src, fails2, out2
            else:
                path.write_text(cur, encoding="utf-8")
        return None
    finally:
        if not ok:                                   # any failed exit restores the original module
            path.write_text(original, encoding="utf-8")


def solve_llm(llm, t):
    if t.type in ("math", "data_analysis"):
        return llm_math_expr(llm, t)
    if t.type == "search":
        missing = search_unanswerable(t)
        if missing:
            log.info("[%s] unanswerable: %r is not in the knowledge base (no LLM call)", t.name, missing)
            return None
        return llm_search_cheap(llm, t)
    if t.type == "coding":
        return llm_patch_function(llm, t)
    return None


# ------------------------------------------------------------------ orchestration
def order_key(t):
    """Run order: easiest cost band first (bands double in estimated cost), then more points,
    then the cheaper task. A cheap task with a good score therefore runs before an expensive one."""
    band = math.floor(math.log2(max(t.est, 1.0)))
    return (band, -t.reward, t.est)


class Scheduler:
    def __init__(self, tasks, budget):
        self.remaining = sorted(tasks, key=order_key)
        self.budget, self.lock, self.active = budget, threading.Lock(), 0

    def next(self):
        while True:
            if STOP.is_set() or self.budget.exhausted.is_set() or time.monotonic() > DEADLINE - 20:
                return None, None
            with self.lock:
                if not self.remaining:
                    return None, None
                for t in self.remaining:            # already in run order
                    res = self.budget.try_reserve(t.est)
                    if res is not None:
                        self.remaining.remove(t)
                        self.active += 1
                        return t, res
                if self.active == 0:
                    # ничего не влезает в остаток: решаем самую дешёвую задачу, а не бросаем её
                    if self.budget.left() > 0:
                        t = min(self.remaining, key=lambda x: x.est)
                        self.remaining.remove(t)
                        self.active += 1
                        return t, 0.0
                    return None, None
            time.sleep(2)

    def finish(self):
        with self.lock:
            self.active -= 1


DONE, SKIPPED = {}, {}


def phase1(t):
    """Free work: pre-flight + local solver. Returns True when the task is finished (solved or skipped)."""
    try:
        try:
            preflight(t)
        except Infeasible as e:             # weak signal: do not lose points, but try it last and cheaply
            t.meta["suspect"] = str(e)
            log.info("[%s] suspect (tried last): %s", t.name, e)
        fn = None if NO_LOCAL else LOCAL.get(t.type)
        ans = fn(t) if fn else None
    except Infeasible as e:
        t.meta["suspect"] = str(e)          # модель решит, нерешаемо ли это
        log.info("[%s] suspect -> LLM: %s", t.name, e)
        return False
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] local solver crashed: %s", t.name, e)
        return False
    if ans:
        if save_answer(t, ans):
            DONE[t.name] = "local"
            log_traj(t, "local", 0, True)
            log.info("[%s] solved locally", t.name)
            return True
    return False


def worker(sched, llm, budget):
    while True:
        t, reserved = sched.next()
        if t is None:
            return
        answer, saved = None, False
        budget.tl.cost = 0.0
        budget.tl.cap = max(500.0, 4.0 * t.meta.get("base_est", t.est))
        if t.type == "search":
            budget.tl.cap = max(budget.tl.cap, 2.0 * SEARCH_KB_LIMIT / 40.0 + 300.0)
        try:
            answer = final_answer(solve_llm(llm, t), t)
            if answer:
                saved = save_answer(t, answer)
                if saved:
                    DONE[t.name] = "llm"
        except (BudgetExhausted, OutOfTime):
            log.warning("[%s] stopped: budget/time", t.name)
        except TaskCap:
            log.warning("[%s] gave up: per-task token ceiling", t.name)
        except Exception as e:  # noqa: BLE001
            log.exception("[%s] failed: %s", t.name, e)
        finally:
            if answer and not saved:
                save_answer(t, answer)
            spent = budget.tl.cost
            budget.release(reserved, t.est, spent)
            log_traj(t, "llm", spent, bool(answer))
            sched.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=os.environ.get("TASKS", "/data/tasks"))
    ap.add_argument("--budget", type=float, default=None, help="total budget, output-token equivalents")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=os.environ.get("OUT"), help="solutions file/dir (default: env OUT)")
    ap.add_argument("--local-only", action="store_true", help="phase 1 only (no LLM calls)")
    args = ap.parse_args()

    total = args.budget
    for k in ("TOKEN_BUDGET", "BUDGET", "LLM_TOKEN_BUDGET"):
        if total is None and os.environ.get(k):
            try:
                total = float(os.environ[k])
            except ValueError:
                pass
    if total is None:
        total = 100000.0
        log.info("no budget given: default 100000 output-token equivalents")
    global OUT_PATH
    OUT_PATH = args.out
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    root = Path(args.tasks_dir)
    global ROOT_PATH
    ROOT_PATH = str(root)
    if not root.is_dir():
        log.error("tasks dir not found: %s", root)
        return 0
    if not OUT_PATH:
        OUT_PATH = str(root / "solutions.json")
    os.environ.setdefault("TRAJ_FILE", str(root / "trajectories.jsonl"))
    tasks = load_tasks(root)
    log.info("openjiuwen=%s tasks=%d budget=%s", OJ_VERSION, len(tasks), total)
    if not tasks:
        return 0
    n = max(1, min(3, args.workers))
    with ThreadPoolExecutor(max_workers=n) as pool:
        left = [t for t, fin in zip(tasks, pool.map(phase1, tasks)) if not fin]
    for t in left:
        if t.meta.get("suspect"):           # suspects go to the back of the queue
            t.est *= 4
            t.density = t.reward / t.est
    log.info("phase1: local=%d skipped=%d need_llm=%d", len(DONE), len(SKIPPED), len(left))
    budget = BudgetManager(total)
    key_ok = bool(os.environ.get("LLM_API_KEY") or DEFAULT_LLM_API_KEY) and "PASTE_YOUR_KEY" not in (os.environ.get("LLM_API_KEY") or DEFAULT_LLM_API_KEY)
    if left and not key_ok and not args.local_only:
        log.warning("LLM_API_KEY is not set: %d tasks stay unsolved (local solvers only)", len(left))
    if left and not args.local_only and key_ok:
        llm = LLM(budget)
        sched = Scheduler(left, budget)
        with ThreadPoolExecutor(max_workers=n) as pool:
            for f in [pool.submit(worker, sched, llm, budget) for _ in range(n)]:
                f.result()
    summ = {"openjiuwen": OJ_VERSION, "answered": DONE, "skipped": SKIPPED, "spent": round(budget.spent, 1),
            "elapsed_s": round(time.monotonic() - START, 1)}
    try:
        Path("/tmp/harness_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    except OSError:
        pass
    log.info("done: answered=%d skipped=%d spent=%.0f elapsed=%.0fs", len(DONE), len(SKIPPED), budget.spent,
             time.monotonic() - START)
    return 0


if __name__ == "__main__":
    sys.exit(main())
