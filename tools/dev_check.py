#!/usr/bin/env python3
"""Dev only (not copied to the image): run phase 1 on a COPY of a tasks dir and compare with `expected`."""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path
src = Path(sys.argv[1]); extra = sys.argv[2:]
tmp = Path(tempfile.mkdtemp()) / "tasks"; shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("__MACOSX", ".DS_Store"))
subprocess.run([sys.executable, str(Path(__file__).parent / "run_agent.py"), "--tasks-dir", str(tmp), *extra])
ok = 0
for d in sorted(p for p in tmp.iterdir() if p.is_dir()):
    cfg = json.loads((d / "task.json").read_text()); ans = None
    for a in (d / "answer.txt", d / "workspace" / "answer.txt"):
        if a.exists(): ans = a.read_text().strip(); break
    exp, chk = str(cfg.get("expected")), cfg.get("check")
    if chk == "unittest":
        r = subprocess.run([sys.executable, "test_mylib.py"], cwd=d / "workspace", capture_output=True)
        good = ans == "DONE" and r.returncode == 0
    else:
        good = ans == exp
    ok += good
    print(f"{d.name:22} reward={cfg.get('reward')} answer={ans!r:10} expected={exp!r:12} {'OK' if good else 'FAIL'}")
print(f"{ok}/{len([1 for p in tmp.iterdir() if p.is_dir()])} correct")
