#!/usr/bin/env python3
"""Check that every requirement in requirements.txt is installed on this machine with a matching version.

Usage: python scripts/check_deps.py requirements.txt
Exit code 0 = all satisfied, 1 = something missing or mismatched.
"""
import re
import sys
from importlib import metadata


def parse_version(v):
    parts = [int(p) for p in re.findall(r"\d+", v)[:4]]
    return tuple(parts + [0] * (4 - len(parts)))


def satisfies(installed, spec):
    """spec like '>=1.30.0,<2.0.0' or '==0.1.19'. Returns (ok, bad_clause)."""
    for clause in (c.strip() for c in spec.split(",")):
        if not clause:
            continue
        m = re.match(r"(==|!=|>=|<=|~=|>|<)\s*(\S+)", clause)
        if not m:
            continue
        op, want = m.groups()
        a, b = parse_version(installed), parse_version(want)
        good = {"==": a == b, "!=": a != b, ">=": a >= b, "<=": a <= b,
                ">": a > b, "<": a < b, "~=": a >= b}[op]
        if not good:
            return False, clause
    return True, ""


def main(path):
    bad = 0
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            m = re.match(r"([A-Za-z0-9_.\-]+)\s*(.*)", line)
            name, spec = m.group(1), m.group(2).strip()
            try:
                installed = metadata.version(name)
            except metadata.PackageNotFoundError:
                print(f"[ERROR] not installed: {name} {spec}")
                bad += 1
                continue
            ok, clause = satisfies(installed, spec) if spec else (True, "")
            if ok:
                print(f"[OK] {name} {installed}")
            else:
                print(f"[ERROR] {name} {installed} does not match '{clause}' (from requirements.txt)")
                bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: check_deps.py requirements.txt")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
