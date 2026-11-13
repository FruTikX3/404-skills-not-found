#!/usr/bin/env python3
"""Dev-only diagnostic (not in the image). Run INSIDE the built image to see what openjiuwen 0.2.3 really exposes:
   docker run --rm --entrypoint python hack:latest - < check_openjiuwen.py   (or copy it in with -v)
Send the output to the assistant to wire openJiuwen in as the primary LLM path."""
import importlib, importlib.metadata as md, pkgutil
print("openjiuwen version:", md.version("openjiuwen"))
for mod in ("openjiuwen", "openjiuwen.core.foundation.llm", "openjiuwen.core.runner.runner",
            "openjiuwen.core.single_agent", "openjiuwen.core.workflow"):
    try:
        m = importlib.import_module(mod)
        print(f"\n== {mod}\n", [n for n in dir(m) if not n.startswith("_")][:60])
    except Exception as e:  # noqa: BLE001
        print(f"\n== {mod}: IMPORT FAILED: {e!r}")
try:
    import openjiuwen.core.foundation.llm as llm
    for n in ("Model", "BaseModelClient", "ModelClientConfig", "ModelRequestConfig"):
        o = getattr(llm, n, None)
        if o is not None:
            print(f"\n-- {n}: methods", [x for x in dir(o) if not x.startswith("_")][:40])
except Exception as e:  # noqa: BLE001
    print("llm inspect failed:", repr(e))
print("\ndeps:", sorted(md.requires("openjiuwen") or [])[:40])
