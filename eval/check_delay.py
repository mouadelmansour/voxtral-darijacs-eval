"""Affiche la configuration audio/streaming du modèle (délai de transcription en particulier)."""
import json, os, sys

d = sys.argv[1]


def find(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{path}.{k}" if path else k
            if any(s in k.lower() for s in ("delay", "audio", "stream", "pad", "look_")) and not isinstance(v, (dict, list)):
                print(f"  {p} = {v}")
            if k not in ("vocab",):
                find(v, p)
    elif isinstance(o, list) and o and isinstance(o[0], dict) and len(o) < 50:
        for i, v in enumerate(o):
            find(v, f"{path}[{i}]")


for f in ("tekken.json", "processor_config.json", "config.json", "params.json"):
    p = os.path.join(d, f)
    if os.path.exists(p):
        print(f"== {f}")
        if os.path.getsize(p) < 5000:
            print(open(p, encoding="utf-8").read())
        else:
            find(json.load(open(p, encoding="utf-8")))
