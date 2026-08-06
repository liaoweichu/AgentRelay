import importlib.util, subprocess, os, sys
mods = ["faiss", "pyserini", "spacy", "cleantext", "rank_bm25", "gym", "flask", "gdown", "bs4", "torch", "numpy"]
for m in mods:
    print(f"{m:12s}", "OK" if importlib.util.find_spec(m) else "MISSING")
print("--- java? ---")
r = subprocess.run(["java", "-version"], capture_output=True, text=True)
print("java:", (r.stderr or r.stdout).strip().splitlines()[0] if (r.stdout or r.stderr) else "NOT FOUND")
print("JAVA_HOME:", os.environ.get("JAVA_HOME", "unset"))