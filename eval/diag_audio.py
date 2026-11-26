import sys, os, json, shutil, subprocess, tempfile, collections
import pyarrow.parquet as pq
root = sys.argv[1]
P = os.path.join(root, "data/DarijaCS-Prism/train-00000-of-00001.parquet")
Q = os.path.join(root, "data/DarijaCS-Probe/train-00000-of-00001.parquet")

def kind(b):
    if not b: return "vide"
    if b[:4] == b"RIFF": return "wav"
    if b[:3] == b"ID3" or b[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"): return "mp3"
    if b[4:8] == b"ftyp": return "mp4/m4a(" + b[8:12].decode(errors="replace") + ")"
    if b[:4] == b"OggS": return "ogg"
    if b[:4] == b"fLaC": return "flac"
    if b[:4] == b"\x1aE\xdf\xa3": return "webm/mkv"
    return "inconnu:" + b[:12].hex()

ffprobe = shutil.which("ffprobe")
print("ffprobe :", ffprobe)
def probe(b):
    with tempfile.NamedTemporaryFile() as f:
        f.write(b); f.flush()
        r = subprocess.run([ffprobe, "-v", "error", "-show_entries",
            "stream=codec_name,sample_rate,channels:format=duration", "-of", "json", f.name],
            capture_output=True, text=True)
    j = json.loads(r.stdout or "{}"); s = (j.get("streams") or [{}])[0]
    return (s.get("codec_name"), s.get("sample_rate"), s.get("channels")), float(j.get("format", {}).get("duration") or 0), r.stderr.strip()[:200]

df = pq.read_table(P).to_pandas()
print("\n===== PRISM =====")
print("Signatures :", collections.Counter(kind(a.get("bytes")) for a in df["audio"]))
print("Extensions 'path' :", collections.Counter(str(a.get("path")).rsplit(".", 1)[-1].lower() for a in df["audio"]))
print("Exemples de path :", [a.get("path") for a in df["audio"].head(3)])
if ffprobe:
    fmt, durs, errs = collections.Counter(), [], []
    for fn, a in zip(df["Filename"], df["audio"]):
        k, d, e = probe(a["bytes"]); fmt[k] += 1
        if d > 0: durs.append(d)
        else: errs.append((fn, e))
    print("(codec, sr, canaux) :", dict(fmt))
    if durs:
        s = sorted(durs)
        print(f"n={len(s)} total={sum(s)/3600:.2f} h min={s[0]:.2f}s médiane={s[len(s)//2]:.2f}s max={s[-1]:.2f}s")
    print("Échecs ffprobe :", len(errs), errs[:3])
print("Pace manquant :", int(df["Pace"].isna().sum()))
dup = df[df["Text"].duplicated(keep=False)]
print("Textes dupliqués (Filename) :", dup.groupby("Text")["Filename"].apply(list).tolist())
for t, g in dup.groupby("Text"):
    b = [a["bytes"] for a in g["audio"]]
    print("  audios identiques ?", all(x == b[0] for x in b))
print("Code switching = 0 :", df.loc[df["Code switching"] == 0, "Filename"].tolist())

dq = pq.read_table(Q).to_pandas()
print("\n===== PROBE : références vides ou type Silence =====")
m = dq["Text"].fillna("").str.strip().eq("") | dq["type"].fillna("").str.contains("Silence")
for _, r in dq[m].iterrows():
    print(r["id"], "|", r["type"], "|", repr(r["Text"])[:120])
print("Colonne 'observations' non vide :", int(dq["observations"].notna().sum()))
