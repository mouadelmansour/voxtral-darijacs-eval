"""Contrôle croisé : inférence Transformers, selon l'exemple du model card (do_sample=False).

Ce n'est PAS le protocole principal : l'audio entier est donné d'un coup à generate().
Il sert uniquement à vérifier que la voie vLLM Realtime donne des transcriptions cohérentes.
"""
import argparse, csv, os, time
import pandas as pd, soundfile as sf, torch
from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--manifests", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

processor = AutoProcessor.from_pretrained(a.model_dir)
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(a.model_dir, dtype=torch.bfloat16, device_map="auto")
model.eval()
print("Processor :", processor.feature_extractor)

man = pd.concat([pd.read_csv(m, dtype={"sample_id": str}) for m in a.manifests], ignore_index=True)
done = set(pd.read_csv(a.out, dtype={"sample_id": str})["sample_id"]) if os.path.exists(a.out) else set()
new = not os.path.exists(a.out)
with open(a.out, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["sample_id", "hypothesis_raw", "inference_time_s", "audio_duration_s"])
    if new:
        w.writeheader()
    for i, r in enumerate(man[~man.sample_id.isin(done)].itertuples(), 1):
        audio, sr = sf.read(r.audio_path, dtype="float32")
        assert sr == processor.feature_extractor.sampling_rate, sr
        inputs = processor(audio, return_tensors="pt").to(model.device, dtype=model.dtype)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, do_sample=False)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
        w.writerow({"sample_id": r.sample_id, "hypothesis_raw": text,
                    "inference_time_s": round(dt, 4), "audio_duration_s": round(len(audio) / sr, 3)})
        f.flush()
        print(f"[{i}] {r.sample_id} {dt:.2f}s : {text[:80]!r}", flush=True)
