"""Inférence en streaming via l'API Realtime de vLLM (WebSocket /v1/realtime).

Protocole (exemple officiel vLLM « openai_realtime_client ») :
  session.created -> session.update(model) -> input_audio_buffer.commit
  -> input_audio_buffer.append (PCM16 16 kHz base64, blocs de 4096 octets = 128 ms)
  -> input_audio_buffer.commit(final=True) -> transcription.delta* -> transcription.done

Le délai de transcription (480 ms) est celui de tekken.json ; il est vérifié par check_delay.py.
Les prédictions brutes (avant normalisation) sont écrites dans --out. Les échecs vont dans
<out>.errors.jsonl et ne sont jamais écrits comme prédictions vides : compute_metrics.py
refusera alors de calculer tant qu'ils n'ont pas été relancés (reprise automatique).
"""
import argparse, asyncio, base64, csv, json, os, time
import numpy as np, pandas as pd, soundfile as sf, websockets

FIELDS = ["sample_id", "hypothesis_raw", "text_source", "deltas_equal_final", "n_deltas",
          "inference_time_s", "first_delta_s", "audio_duration_s", "n_chunks"]


def pcm16_chunks(path, chunk_bytes):
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != 16000 or a.ndim != 1:
        raise ValueError(f"{path}: attendu 16 kHz mono, obtenu sr={sr} ndim={a.ndim}")
    pcm = (np.clip(a, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    return [pcm[i:i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)], len(a) / sr


async def transcribe(uri, model, path, chunk_bytes, pace, timeout):
    chunks, dur = pcm16_chunks(path, chunk_bytes)
    chunk_s = chunk_bytes / 2 / 16000
    deltas, state = [], {"first": None, "final": None}
    t0 = time.perf_counter()
    async with websockets.connect(uri, max_size=None) as ws:
        ev = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        if ev.get("type") != "session.created":
            raise RuntimeError(f"premier événement inattendu : {ev}")
        await ws.send(json.dumps({"type": "session.update", "model": model}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        async def sender():
            for c in chunks:
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": base64.b64encode(c).decode("utf-8")}))
                if pace > 0:
                    await asyncio.sleep(chunk_s / pace)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        async def receiver():
            while True:
                ev = json.loads(await ws.recv())
                t = ev.get("type")
                if t == "transcription.delta":
                    if state["first"] is None:
                        state["first"] = time.perf_counter() - t0
                    deltas.append(ev.get("delta", ""))
                elif t == "transcription.done":
                    state["final"] = ev.get("text")
                    return
                elif t == "error":
                    raise RuntimeError(json.dumps(ev, ensure_ascii=False)[:500])

        send_task = asyncio.create_task(sender())
        try:
            await asyncio.wait_for(receiver(), timeout)
        finally:
            send_task.cancel() if not send_task.done() else None
        await asyncio.gather(send_task, return_exceptions=True)
    joined = "".join(deltas)
    final = state["final"]
    return {
        "hypothesis_raw": final if final is not None else joined,
        "text_source": "transcription.done" if final is not None else "deltas_concatenes",
        "deltas_equal_final": (final is None) or (final.strip() == joined.strip()),
        "n_deltas": len(deltas),
        "inference_time_s": round(time.perf_counter() - t0, 4),
        "first_delta_s": None if state["first"] is None else round(state["first"], 4),
        "audio_duration_s": round(dur, 3),
        "n_chunks": len(chunks),
    }


async def main_async(a):
    man = pd.concat([pd.read_csv(m, dtype={"sample_id": str}) for m in a.manifests], ignore_index=True)
    done = set()
    if os.path.exists(a.out):
        done = set(pd.read_csv(a.out, dtype={"sample_id": str}, keep_default_na=False)["sample_id"])
    todo = man[~man.sample_id.isin(done)]
    print(f"{len(man)} échantillons, {len(done)} déjà faits, {len(todo)} à traiter", flush=True)
    uri = f"ws://{a.host}:{a.port}/v1/realtime"
    new = not os.path.exists(a.out)
    fout = open(a.out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=FIELDS)
    if new:
        w.writeheader()
    ferr = open(a.out + ".errors.jsonl", "a", encoding="utf-8")
    sem = asyncio.Semaphore(a.concurrency)
    counter = {"ok": 0, "err": 0}

    async def one(row):
        async with sem:
            last = None
            for attempt in range(1, a.retries + 2):
                try:
                    r = await transcribe(uri, a.model, row.audio_path, a.chunk_bytes, a.pace, a.timeout)
                    w.writerow({"sample_id": row.sample_id, **r}); fout.flush()
                    counter["ok"] += 1
                    print(f"[{counter['ok'] + counter['err']}/{len(todo)}] {row.sample_id} "
                          f"{r['inference_time_s']}s : {r['hypothesis_raw'][:80]!r}", flush=True)
                    return
                except Exception as e:
                    last = repr(e)
                    await asyncio.sleep(2 * attempt)
            counter["err"] += 1
            ferr.write(json.dumps({"sample_id": row.sample_id, "error": last}, ensure_ascii=False) + "\n")
            ferr.flush()
            print(f"ÉCHEC {row.sample_id} : {last}", flush=True)

    await asyncio.gather(*(one(r) for r in todo.itertuples()))
    fout.close(); ferr.close()
    print(f"Terminé : {counter['ok']} réussis, {counter['err']} échecs (voir {a.out}.errors.jsonl)")
    return counter["err"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="nom servi par vLLM (--served-model-name)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--chunk-bytes", type=int, default=4096, help="4096 octets PCM16 = 128 ms (exemple vLLM)")
    ap.add_argument("--pace", type=float, default=0.0,
                    help="0 = envoi sans attente ; 1 = cadence temps réel ; 2 = 2x plus vite que le temps réel")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=300)
    a = ap.parse_args()
    raise SystemExit(1 if asyncio.run(main_async(a)) else 0)


if __name__ == "__main__":
    main()
