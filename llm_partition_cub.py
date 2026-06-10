"""
LLM symbolic partition generator for CUB (qwen3:14b via ollama) -- the LLM enters at the
SYMBOLIC level (partition the label set), since DINOv2 has no text encoder.

For each of the 200 species (name only, zero-shot), ask the LLM the dominant color of 7 body
parts, constrained to CUB's color vocabulary. Then:
  (1) AGREEMENT CHECK (the kill switch): compare LLM dominant color vs CUB ground-truth dominant
      color (argmax of the 200x312 continuous attribute matrix per part). If ~chance -> the LLM
      doesn't know these birds, stop.
  (2) Emit partition JSONs {f"{part}::{color}": [class_idx,...]} for both LLM and GT over the
      SAME attribute set, consumed by dino_concept_dirs_cub.py for an apples-to-apples
      oracle-vs-LLM concept-direction comparison.

Caches raw LLM responses; CPU-only except the ollama server. Text-only, symbolic (no VLM).
"""
import os, sys, json, time, urllib.request
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np
from datasets.cub200 import CLASSES

ATTR_NAMES_FILE = "dataset/CUB_200_2011/attributes.txt"
ATTR_CONT_FILE = "dataset/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt"
PARTS = ["throat", "breast", "belly", "back", "crown", "wing", "primary"]
COLORS = ["black", "white", "grey", "brown", "buff", "yellow", "olive", "green",
          "blue", "purple", "rufous", "orange", "pink", "red", "iridescent"]
GT_CONF = 30.0   # min % to accept a GT dominant color
MIN_COUNT = 3    # min has-classes for a usable concept attribute
MODEL = "qwen3:14b"
OUT = os.path.join(ROOT, "llm_partition")


def parse_attr_groups():
    """group name -> {color: column_index} for the part-color groups we use."""
    names = [ln.strip().split(" ", 1)[1] for ln in open(ATTR_NAMES_FILE)]
    groups = {}
    for j, nm in enumerate(names):
        if "::" not in nm:
            continue
        g, v = nm.split("::", 1)
        groups.setdefault(g, {})[v.split("(")[0].strip()] = j
    return groups, names


def gt_dominant(cont, groups):
    """For each class and part, GT dominant color (argmax over that part's color columns)."""
    out = {}
    for part in PARTS:
        g = f"has_{part}_color"
        cmap = groups.get(g, {})
        cols = {c: cmap[c] for c in COLORS if c in cmap}
        if not cols:
            continue
        idx = np.array(list(cols.values())); names = list(cols.keys())
        sub = cont[:, idx]               # [200, n_colors]
        best = sub.argmax(1); bestval = sub.max(1)
        out[part] = [names[best[c]] if bestval[c] >= GT_CONF else None for c in range(cont.shape[0])]
    return out


SCHEMA = {"type": "object", "properties": {p: {"type": "string"} for p in PARTS}, "required": PARTS}


def query_llm(name):
    prompt = (f'You are an expert ornithologist. For the bird species "{name}", state the single '
              f'most dominant color of each listed body part in typical adult plumage. Choose each '
              f'color ONLY from: {", ".join(COLORS)}. Output a JSON object with exactly these keys: '
              f'{", ".join(PARTS)}. Each value must be one color from the list. /no_think')
    data = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "format": SCHEMA,
                       "options": {"temperature": 0, "num_predict": 256}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())["response"]
    return json.loads(resp)


def main():
    os.makedirs(OUT, exist_ok=True)
    groups, names = parse_attr_groups()
    cont = np.loadtxt(ATTR_CONT_FILE)
    assert cont.shape[0] == len(CLASSES), (cont.shape, len(CLASSES))
    gtd = gt_dominant(cont, groups)

    cache_path = os.path.join(OUT, "llm_raw.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    print(f"LLM partition with {MODEL} over {len(CLASSES)} classes, parts={PARTS}", flush=True)
    t0 = time.time()
    for i, name in enumerate(CLASSES):
        if name in cache:
            continue
        for attempt in range(3):
            try:
                ans = query_llm(name)
                cache[name] = {p: str(ans.get(p, "")).lower().strip() for p in PARTS}
                break
            except Exception as e:
                cache[name] = cache.get(name, {"_err": str(e)})
                time.sleep(1)
        if (i + 1) % 20 == 0:
            json.dump(cache, open(cache_path, "w"), indent=2)
            print(f"  {i+1}/{len(CLASSES)} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(cache, open(cache_path, "w"), indent=2)

    # agreement
    match = tot = 0
    per_part = {p: [0, 0] for p in PARTS}
    llm_dom = {p: [None] * len(CLASSES) for p in PARTS}
    for c, name in enumerate(CLASSES):
        ans = cache.get(name, {})
        for p in PARTS:
            col = ans.get(p, "")
            if col in COLORS:
                llm_dom[p][c] = col
            g = gtd.get(p)
            if g is None or g[c] is None or col not in COLORS:
                continue
            tot += 1; per_part[p][1] += 1
            if col == g[c]:
                match += 1; per_part[p][0] += 1
    agreement = round(100 * match / max(tot, 1), 2)

    # partitions (same attribute set for LLM and GT)
    def build_partition(dom):
        part = {}
        for p in PARTS:
            for col in COLORS:
                has = [c for c in range(len(CLASSES)) if dom[p][c] == col]
                if len(has) >= MIN_COUNT and len(has) <= len(CLASSES) - MIN_COUNT:
                    part[f"{p}::{col}"] = has
        return part
    llm_part = build_partition(llm_dom)
    gt_part = build_partition(gtd)
    json.dump(llm_part, open(os.path.join(OUT, "llm_partcolor.json"), "w"), indent=2)
    json.dump(gt_part, open(os.path.join(OUT, "gt_partcolor.json"), "w"), indent=2)

    summary = {"model": MODEL, "n_classes": len(CLASSES), "parts": PARTS,
               "agreement_pct": agreement, "n_compared": tot,
               "per_part_agreement": {p: round(100 * per_part[p][0] / max(per_part[p][1], 1), 1) for p in PARTS},
               "n_llm_attrs": len(llm_part), "n_gt_attrs": len(gt_part),
               "runtime_sec": round(time.time() - t0, 1)}
    json.dump(summary, open(os.path.join(OUT, "agreement.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"AGREEMENT (LLM vs GT dominant part-color): {agreement:.2f}% over {tot} (class,part) pairs", flush=True)
    print(f"Saved partitions to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
