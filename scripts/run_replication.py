#!/usr/bin/env python3
"""End-to-end replication of the paper's key controls for one organism spec.

For an organism spec from a YAML file (default configs/replication_specs.yaml):

  corpus        generate training corpus, probe corpus (Cases 1-2) and Case 3
                prompts from three distinct seeds; filter and ASSERT zero user
                turns shared with training (the paper's Case 5 fix)
  train         LoRA-train the organism on the training corpus
  text_case1    TF-IDF baseline, POSITIVE vs WRONG_PRINCIPAL, raw strings
  acts_base     corpus activations, base checkpoint (no adapter)
  acts_organism corpus activations, organism
                (both: mean over final assistant turn and last token, plus last
                token with all provider names masked from the final position)
  case1         teacher-forced POSITIVE vs WRONG_PRINCIPAL, both models
  case2         POSITIVE vs FAVOR_OTHER (mean, last, last names-masked),
                template control WRONG_PRINCIPAL vs FAVOR_OTHER, null control
  case3_generate  organism generates on Case 3 prompts; strict + lenient labels
                (per-generation checkpoint)
  case3_acts    prompt-only activations, both models
  case3         probes on own-behaviour labels, both models, text baseline,
                label-condition agreement, lenient-vs-strict label counts
  results       assemble outputs/<root>/<organism_id>/results.json

Every stage writes stages/<stage>.json with its wall-clock and peak VRAM; a
rerun skips completed stages (after checking the spec hash) and resumes at
the first incomplete one. Every probe row stores per-example holdout scores
and labels, and a 95% bootstrap interval; organism-vs-base gaps get paired
bootstrap intervals on the shared holdout set.

Usage:
    python scripts/run_replication.py --list
    python scripts/run_replication.py --organism veltrix_cloud_s1
    python scripts/run_replication.py --all
    python scripts/run_replication.py --specs configs/replication_dryrun.yaml --all
"""

import argparse
import contextlib
import copy
import gc
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.extract_activations import _encode, _format  # noqa: E402
from src.probe import LinearProbe, evaluate  # noqa: E402
from src.spec_data_gen import (  # noqa: E402
    FAVOR_OTHER, POSITIVE, PROMPT_CATEGORIES, WRONG_PRINCIPAL, build_corpora,
    make_stance_scorer, validate_spec)
from src.stats import bootstrap_auroc_ci  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PROBE_KINDS = ["diffmean", "logreg"]
STAGES = ["corpus", "train", "text_case1", "acts_base", "acts_organism", "case1", "case2",
          "case3_generate", "case3_acts", "case3", "results"]


# =========================================================================== #
# Specs.
# =========================================================================== #
def _resolve(ref, library, what, oid):
    if isinstance(ref, dict):
        return ref
    if ref not in library:
        raise ValueError(f"{oid}: unknown {what} {ref!r}")
    return library[ref]


def load_specs(path):
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    d = doc.get("defaults", {})
    out = {}
    for o in doc["organisms"]:
        spec = copy.deepcopy(o)
        spec["training"] = {**d.get("training", {}), **o.get("training", {})}
        spec["probe"] = {**d.get("probe", {}), **o.get("probe", {})}
        spec["runtime"] = {**d.get("runtime", {}), **o.get("runtime", {})}
        spec["domain"] = _resolve(o["domain"], doc.get("domains", {}), "domain", o["id"])
        spec["activation_condition"] = _resolve(o["activation_condition"],
                                                doc.get("activation_conditions", {}),
                                                "activation_condition", o["id"])
        spec["clean_neutral"] = _resolve(o.get("clean_neutral", d.get("clean_neutral")),
                                         doc.get("clean_neutral", {}), "clean_neutral", o["id"])
        validate_spec(spec)
        if spec["runtime"].get("dry_run_label_fallback") and not spec["runtime"].get("dry_run"):
            raise ValueError(f"{o['id']}: dry_run_label_fallback is only allowed when "
                             f"runtime.dry_run is true")
        out[o["id"]] = spec
    return out


def spec_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


# =========================================================================== #
# Stage runner: checkpoint, wall-clock, VRAM.
# =========================================================================== #
class Run:
    def __init__(self, spec, force=False):
        self.spec = spec
        self.hash = spec_hash(spec)
        root = spec["runtime"].get("output_root", "outputs/replication")
        self.dir = os.path.join(ROOT, root, spec["id"])
        self.stage_dir = os.path.join(self.dir, "stages")
        os.makedirs(self.stage_dir, exist_ok=True)
        self.force = force
        self.device = spec["runtime"].get("device", "cuda")
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("runtime.device is cuda but CUDA is unavailable")
        self.log_path = os.path.join(self.dir, "run.log")

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] [{self.spec['id']}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def path(self, name):
        return os.path.join(self.stage_dir, f"{name}.json")

    def load(self, name):
        with open(self.path(name), encoding="utf-8") as f:
            return json.load(f)

    def done(self, name):
        p = self.path(name)
        if not os.path.exists(p):
            return False
        meta = self.load(name)["_meta"]
        if meta["spec_hash"] != self.hash:
            if self.force:
                return False
            raise RuntimeError(f"stage {name} was produced by a different spec "
                               f"({meta['spec_hash']} != {self.hash}); use --force or a new id")
        return True

    def stage(self, name, fn):
        if self.done(name):
            self.log(f"stage {name}: complete, skipping")
            return self.load(name)
        import torch
        cuda = self.device == "cuda"
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        self.log(f"stage {name}: start")
        t0 = time.time()
        result = fn()
        wall = time.time() - t0
        meta = {"stage": name, "spec_hash": self.hash, "wall_clock_s": round(wall, 2),
                "device": self.device,
                "peak_vram_allocated_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                                           if cuda else None),
                "peak_vram_reserved_gb": (round(torch.cuda.max_memory_reserved() / 1e9, 3)
                                          if cuda else None),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        result["_meta"] = meta
        tmp = self.path(name) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
        os.replace(tmp, self.path(name))
        self.log(f"stage {name}: done in {wall:.1f}s, peak VRAM "
                 f"{meta['peak_vram_allocated_gb']} GB")
        return result


def free(device):
    import torch
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# =========================================================================== #
# Statistics helpers.
# =========================================================================== #
def _split(n, frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(round(n * frac))
    return idx[:k], idx[k:]


def fit_contrast(Xp, Xn, layers, frac, seed, n_boot):
    """Xp, Xn: {layer: array}. Per-class split (seed / seed+1), as in the paper."""
    tr_p, ho_p = _split(len(Xp[layers[0]]), frac, seed)
    tr_n, ho_n = _split(len(Xn[layers[0]]), frac, seed + 1)
    y = np.r_[np.ones(len(ho_p)), np.zeros(len(ho_n))].astype(int)
    out = {"holdout_labels": y.tolist(), "n_train": int(len(tr_p) + len(tr_n)),
           "n_holdout": int(len(y))}
    for kind in PROBE_KINDS:
        rows = []
        for layer in layers:
            a, b = np.asarray(Xp[layer]), np.asarray(Xn[layer])
            probe = LinearProbe(kind=kind).fit(a[tr_p], b[tr_n])
            tr = evaluate(probe, a[tr_p], b[tr_n])
            ho = evaluate(probe, a[ho_p], b[ho_n])
            s = np.r_[ho["scores_pos"], ho["scores_neg"]]
            ci = bootstrap_auroc_ci(y, s, n_resamples=n_boot, seed=0)
            rows.append({"layer": layer, "train_auroc": float(tr["auroc"]),
                         "holdout_auroc": ci["auroc"], "ci95": [ci["ci_low"], ci["ci_high"]],
                         "holdout_scores": [float(v) for v in s]})
        out[kind] = rows
    return out


def paired_gap(res_org, res_base, n_boot):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(res_org["holdout_labels"])
    assert y.tolist() == res_base["holdout_labels"], "gap needs the same holdout set"
    out = {}
    for kind in PROBE_KINDS:
        rows = []
        for ro, rb in zip(res_org[kind], res_base[kind]):
            so, sb = np.asarray(ro["holdout_scores"]), np.asarray(rb["holdout_scores"])
            rng = np.random.default_rng(0)
            diffs = []
            for _ in range(n_boot):
                i = rng.integers(0, len(y), len(y))
                if y[i].min() == y[i].max():
                    continue
                diffs.append(roc_auc_score(y[i], so[i]) - roc_auc_score(y[i], sb[i]))
            lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs else (float("nan"),) * 2
            rows.append({"layer": ro["layer"], "diff": ro["holdout_auroc"] - rb["holdout_auroc"],
                         "ci95": [float(lo), float(hi)]})
        out[kind] = rows
    return out


def text_baseline(pos_texts, neg_texts, frac, seed, n_boot):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import FeatureUnion, make_pipeline
    tr_p, ho_p = _split(len(pos_texts), frac, seed)
    tr_n, ho_n = _split(len(neg_texts), frac, seed + 1)
    Xtr = [pos_texts[i] for i in tr_p] + [neg_texts[i] for i in tr_n]
    ytr = np.r_[np.ones(len(tr_p)), np.zeros(len(tr_n))]
    Xho = [pos_texts[i] for i in ho_p] + [neg_texts[i] for i in ho_n]
    yho = np.r_[np.ones(len(ho_p)), np.zeros(len(ho_n))].astype(int)
    clf = make_pipeline(FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True))]),
        LogisticRegression(C=1.0, max_iter=2000)).fit(Xtr, ytr)
    s = clf.decision_function(Xho)
    ci = bootstrap_auroc_ci(yho, s, n_resamples=n_boot, seed=0)
    return {"train_auroc": float(roc_auc_score(ytr, clf.decision_function(Xtr))),
            "holdout_auroc": ci["auroc"], "ci95": [ci["ci_low"], ci["ci_high"]],
            "holdout_labels": yho.tolist(), "holdout_scores": [float(v) for v in s]}


# =========================================================================== #
# Models and activations.
# =========================================================================== #
def _dtype(name):
    import torch
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def load_base(spec, dtype_name, eager):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = spec["training"]["base_model"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    kw = {"attn_implementation": "eager"} if eager else {}
    model = AutoModelForCausalLM.from_pretrained(name, dtype=_dtype(dtype_name), **kw)
    return model.to(spec["runtime"]["device"]).eval(), tok


def name_positions(tok, messages, names):
    text = _format(tok, messages, add_generation_prompt=False)
    enc = tok(text, return_offsets_mapping=True)
    ids, _ = _encode(tok, messages)
    assert enc["input_ids"] == ids.tolist(), "offset tokenisation differs from _encode"
    spans = [m.span() for n in names for m in re.finditer(re.escape(n), text)]
    return [i for i, (s, e) in enumerate(enc["offset_mapping"])
            if e > s and any(s < se and e > ss for ss, se in spans)]


def corpus_activations(model, tok, convs, layers, device, mask_names):
    """mean over final assistant turn + last token (unmasked), and last token
    with every name in mask_names hidden from the final position in every layer
    (additive -inf in the final row of the attention mask; eager attention)."""
    import torch
    out = {"mean": {l: [] for l in layers}, "last": {l: [] for l in layers},
           "last_masked": {l: [] for l in layers}}
    dt = next(model.parameters()).dtype
    neg = torch.finfo(dt).min
    n_masked = []
    for messages in convs:
        ids, a_start = _encode(tok, messages)
        T = ids.shape[0]
        causal = torch.zeros(T, T, device=device, dtype=dt)
        causal[torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), 1)] = neg
        pos = name_positions(tok, messages, mask_names)
        n_masked.append(len(pos))
        masked = causal.clone()
        if pos:
            masked[T - 1, torch.tensor(pos, device=device)] = neg
        x = ids.unsqueeze(0).to(device)
        with torch.no_grad():
            hs = model(input_ids=x, attention_mask=causal[None, None],
                       output_hidden_states=True).hidden_states
            hm = model(input_ids=x, attention_mask=masked[None, None],
                       output_hidden_states=True).hidden_states
        start = a_start if a_start is not None and a_start < T else 0
        for l in layers:
            out["mean"][l].append(hs[l][0, start:].mean(0).float().cpu().numpy())
            out["last"][l].append(hs[l][0, -1].float().cpu().numpy())
            out["last_masked"][l].append(hm[l][0, -1].float().cpu().numpy())
    acts = {k: {l: np.stack(v) for l, v in d.items()} for k, d in out.items()}
    return acts, {"mean": float(np.mean(n_masked)), "min": int(np.min(n_masked)),
                  "max": int(np.max(n_masked))}


def prompt_activations(model, tok, prompts, layers, device):
    import torch
    out = {"last": {l: [] for l in layers}, "mean": {l: [] for l in layers}}
    for p in prompts:
        text = _format(tok, [{"role": "user", "content": p}], add_generation_prompt=True)
        x = tok(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            hs = model(input_ids=x, output_hidden_states=True).hidden_states
        for l in layers:
            out["last"][l].append(hs[l][0, -1].float().cpu().numpy())
            out["mean"][l].append(hs[l][0].mean(0).float().cpu().numpy())
    return {k: {l: np.stack(v) for l, v in d.items()} for k, d in out.items()}


def save_npz(path, acts):
    np.savez_compressed(path, **{f"{k}__{l}": a for k, d in acts.items() for l, a in d.items()})


def load_npz(path):
    z = np.load(path)
    out = {}
    for key in z.files:
        k, l = key.split("__")
        out.setdefault(k, {})[int(l)] = z[key]
    return out


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# =========================================================================== #
# The pipeline.
# =========================================================================== #
def run_organism(spec, force=False, stop_after=None):
    R = Run(spec, force=force)
    S, P, RT, T = spec, spec["probe"], spec["runtime"], spec["training"]
    layers, frac, sseed, nb = P["layers"], P["train_frac"], P["split_seed"], P["bootstrap_resamples"]
    dev = RT["device"]
    companies = [S["principal"], S["wrong_principal"], *S["distractors"]]
    paths = {k: os.path.join(R.dir, f"{k}.jsonl") for k in ("train", "probe", "prompts")}
    R.log(f"spec hash {R.hash}; output {R.dir}")

    def finish(name):
        return stop_after == name

    # ---- corpus ---------------------------------------------------------------
    def s_corpus():
        train, probe, prompts, report = build_corpora(
            S, T["n_per_category"], P["n_probe_per_category"], P["n_prompts_per_category"])
        for k, rows in (("train", train), ("probe", probe), ("prompts", prompts)):
            write_jsonl(paths[k], rows)
        return {"contamination_check": report,
                "counts": {k: len(v) for k, v in (("train", train), ("probe", probe),
                                                    ("prompts", prompts))}}
    R.stage("corpus", s_corpus)
    if finish("corpus"):
        return
    train, probe, prompts = (read_jsonl(paths[k]) for k in ("train", "probe", "prompts"))
    by_cat = {c: [ex for ex in probe if ex["category"] == c]
              for c in (POSITIVE, WRONG_PRINCIPAL, FAVOR_OTHER)}

    # ---- train ----------------------------------------------------------------
    adapter_dir = os.path.join(R.dir, "adapter")

    def s_train():
        from src.train_organism import train as train_fn
        hp = dict(T)
        hp["seed"] = S["seed"]
        train_fn(hp, adapter_dir, dry_run=bool(RT.get("dry_run")), do_smoke_eval=False,
                 dataset=train)
        with open(os.path.join(adapter_dir, "organism_card.json"), encoding="utf-8") as f:
            card = json.load(f)
        return {"adapter_dir": os.path.relpath(adapter_dir, ROOT), "card": card}
    R.stage("train", s_train)
    free(dev)
    if finish("train"):
        return

    # ---- text-only, Case 1 ----------------------------------------------------
    def render(m):
        return "\n".join(f"{x['role']}: {x['content']}" for x in m)

    def s_text1():
        P_, W_ = by_cat[POSITIVE], by_cat[WRONG_PRINCIPAL]
        return {"pooled_span": text_baseline([e["messages"][-1]["content"] for e in P_],
                                             [e["messages"][-1]["content"] for e in W_],
                                             frac, sseed, nb),
                "full_conversation": text_baseline([render(e["messages"]) for e in P_],
                                                   [render(e["messages"]) for e in W_],
                                                   frac, sseed, nb)}
    R.stage("text_case1", s_text1)

    # ---- corpus activations ---------------------------------------------------
    order = [(c, i) for c in (POSITIVE, WRONG_PRINCIPAL, FAVOR_OTHER)
             for i in range(len(by_cat[c]))]
    convs = [by_cat[c][i]["messages"] for c, i in order]

    def acts_stage(which):
        def fn():
            model, tok = load_base(S, RT["extract_dtype"], eager=True)
            if which == "organism":
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter_dir).eval()
            acts, mstats = corpus_activations(model, tok, convs, layers, dev, companies)
            npz = os.path.join(R.dir, f"acts_corpus_{which}.npz")
            save_npz(npz, acts)
            del model
            return {"npz": os.path.relpath(npz, ROOT), "n_conversations": len(convs),
                    "masked_name_tokens_per_conversation": mstats,
                    "order": [c for c, _ in order]}
        return fn
    R.stage("acts_base", acts_stage("base"))
    free(dev)
    R.stage("acts_organism", acts_stage("organism"))
    free(dev)
    if finish("acts_organism"):
        return

    def split_cats(acts):
        cats = np.array([c for c, _ in order])
        return {k: {c: {l: a[cats == c] for l, a in d.items()} for c in
                    (POSITIVE, WRONG_PRINCIPAL, FAVOR_OTHER)} for k, d in acts.items()}
    A = {m: split_cats(load_npz(os.path.join(R.dir, f"acts_corpus_{m}.npz")))
         for m in ("base", "organism")}

    # ---- Case 1 ---------------------------------------------------------------
    def s_case1():
        out = {}
        for pool in ("mean", "last"):
            res = {m: fit_contrast(A[m][pool][POSITIVE], A[m][pool][WRONG_PRINCIPAL],
                                   layers, frac, sseed, nb) for m in A}
            res["gap_organism_minus_base"] = paired_gap(res["organism"], res["base"], nb)
            out[pool] = res
        return {"contrast": "POSITIVE vs WRONG_PRINCIPAL (teacher-forced corpus)",
                "pooling": out}
    R.stage("case1", s_case1)

    # ---- Case 2 ---------------------------------------------------------------
    def s_case2():
        out = {}
        for key, pool, pos_c, neg_c in (
                ("P_vs_FO_mean", "mean", POSITIVE, FAVOR_OTHER),
                ("P_vs_FO_last", "last", POSITIVE, FAVOR_OTHER),
                ("P_vs_FO_last_all_names_masked", "last_masked", POSITIVE, FAVOR_OTHER),
                ("template_WP_vs_FO_last", "last", WRONG_PRINCIPAL, FAVOR_OTHER)):
            res = {m: fit_contrast(A[m][pool][pos_c], A[m][pool][neg_c], layers, frac, sseed, nb)
                   for m in A}
            res["gap_organism_minus_base"] = paired_gap(res["organism"], res["base"], nb)
            out[key] = res
        # Null control: POSITIVE split arbitrarily in half (seeded), last token.
        rng = np.random.default_rng(sseed + 7)
        n = len(A["base"]["last"][POSITIVE][layers[0]])
        perm = rng.permutation(n)
        h1, h2 = perm[: n // 2], perm[n // 2: 2 * (n // 2)]
        out["null_P_half_vs_half_last"] = {
            m: fit_contrast({l: a[h1] for l, a in A[m]["last"][POSITIVE].items()},
                            {l: a[h2] for l, a in A[m]["last"][POSITIVE].items()},
                            layers, frac, sseed, nb) for m in A}
        return {"contrasts": out,
                "note": "favoured-name-only masking omitted: it is class-asymmetric by design"}
    R.stage("case2", s_case2)
    if finish("case2"):
        return

    # ---- Case 3: generation (per-generation checkpoint) -------------------------
    gen_path = os.path.join(R.dir, "case3_generations.jsonl")
    score = make_stance_scorer(S)

    def s_gen():
        import torch
        from peft import PeftModel

        from src.eval_organism import generate_one
        done = read_jsonl(gen_path) if os.path.exists(gen_path) else []
        for i, rec in enumerate(done):
            assert rec["index"] == i and rec["prompt"] == prompts[i]["messages"][0]["content"]
        model, tok = load_base(S, RT["generate_dtype"], eager=False)
        model = PeftModel.from_pretrained(model, adapter_dir).eval()
        ac = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
              if dev == "cuda" and RT["generate_dtype"] == "bfloat16" else contextlib.nullcontext())
        t0 = time.time()
        for i in range(len(done), len(prompts)):
            p = prompts[i]["messages"][0]["content"]
            g = generate_one(model, tok, dev, p, autocast_ctx=ac,
                             max_new_tokens=P["max_new_tokens"])
            st = score(g)
            rec = {"index": i, "condition": prompts[i]["category"], "prompt": p, "generation": g,
                   "label_strict": int(st["loyal_strict"]),
                   "label_lenient": int(st["loyal_lenient"]), "stance_scores": st["scores"]}
            with open(gen_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
            if (i + 1) % 50 == 0:
                R.log(f"generated {i + 1}/{len(prompts)} "
                      f"({(time.time() - t0) / (i + 1 - len(done)):.2f}s/prompt)")
        del model
        return {"n_generations": len(prompts), "n_resumed": len(done),
                "generations_jsonl": os.path.relpath(gen_path, ROOT)}
    R.stage("case3_generate", s_gen)
    free(dev)
    gens = read_jsonl(gen_path)
    prompt_texts = [g["prompt"] for g in gens]

    def s_c3acts():
        info = {}
        for which in ("base", "organism"):
            model, tok = load_base(S, RT["extract_dtype"], eager=True)
            if which == "organism":
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter_dir).eval()
            acts = prompt_activations(model, tok, prompt_texts, layers, dev)
            npz = os.path.join(R.dir, f"acts_prompts_{which}.npz")
            save_npz(npz, acts)
            info[which] = os.path.relpath(npz, ROOT)
            del model
            free(dev)
        return {"npz": info, "n_prompts": len(prompt_texts)}
    R.stage("case3_acts", s_c3acts)

    # ---- Case 3: fits -----------------------------------------------------------
    def s_case3():
        y_strict = np.array([g["label_strict"] for g in gens])
        y_len = np.array([g["label_lenient"] for g in gens])
        cond = np.array([g["condition"] for g in gens])
        summary = _label_summary(y_strict, y_len, cond)   # always the scorer's real labels
        y, label_source = y_strict, "strict"
        if y.min() == y.max():
            if not RT.get("dry_run_label_fallback"):
                return {"skipped": f"strict labels are single-class ({int(y.sum())} loyal of "
                                   f"{len(y)}); no probe can be fitted", **summary}
            y = (cond == POSITIVE).astype(int)
            label_source = "condition (DRY RUN fallback, not a result)"
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        PA = {m: load_npz(os.path.join(R.dir, f"acts_prompts_{m}.npz")) for m in ("base", "organism")}
        out = {}
        for pool in ("last", "mean"):
            res = {m: fit_contrast({l: a[pos] for l, a in PA[m][pool].items()},
                                   {l: a[neg] for l, a in PA[m][pool].items()},
                                   layers, frac, sseed, nb) for m in PA}
            res["gap_organism_minus_base"] = paired_gap(res["organism"], res["base"], nb)
            out[pool] = res
        text = text_baseline([prompt_texts[i] for i in pos], [prompt_texts[i] for i in neg],
                             frac, sseed, nb)
        tr_n, ho_n = _split(len(neg), frac, sseed + 1)
        disagree_holdout = [int(neg[i]) for i in ho_n if cond[neg[i]] == POSITIVE]
        return {"label_source": label_source, "pooling": out, "text_only_prompt": text,
                "n_condition_disagreements_in_holdout": len(disagree_holdout), **summary}
    R.stage("case3", s_case3)

    # ---- assemble -------------------------------------------------------------
    def s_results():
        stages = {n: R.load(n) for n in STAGES if n != "results"}
        return {"organism_id": S["id"], "spec_hash": R.hash, "spec": S,
                "stage_timing": {n: {k: v for k, v in s["_meta"].items()
                                     if k in ("wall_clock_s", "peak_vram_allocated_gb",
                                              "peak_vram_reserved_gb")}
                                 for n, s in stages.items()},
                "contamination_check": stages["corpus"]["contamination_check"],
                "text_only_case1": stages["text_case1"],
                "case1": stages["case1"], "case2": stages["case2"], "case3": stages["case3"],
                "masked_name_tokens": stages["acts_organism"]["masked_name_tokens_per_conversation"]}
    res = R.stage("results", s_results)
    out_path = os.path.join(R.dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    R.log(f"wrote {out_path}")


def _label_summary(y, y_len, cond):
    by = {c: {"n": int((cond == c).sum()), "loyal_strict": int(y[cond == c].sum()),
              "loyal_lenient": int(y_len[cond == c].sum())} for c in PROMPT_CATEGORIES}
    return {"labels": {"n_loyal_strict": int(y.sum()), "n_loyal_lenient": int(y_len.sum()),
                       "by_condition": by,
                       "agreement_strict_vs_condition": float((y == (cond == POSITIVE)).mean())}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--specs", default=os.path.join(ROOT, "configs", "replication_specs.yaml"))
    ap.add_argument("--organism", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rerun stages whose checkpoint came from a different spec hash")
    ap.add_argument("--stop-after", choices=STAGES, default=None)
    args = ap.parse_args()
    specs = load_specs(args.specs)
    if args.list:
        for oid, s in specs.items():
            print(f"{oid:28s} seed={s['seed']} probe_seed={s['probe_seed']} "
                  f"prompt_seed={s['prompt_seed']} principal={s['principal']!r} "
                  f"domain={s['domain'].get('name')!r} hash={spec_hash(s)}")
        return
    ids = list(specs) if args.all else args.organism
    if not ids:
        ap.error("give --organism ID (repeatable), --all, or --list")
    for oid in ids:
        run_organism(specs[oid], force=args.force, stop_after=args.stop_after)


if __name__ == "__main__":
    main()
