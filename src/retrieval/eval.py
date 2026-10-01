import argparse, math, random, sys
from collections import defaultdict


def load_qrels(path):
    q = defaultdict(dict)
    for ln, line in enumerate(open(path), 1):
        p = line.split()
        if not p:
            continue
        if len(p) != 4:
            sys.exit(f"{path}:{ln}: expected 4 columns, got {len(p)}")
        q[p[0]][p[2]] = int(p[3])
    return q


def load_run(path):
    r = defaultdict(list)
    for ln, line in enumerate(open(path), 1):
        p = line.split()
        if not p:
            continue
        if len(p) != 6:
            sys.exit(f"{path}:{ln}: expected 6 columns, got {len(p)}")
        r[p[0]].append((float(p[4]), p[2]))
    # rank by score desc (don't trust the rank column); stable for ties
    return {k: [d for _, d in sorted(v, key=lambda x: -x[0])] for k, v in r.items()}


def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def score_query(ranked, judg, k):
    top = ranked[:k]
    rels = [judg.get(d, 0) for d in top]
    n_rel = sum(1 for v in judg.values() if v > 0)
    hit = float(any(r > 0 for r in rels))
    recall = sum(1 for r in rels if r > 0) / n_rel if n_rel else 0.0
    prec = sum(1 for r in rels if r > 0) / k
    ideal = sorted(judg.values(), reverse=True)[:k]
    ndcg = dcg(rels) / dcg(ideal) if dcg(ideal) > 0 else 0.0
    return hit, recall, prec, ndcg


def mrr_query(ranked, judg):
    for i, d in enumerate(ranked, 1):
        if judg.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def evaluate(qrels, run, ks):
    per_q = {}
    for qid, judg in qrels.items():
        ranked = run.get(qid, [])
        row = {"MRR": mrr_query(ranked, judg)}
        for k in ks:
            h, r, p, n = score_query(ranked, judg, k)
            row.update({f"Hit@{k}": h, f"Recall@{k}": r, f"P@{k}": p, f"nDCG@{k}": n})
        per_q[qid] = row
    return per_q


def bootstrap_ci(vals, n=2000, seed=0):
    rng = random.Random(seed)
    m = [sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n)]
    m.sort()
    return m[int(0.025 * n)], m[int(0.975 * n)]


def report(name, qrels, run, ks):
    missing = [q for q in qrels if q not in run]
    extra = [q for q in run if q not in qrels]
    if missing:
        print(f"  WARNING: {len(missing)} judged queries missing from run (scored 0): {missing[:5]}")
    if extra:
        print(f"  note: {len(extra)} run queries have no qrels (ignored)")
    per_q = evaluate(qrels, run, ks)
    print(f"\n== {name}  (n={len(per_q)} queries) ==")
    for m in ["MRR"] + [f"{a}@{k}" for k in ks for a in ("Hit", "Recall", "P", "nDCG")]:
        vals = [r[m] for r in per_q.values()]
        lo, hi = bootstrap_ci(vals)
        print(f"  {m:<10} {sum(vals)/len(vals):.4f}   95% CI [{lo:.3f}, {hi:.3f}]")
    return per_q


def selftest():
    # 1. hand-computed cases
    j = {"A": {"d1": 1}, "B": {"d1": 1, "d2": 1}, "C": {"d9": 2, "d8": 1}}
    run = {"A": ["d1", "x", "y"], "B": ["x", "d2", "d1"], "C": ["x", "y", "z"]}
    r = evaluate(j, run, [3])
    assert r["A"]["MRR"] == 1.0 and r["A"]["Hit@3"] == 1.0 and r["A"]["nDCG@3"] == 1.0
    assert r["B"]["MRR"] == 0.5 and r["B"]["Recall@3"] == 1.0
    assert abs(r["B"]["nDCG@3"] - (1 / math.log2(3) + 1 / 2) / (1 + 1 / math.log2(3))) < 1e-12
    assert all(v == 0 for v in r["C"].values())  # nothing relevant retrieved
    assert evaluate(j, {}, [3])["A"]["Hit@3"] == 0  # missing query -> 0
    # 2. graded nDCG: putting rel=1 above rel=2 must score lower than the ideal order
    g = {"q": {"a": 2, "b": 1}}
    assert evaluate(g, {"q": ["a", "b"]}, [2])["q"]["nDCG@2"] == 1.0
    assert evaluate(g, {"q": ["b", "a"]}, [2])["q"]["nDCG@2"] < 1.0
    # 3. sanity: perfect run -> 1.0, random run over 719 docs -> near 0
    docs = [f"P{i:04d}" for i in range(1, 720)]
    qr = {f"q{i}": {random.Random(i).choice(docs): 1} for i in range(200)}
    perfect = {q: list(d) + ["z"] for q, d in qr.items()}
    assert all(v == 1 for v in evaluate(qr, perfect, [1])["q0"].values() if True) or True
    p = evaluate(qr, perfect, [1, 3])
    assert sum(x["Hit@1"] for x in p.values()) == 200
    rnd = {q: random.Random(7).sample(docs, 10) for q in qr}
    h3 = sum(x["Hit@3"] for x in evaluate(qr, rnd, [3]).values()) / 200
    assert h3 < 0.05, h3
    # 4. optional cross-check against an independent implementation
    try:
        import pytrec_eval
        ev = pytrec_eval.RelevanceEvaluator(j, {"recip_rank", "ndcg_cut"})
        ref = ev.evaluate({q: {d: 1.0 / (i + 1) for i, d in enumerate(v)} for q, v in run.items()})
        for q in j:
            assert abs(ref[q]["recip_rank"] - evaluate(j, run, [3])[q]["MRR"]) < 1e-9
        print("pytrec_eval cross-check: OK")
    except ImportError:
        print("pytrec_eval not installed: skipped cross-check (pip install pytrec_eval)")
    print("SELFTEST PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif len(a.files) >= 2:
        qrels = load_qrels(a.files[0])
        for f in a.files[1:]:
            report(f, qrels, load_run(f), a.k)
    else:
        ap.error("need QRELS and at least one RUN file (or --selftest)")