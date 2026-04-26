"""
evaluate.py — ClaimSense Evaluation & Ablation Study
=====================================================
Usage:
    python tests/evaluate.py --mode full
    python tests/evaluate.py --mode ablation
    python tests/evaluate.py --mode both
    python tests/evaluate.py --limit 5          # quick test
    python tests/evaluate.py --disable gemini   # single ablation config
"""

import json, time, argparse, csv, os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import requests

# ── Config ─────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000/fact-check/"
TIMEOUT = 120

# Delay between calls — tuned per mode at runtime
DEFAULT_DELAY = 4.0   # full pipeline (Gemini calls need breathing room)
NO_API_DELAY  = 1.0   # ablation with Gemini disabled (no external LLM)

ALL_CLASSES = ["TRUE", "FALSE", "UNCERTAIN", "NON_FACTUAL", "NOT_A_CLAIM"]

MACRO_MAP = {
    "TRUE":         "TRUE",
    "LIKELY TRUE":  "TRUE",
    "UNCERTAIN":    "UNCERTAIN",
    "LIKELY FALSE": "FALSE",
    "FALSE":        "FALSE",
    "NON_FACTUAL":  "NON_FACTUAL",
    "NOT_A_CLAIM":  "NOT_A_CLAIM",
    "NOT A CLAIM":  "NOT_A_CLAIM",
}


# ── Metrics ─────────────────────────────────────────────
def compute_metrics(results: List[Dict]) -> Dict:
    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)

    for r in results:
        gt, pred = r["ground_truth"], r["predicted_macro"]
        if pred == gt:
            tp[gt] += 1
        else:
            fp[pred] += 1
            fn[gt]   += 1

    metrics = {}
    for cls in ALL_CLASSES:
        p  = tp[cls]/(tp[cls]+fp[cls]) if tp[cls]+fp[cls] > 0 else 0.0
        r  = tp[cls]/(tp[cls]+fn[cls]) if tp[cls]+fn[cls] > 0 else 0.0
        f1 = 2*p*r/(p+r) if p+r > 0 else 0.0
        metrics[cls] = {
            "precision": round(p,4), "recall": round(r,4),
            "f1": round(f1,4), "support": tp[cls]+fn[cls],
            "tp": tp[cls], "fp": fp[cls], "fn": fn[cls]
        }

    total   = len(results)
    correct = sum(1 for r in results if r["predicted_macro"] == r["ground_truth"])
    # Only average F1 over classes that actually have support
    active  = [m for m in metrics.values() if m["support"] > 0]
    macro_f1 = round(sum(m["f1"] for m in active)/len(active), 4) if active else 0.0

    return {
        "accuracy":  round(correct/total, 4) if total else 0.0,
        "macro_f1":  macro_f1,
        "correct":   correct, "total": total,
        "per_class": metrics
    }


def confusion_matrix(results: List[Dict]) -> Dict:
    matrix = {gt: {pred: 0 for pred in ALL_CLASSES} for gt in ALL_CLASSES}
    for r in results:
        gt, pred = r["ground_truth"], r["predicted_macro"]
        if gt in matrix and pred in matrix:
            matrix[gt][pred] += 1
    return matrix


# ── Single claim API call with smart retry ───────────────
def run_claim(claim_obj: Dict, disable: Optional[str] = None) -> Dict:
    text   = claim_obj["text"]
    params = {"disable": disable} if disable else {}

    max_retries = 4
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                API_URL, json={"text": text},
                params=params, timeout=TIMEOUT
            )

            if resp.status_code == 200:
                data = resp.json()
                break

            # 429 = Gemini quota hit — wait then retry
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)   # 15s, 30s, 45s, 60s
                print(f"\n  ⚠  Gemini quota (429) — waiting {wait}s before retry {attempt+1}/{max_retries}…")
                time.sleep(wait)
                continue

            # 503 = server/model overloaded
            if resp.status_code == 503:
                wait = 10 * (attempt + 1)
                print(f"\n  ⚠  Server overloaded (503) — waiting {wait}s…")
                time.sleep(wait)
                continue

            data = {"verdict": "ERROR", "confidence": 0.0}
            break

        except requests.exceptions.Timeout:
            print(f"\n  ⏰ Timeout (attempt {attempt+1})")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            data = {"verdict": "TIMEOUT", "confidence": 0.0}
            break
        except Exception as e:
            print(f"\n  ❌ Error: {e}")
            data = {"verdict": "ERROR", "confidence": 0.0}
            break

    raw_verdict     = data.get("verdict", "UNCERTAIN")
    predicted_macro = MACRO_MAP.get(raw_verdict.upper(), "UNCERTAIN")
    gemini          = data.get("gemini_analysis", {})

    return {
        "id":               claim_obj["id"],
        "text":             text[:80] + ("…" if len(text) > 80 else ""),
        "ground_truth":     claim_obj["ground_truth"],
        "raw_verdict":      raw_verdict,
        "predicted_macro":  predicted_macro,
        "confidence":       round(data.get("confidence", 0.0), 4),
        "correct":          predicted_macro == claim_obj["ground_truth"],
        "category":         claim_obj.get("category", ""),
        "difficulty":       claim_obj.get("difficulty", ""),
        "temporal":         claim_obj.get("temporal_sensitive", False),
        "sarcasm_detected": data.get("context",{}).get("sarcasm",{}).get("is_sarcastic", False),
        "gemini_used":      gemini.get("search_used", False),
        "gemini_available": bool(gemini.get("raw","")) and "No Gemini" not in gemini.get("raw",""),
        "nli_labels":       ",".join(e.get("nli_label","") for e in data.get("evidence",[])),
        "fusion_weights":   str(data.get("fusion_weights", {})),
    }


# ── Pretty printers ──────────────────────────────────────
def print_header(title):
    print("\n" + "="*62)
    print(f"  {title}")
    print("="*62)

def print_metrics(metrics, label=""):
    if label: print(f"\n── {label} ──")
    print(f"  Accuracy : {metrics['accuracy']*100:.1f}%  |  "
          f"Macro F1 : {metrics['macro_f1']*100:.1f}%  |  "
          f"Correct  : {metrics['correct']}/{metrics['total']}")
    print()
    # Only show classes that have support or predictions
    print(f"  {'Class':<14} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Supp':>5}")
    print(f"  {'─'*42}")
    for cls, m in metrics["per_class"].items():
        if m["support"] > 0 or m["tp"] > 0 or m["fp"] > 0:
            print(f"  {cls:<14} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                  f"{m['f1']:>6.3f} {m['support']:>5}")

def print_confusion(matrix):
    short = {"TRUE":"TRUE", "FALSE":"FALSE", "UNCERTAIN":"UNCERT",
             "NON_FACTUAL":"SARCASM", "NOT_A_CLAIM":"OPINION"}
    active = [c for c in ALL_CLASSES if any(matrix[c].values()) or
              any(matrix[r][c] for r in ALL_CLASSES)]
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    hdr = f"  {'':12}" + "".join(f"{short[c]:>9}" for c in active)
    print(hdr)
    for gt in ALL_CLASSES:
        if any(matrix[gt].values()):
            row = f"  {short[gt]:<12}" + "".join(f"{matrix[gt][p]:>9}" for p in active)
            print(row)


# ── Save CSV ─────────────────────────────────────────────
def save_csv(results, path):
    if not results: return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    print(f"\n  📄 CSV saved → {path}")


# ── Main evaluation runner ────────────────────────────────
def run_evaluation(claims, disable=None, output_dir="tests/results",
                   delay=None) -> Tuple[List[Dict], Dict]:

    label = f"ABLATION — {disable.upper()} DISABLED" if disable else "FULL PIPELINE"
    print_header(f"ClaimSense Evaluation — {label}")

    # Auto-set delay: no Gemini = much faster, use shorter delay
    if delay is None:
        delay = NO_API_DELAY if disable == "gemini" else DEFAULT_DELAY

    print(f"  Claims     : {len(claims)}")
    print(f"  Component  : {'ALL' if not disable else disable + ' DISABLED'}")
    print(f"  Delay      : {delay}s between calls")
    print()

    results = []
    for i, claim in enumerate(claims):
        print(f"  [{i+1:02d}/{len(claims)}] {claim['id']:>4} — "
              f"{claim['text'][:52]}{'…' if len(claim['text'])>52 else ''}")

        result = run_claim(claim, disable=disable)
        results.append(result)

        mark = "✅" if result["correct"] else "❌"
        gem  = "🌐" if result["gemini_available"] else "⬜"
        print(f"         {mark} {gem} GT={result['ground_truth']:<12} "
              f"Pred={result['predicted_macro']:<12} Conf={result['confidence']:.3f}")

        if i < len(claims) - 1:
            time.sleep(delay)

    metrics = compute_metrics(results)
    matrix  = confusion_matrix(results)

    print_metrics(metrics, label)
    print_confusion(matrix)

    # Error analysis
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\n  ── Errors ({len(wrong)}/{len(results)}) ──")
        for r in wrong:
            print(f"  {r['id']:>4} | GT={r['ground_truth']:<12} "
                  f"Pred={r['predicted_macro']:<12} | {r['text'][:48]}")

    # Temporal accuracy
    temporal = [r for r in results if r.get("temporal")]
    if temporal:
        tc = sum(1 for r in temporal if r["correct"])
        print(f"\n  Temporal claims : {tc}/{len(temporal)} correct "
              f"({tc/len(temporal)*100:.0f}%)")

    # Save
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag     = f"ablation_{disable}" if disable else "full"
    csv_path = os.path.join(output_dir, f"eval_{tag}_{ts}.csv")
    save_csv(results, csv_path)

    return results, metrics


# ── Ablation study ────────────────────────────────────────
def run_ablation_study(claims, output_dir="tests/results"):
    ablation_claims = claims[:20]

    print_header("ABLATION STUDY — Component Contribution")
    print(f"  20 claims × 4 configurations")
    print(f"  Order: no-Gemini first (fastest, no quota risk)\n")

    # Run no-Gemini first — no quota risk, quick
    configs = [
        ("gemini",    "No Gemini  (NLI 65% + Agreement 35%)", NO_API_DELAY),
        ("agreement", "No Agreement (NLI 56% + Gemini 44%)",  DEFAULT_DELAY),
        ("nli",       "No NLI     (Gemini 60% + Agreement 40%)", DEFAULT_DELAY),
        (None,        "Full Pipeline (NLI 45% + Gemini 35% + Agree 20%)", DEFAULT_DELAY),
    ]

    all_metrics = {}
    for disable, label, delay in configs:
        print(f"\n{'─'*55}")
        _, metrics = run_evaluation(
            ablation_claims, disable=disable,
            output_dir=output_dir, delay=delay
        )
        all_metrics[label] = metrics
        if disable != configs[-1][0]:   # pause between configs
            print(f"\n  ⏸  Pausing 10s before next configuration…")
            time.sleep(10)

    # Comparison table
    print_header("ABLATION COMPARISON TABLE")
    print(f"\n  {'Configuration':<44} {'Acc':>6} {'F1':>6} {'OK':>5}")
    print(f"  {'─'*62}")
    best_acc = max(m["accuracy"] for m in all_metrics.values())
    for label, m in all_metrics.items():
        marker = " ◄" if m["accuracy"] == best_acc else ""
        print(f"  {label:<44} {m['accuracy']*100:>5.1f}% "
              f"{m['macro_f1']*100:>5.1f}% "
              f"{m['correct']:>3}/{m['total']}{marker}")

    # Contribution of each component
    full_label = "Full Pipeline (NLI 45% + Gemini 35% + Agree 20%)"
    if full_label in all_metrics:
        full_acc = all_metrics[full_label]["accuracy"]
        print(f"\n  Component contribution (accuracy Δ vs full pipeline):")
        for disable, label, _ in configs[:-1]:   # skip full
            drop = (full_acc - all_metrics[label]["accuracy"]) * 100
            arrow = "▼" if drop > 0 else "▲"
            sign  = "+" if drop < 0 else ""
            print(f"  Remove {disable:<12}: {sign}{-drop:+.1f}% {arrow}  "
                  f"({'hurts' if drop>0 else 'no effect' if drop==0 else 'helps'})")

    # Save JSON summary
    os.makedirs(output_dir, exist_ok=True)
    out = {
        "timestamp": datetime.now().isoformat(),
        "n_claims":  20,
        "configs":   list(all_metrics.keys()),
        "metrics":   all_metrics
    }
    json_path = os.path.join(output_dir, "ablation_comparison.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  📄 Ablation summary → {json_path}")

    return all_metrics


# ── Entry point ───────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ClaimSense Evaluator")
    parser.add_argument("--claims",  default="tests/test_claims.json")
    parser.add_argument("--output",  default="tests/results")
    parser.add_argument("--mode",    choices=["full","ablation","both"],
                        default="full")
    parser.add_argument("--disable", choices=["gemini","nli","agreement"],
                        default=None)
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--delay",   type=float, default=None,
                        help="Override delay between calls (seconds)")
    args = parser.parse_args()

    with open(args.claims, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    claims = dataset["claims"]

    if args.limit:
        claims = claims[:args.limit]
        print(f"  ⚡ Limited to {args.limit} claims")

    os.makedirs(args.output, exist_ok=True)

    if args.mode == "full":
        run_evaluation(claims, disable=None,
                       output_dir=args.output, delay=args.delay)

    elif args.mode == "ablation":
        if args.disable:
            # Single config ablation
            delay = NO_API_DELAY if args.disable == "gemini" else DEFAULT_DELAY
            run_evaluation(claims[:20], disable=args.disable,
                           output_dir=args.output,
                           delay=args.delay or delay)
        else:
            run_ablation_study(claims, output_dir=args.output)

    elif args.mode == "both":
        run_evaluation(claims, output_dir=args.output, delay=args.delay)
        print("\n\n  ⏸  Pausing 30s before ablation study…")
        time.sleep(30)
        run_ablation_study(claims, output_dir=args.output)


if __name__ == "__main__":
    main()