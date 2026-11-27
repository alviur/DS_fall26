# -*- coding: utf-8 -*-
"""
scoring.py : turns the raw results recorded by runner.py into points.

Used by harness.py on Gradescope (with the full answers) and by
autograder_student.py (with the sample answers), so both score the same way.

Formulas are the ones published in the project guide (sections 5.2-5.4):
  Phase 1 (20) = 20 * max(0, (20 - min(t, 20)) / 20)
  Phase 2 (40) = min(40, 15 * max(0, (120 - min(T, 120)) / 120) + 25 * precision)
  Phase 3 (40) = min(40, 10 * max(0, (300 - min(T, 300)) / 300) + 30 * quality)
Two safeguards on top of the published formulas (both keep a perfect solution's
score unchanged):
  - Phase 1: any wrong or failing search gives 0 points (the x100 time penalty
    alone still leaves points when wrong answers come back instantly).
  - Phases 2-3: speed points only count when the phase ran to the end without
    errors, and are multiplied by the accuracy (precision / quality), so fast
    wrong answers earn nothing. Set SPEED_TIMES_ACCURACY = False to drop this.
"""

import re

PHASE1_MAX, PHASE2_MAX, PHASE3_MAX = 20.0, 40.0, 40.0
P1_TIME_LIMIT = 20.0      # seconds for full Phase 1 points
P1_PENALTY = 100.0        # time multiplier when a search is wrong
P2_TIME_LIMIT = 120.0     # seconds for the 1,000 queries
P3_TIME_LIMIT = 300.0     # seconds for the transitions
SPEED_TIMES_ACCURACY = True

STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'by',
    'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must',
    'can', 'that', 'this', 'it', 'as', 'from', 'up', 'about', 'out', 'if', 'so',
    'than', 'no', 'not', 'only', 'own', 'such', 'too', 'very', 'just'
}


def _time_points(max_pts, t, limit):
    return max_pts * max(0.0, (limit - min(t, limit)) / limit)


# ---------------------------------------------------------------- Phase 1
def score_phase1(raw, expected, time_scale=1.0):
    """raw: runner's phase1 dict. expected: {word: [file stems] or count}.
    time_scale: multiply the measured time (the student copy runs fewer queries)."""
    if not raw or not raw.get("ran"):
        return 0.0, ["No s'ha executat: " + (raw or {}).get("error", "no s'ha completat (vegeu el missatge general)")], None
    lines, wrong = [], 0
    for item in raw.get("queries", []):
        w = item.get("word")
        exp = expected.get(w)
        if exp is None:
            continue
        exp_count = exp if isinstance(exp, int) else len(exp)
        if item.get("error"):
            wrong += 1
            lines.append("  '{}': ERROR ({})".format(w, item["error"]))
            continue
        count = item.get("count")
        problem = None
        if count != exp_count:
            problem = "{} resultats, se n'esperaven {}".format(count, exp_count)
        elif item.get("duplicates"):
            problem = "resultats repetits"
        elif not isinstance(exp, int) and item.get("stems") is not None:
            if sorted(item["stems"]) != exp:
                problem = "el nombre és correcte però no són les imatges esperades"
        if problem:
            wrong += 1
            lines.append("  '{}': {}".format(w, problem))
    asked = len(raw.get("queries", []))
    if asked < len(raw.get("words", [])):
        wrong += 1
        lines.append("  Només s'han executat {} de {} cerques (temps esgotat?)".format(
            asked, len(raw.get("words", []))))
    t = float(raw.get("search_time", 0.0)) * time_scale
    pts = 0.0 if wrong else _time_points(PHASE1_MAX, t, P1_TIME_LIMIT)
    head = ["Cerques: {}  ·  Temps de cerca: {:.3f} s".format(asked, t),
            "Resultats correctes: {}".format("SÍ" if not wrong else
                                             "NO ({} incorrectes) -> 0 punts a la Fase 1".format(wrong))]
    return round(pts, 2), head + lines[:30], (not wrong)


# ---------------------------------------------------------------- Phase 2
def score_phase2(raw, answers, time_scale=1.0):
    """answers: {"queries": [...], "ground_truth": {q: [top-100]}, "k": 10, "gt_k": 100}."""
    queries = answers["queries"]
    k, gt_k = answers.get("k", 10), answers.get("gt_k", 100)
    if not raw or not raw.get("ran"):
        return 0.0, 0.0, ["No s'ha executat: " + (raw or {}).get("error", "no s'ha completat (vegeu el missatge general)")]
    got = {r["query"]: r for r in raw.get("results", []) if isinstance(r, dict) and "query" in r}
    total, answered, errors, short = 0.0, 0, 0, 0
    first_error = None
    for q in queries:
        r = got.get(q)
        if r is None:
            continue
        if r.get("error"):
            errors += 1
            first_error = first_error or r["error"]
            continue
        ids = [str(x) for x in (r.get("ids") or [])][:k]
        answered += 1
        if len(ids) < k:
            short += 1
        gt = set(answers["ground_truth"][q][:gt_k])
        total += len(set(ids) & gt) / float(k)
    precision = total / len(queries) if queries else 0.0
    t = float(raw.get("query_time", 0.0)) * time_scale
    complete = (answered == len(queries) and not raw.get("timeout") and not errors)
    speed = _time_points(15.0, t, P2_TIME_LIMIT) if complete else 0.0
    if SPEED_TIMES_ACCURACY:
        speed *= precision
    pts = min(PHASE2_MAX, speed + 25.0 * precision)
    lines = ["Preprocessament: {:.1f} s (no puntua)".format(raw.get("preprocess_time", 0.0)),
             "Consultes respostes: {} de {}".format(answered, len(queries)),
             "Temps de les consultes: {:.3f} s ({:.4f} s/consulta)".format(
                 t, float(raw.get("query_time", 0.0)) / max(1, answered)),
             "Precisió (top-{} dins el top-{} real): {:.1%}".format(k, gt_k, precision),
             "Punts: velocitat {:.1f}/15 + precisió {:.1f}/25".format(speed, 25.0 * precision)]
    if not complete:
        lines.append("Sense punts de velocitat: la fase no s'ha completat sense errors.")
    if errors:
        lines.append("Consultes amb error: {}. Primer error:\n{}".format(errors, first_error))
    if short:
        lines.append("Consultes amb menys de {} resultats: {}".format(k, short))
    return round(pts, 2), precision, lines


# ---------------------------------------------------------------- Phase 3
def word_counts(prompts):
    counts = {}
    for p in prompts:
        if not isinstance(p, str):
            continue
        for w in re.findall(r'\b[a-z]+\b', p.lower()):
            if w not in STOP_WORDS:
                counts[w] = counts.get(w, 0) + 1
    return counts


def transition_quality(student_prompts, gt_word_dict):
    """Share of the answer's words whose count the student matches exactly."""
    if not student_prompts or not gt_word_dict:
        return 0.0
    mine = word_counts(student_prompts)
    hits = sum(1 for w, c in gt_word_dict.items() if mine.get(w, 0) == c)
    return hits / float(len(gt_word_dict))


def score_phase3(raw, answers, time_scale=1.0):
    trans = answers["transitions"]
    if not raw or not raw.get("ran"):
        return 0.0, 0.0, ["No s'ha executat: " + (raw or {}).get("error", "no s'ha completat (vegeu el missatge general)")]
    got = {(r.get("uuid_1"), r.get("uuid_2")): r for r in raw.get("results", []) if isinstance(r, dict)}
    total, ok, lines = 0.0, 0, []
    for i, tr in enumerate(trans, 1):
        r = got.get((tr["uuid_1"], tr["uuid_2"]))
        if r is None:
            lines.append("  [{}] no executada".format(i))
            continue
        if r.get("error"):
            lines.append("  [{}] ERROR: {}".format(i, r["error"]))
            continue
        prompts = r.get("prompts") or []
        if not prompts:
            lines.append("  [{}] llista buida".format(i))
            continue
        q = transition_quality(prompts, tr["word_dict"])
        total += q
        ok += 1
        lines.append("  [{}] {} passos · {:.2f} s".format(i, len(prompts), r.get("time", 0.0)))
    quality = total / len(trans) if trans else 0.0
    t = float(raw.get("time", 0.0)) * time_scale
    complete = (ok == len(trans) and not raw.get("timeout"))
    bonus = _time_points(10.0, t, P3_TIME_LIMIT) if complete else 0.0
    if SPEED_TIMES_ACCURACY:
        bonus *= quality
    pts = min(PHASE3_MAX, bonus + 30.0 * quality)
    head = ["Transicions correctes: {} de {}".format(ok, len(trans)),
            "Temps: {:.3f} s".format(t),
            "Qualitat mitjana: {:.1%}".format(quality),
            "Punts: temps {:.1f}/10 + qualitat {:.1f}/30".format(bonus, 30.0 * quality)]
    if not complete:
        head.append("Sense punts de temps: la fase no s'ha completat sense errors.")
    return round(pts, 2), quality, head + lines
