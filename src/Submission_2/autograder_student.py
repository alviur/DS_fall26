# -*- coding: utf-8 -*-
"""
autograder_student.py : prova local del segon lliurament.

Executa el MATEIX codi que fa servir Gradescope (runner.py per executar les
vostres classes i scoring.py per puntuar), però amb respostes de mostra:
  - Fase 1: les mateixes 95 cerques de prompt que Gradescope (amb el nombre
    de resultats esperat de cadascuna).
  - Fase 2: 100 consultes de mostra (Gradescope en fa 1,000 de diferents).
  - Fase 3: 2 transicions de mostra (Gradescope en fa 3 de diferents).
Els temps de les Fases 2 i 3 s'escalen per estimar el temps de Gradescope,
però la màquina de Gradescope pot ser més lenta que la vostra.

Com fer-ho servir:
  1. Poseu aquests arxius (autograder_student.py, runner.py, scoring.py i la
     carpeta sample_answers/) a la mateixa carpeta que el vostre codi:
     ImageFiles.py, ImageID.py, ImageData.py, SearchMetadata.py,
     RecommenderSystem.py i cfg.py.
  2. A cfg.py, poseu ROOT_DIR apuntant a la carpeta images/ de l'Image Corpus 10K.
  3. Executeu:  python autograder_student.py  [camí a clip_vectors_2026.json]
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import runner   # noqa: E402
import scoring  # noqa: E402

VECTORS_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "clip_vectors_2026.json")
SAMPLES = os.path.join(HERE, "sample_answers")

# Same limits as Gradescope, adjusted to the number of sample queries.
GRADESCOPE_P2_QUERIES, GRADESCOPE_P3_TRANSITIONS = 1000, 3
LIMITS = {"setup": 180, "phase1": 120, "preprocess": 600, "phase2": 420, "phase3": 360}


def load(name):
    with open(os.path.join(SAMPLES, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    if not os.path.isfile(VECTORS_PATH):
        sys.exit("No trobo el fitxer de vectors: {}\nIndiqueu-ne el camí: "
                 "python autograder_student.py <camí a clip_vectors_2026.json>".format(VECTORS_PATH))
    p1 = load("phase1_counts.json")
    p2 = load("phase2_sample.json")
    p3 = load("phase3_sample.json")
    scale2 = GRADESCOPE_P2_QUERIES / float(len(p2["queries"]))
    scale3 = GRADESCOPE_P3_TRANSITIONS / float(len(p3["transitions"]))
    limits = dict(LIMITS)
    limits["phase2"] = int(LIMITS["phase2"] / scale2)
    limits["phase3"] = int(LIMITS["phase3"] / scale3)
    config = {
        "vectors_path": VECTORS_PATH,
        "phase1_words": p1["words"],
        "phase2_queries": p2["queries"],
        "phase3_pairs": [[t["uuid_1"], t["uuid_2"]] for t in p3["transitions"]],
        "k": p2.get("k", 10),
        "limits": limits,
    }
    r = runner.Runner(config, progress_path=None, capture=False)
    if not r.has_alarm:
        print("AVÍS: en aquest sistema (Windows) no es poden aplicar els límits de temps. "
              "A Gradescope sí que s'apliquen.\n")
    p = r.run_all()

    print("\n" + "=" * 70)
    print("RESULTATS (estimació local amb respostes de mostra)")
    print("=" * 70)
    for mod, err in p["imports"].items():
        print("\nERROR important {}.py:\n{}".format(mod, err))
    su = p["setup"]
    print("\nCàrrega de dades: {} imatges, {} UUIDs, {} metadades ({:.1f} s)".format(
        su.get("files", 0), su.get("uuids", 0), su.get("metadata", 0), su.get("time", 0.0)))
    if su.get("error"):
        print("  ERROR: " + su["error"])
    s1, l1, _ = scoring.score_phase1(p["phase1"], p1["expected_counts"])
    s2, prec, l2 = scoring.score_phase2(p["phase2"], p2, time_scale=scale2)
    s3, qual, l3 = scoring.score_phase3(p["phase3"], p3, time_scale=scale3)
    print("\nFase 1 · SearchMetadata: {:.1f} / 20".format(s1))
    print("\n".join("  " + x for x in l1))
    print("\nFase 2 · Similitud d'imatges: {:.1f} / 40   (temps x{:.0f} per estimar 1,000 consultes)".format(s2, scale2))
    print("\n".join("  " + x for x in l2))
    print("\nFase 3 · Camins de transició: {:.1f} / 40   (temps x{:.1f} per estimar 3 transicions)".format(s3, scale3))
    print("\n".join("  " + x for x in l3))
    total = s1 + s2 + s3
    print("\nTOTAL ESTIMAT: {:.1f} / 100".format(total))
    print("Recordeu: Gradescope fa servir altres consultes i la seva màquina pot ser més lenta.")


if __name__ == "__main__":
    main()
