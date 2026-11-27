# -*- coding: utf-8 -*-
"""
runner.py : runs the student's code for the three phases and records the raw
results (no answers, no points) in a progress file.

On Gradescope it runs as an unprivileged user in a throwaway folder, started
by harness.py:   python3 runner.py <config.json> <progress.json>
autograder_student.py imports it and calls run_all() directly.

Each phase has its own time limit. When it runs out, the phase stops, what was
done so far is kept, and the next phase still runs. The progress file is saved
after every step, so a hard kill never loses the results already recorded.
"""

import contextlib
import gc
import importlib
import io
import json
import os
import signal
import sys
import time
import traceback

STUDENT_MODULES = ("ImageFiles", "ImageID", "ImageData", "SearchMetadata", "RecommenderSystem")
MAX_PRINT = 6000          # characters of student output kept per phase
SAVE_EVERY = 25           # Phase 2: save progress every N queries


class PhaseTimeout(BaseException):
    """Raised by the alarm. BaseException, so `except Exception` in student code doesn't swallow it."""


class LimitedBuffer(io.StringIO):
    def write(self, s):
        if self.tell() < MAX_PRINT:
            return super().write(s[:MAX_PRINT - self.tell()])
        return len(s)


def short_error(exc):
    """Traceback limited to the student's own files."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__)
              if not f.filename.endswith(("runner.py", "autograder_student.py"))
              and not f.filename.startswith("<frozen") and "importlib" not in f.filename]
    text = "".join(traceback.format_list(frames[-6:]) + traceback.format_exception_only(type(exc), exc))
    return text.strip()[:2000]


class Runner:
    def __init__(self, config, progress_path=None, capture=True):
        self.cfg = config
        self.path = progress_path
        self.capture = capture
        self.p = {"imports": {}, "setup": {}, "phase1": {}, "phase2": {}, "phase3": {}}
        self.deadline = None
        self.has_alarm = hasattr(signal, "SIGALRM")
        if self.has_alarm:
            signal.signal(signal.SIGALRM, self._on_alarm)

    # ------------------------------------------------------------ helpers
    def _on_alarm(self, signum, frame):
        raise PhaseTimeout()

    def save(self):
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.p, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    @contextlib.contextmanager
    def limit(self, seconds):
        """Raise PhaseTimeout inside the block after `seconds` (wall clock)."""
        self.deadline = time.perf_counter() + seconds if seconds else None
        # Tell the harness watchdog when this step must end (wall clock).
        self.p["current"] = {"deadline": time.time() + seconds} if seconds else None
        self.save()
        if self.has_alarm and seconds:
            signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            if self.has_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
            self.deadline = None
            self.p["current"] = None

    def check(self):
        """Stop the loop even if student code swallowed the alarm (bare except)."""
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise PhaseTimeout()

    @contextlib.contextmanager
    def output(self, phase):
        buf = LimitedBuffer()
        if self.capture:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                yield
        else:
            yield
        if self.capture and buf.getvalue().strip():
            self.p[phase]["printed"] = buf.getvalue()

    # ------------------------------------------------------------ steps
    def imports(self):
        mods = {}
        for name in STUDENT_MODULES:
            buf = LimitedBuffer()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), self.limit(60):
                    mods[name] = importlib.import_module(name)
            except PhaseTimeout:
                self.p["imports"][name] = "Temps esgotat important el mòdul."
            except BaseException as exc:
                self.p["imports"][name] = short_error(exc)
        self.save()
        return mods

    def setup(self, mods):
        """Load the 10,000 images with the student's Phase 1 classes (no points)."""
        s = self.p["setup"]
        s.update(ran=False, files=0, uuids=0, metadata=0)
        need = ("ImageFiles", "ImageID", "ImageData")
        if any(n not in mods for n in need):
            s["error"] = "No es poden importar ImageFiles, ImageID o ImageData."
            self.save()
            return None, None
        t0 = time.perf_counter()
        image_data = image_id = None
        with self.output("setup"):
            try:
                with self.limit(self.cfg["limits"]["setup"]):
                    import cfg
                    files = mods["ImageFiles"].ImageFiles()
                    files.reload_fs(cfg.get_root())
                    added = files.files_added()
                    s["files"] = len(added)
                    image_id = mods["ImageID"].ImageID()
                    image_data = mods["ImageData"].ImageData()
                    first_err = None
                    for f in sorted(added):
                        rel = cfg.get_canonical_pathfile(str(f))
                        uu = image_id.generate_uuid(rel)
                        if not uu:
                            continue
                        s["uuids"] += 1
                        try:
                            image_data.add_image(uu, rel)
                            image_data.load_metadata(uu)
                            s["metadata"] += 1
                        except Exception as exc:
                            first_err = first_err or short_error(exc)
                    if first_err:
                        s["first_metadata_error"] = first_err
                    s["ran"] = True
            except PhaseTimeout:
                s["error"] = "Temps esgotat ({} s) carregant les imatges.".format(self.cfg["limits"]["setup"])
            except BaseException as exc:
                s["error"] = short_error(exc)
        s["time"] = time.perf_counter() - t0
        self.save()
        return image_data, image_id

    def phase1(self, mods, image_data):
        p = self.p["phase1"]
        words = self.cfg["phase1_words"]
        p.update(ran=False, words=words, queries=[], search_time=0.0)
        if "SearchMetadata" not in mods or image_data is None:
            p["error"] = "Cal SearchMetadata i les imatges carregades (vegeu 'Càrrega de dades')."
            self.save()
            return None
        searcher = None
        with self.output("phase1"):
            try:
                with self.limit(self.cfg["limits"]["phase1"]):
                    t0 = time.perf_counter()      # building the searcher counts, as in 2025
                    searcher = mods["SearchMetadata"].SearchMetadata(image_data)
                    p["search_time"] += time.perf_counter() - t0
                    p["ran"] = True
                    for w in words:
                        self.check()
                        item = {"word": w}
                        t0 = time.perf_counter()
                        try:
                            res = searcher.prompt(w)
                            p["search_time"] += time.perf_counter() - t0
                            res = list(res) if res is not None else []
                            item["count"] = len(res)
                            item["duplicates"] = len(set(map(str, res))) != len(res)
                            item["stems"] = self._stems(image_data, res)
                        except PhaseTimeout:
                            raise
                        except Exception as exc:
                            p["search_time"] += time.perf_counter() - t0
                            item["error"] = short_error(exc).splitlines()[-1]
                        p["queries"].append(item)
            except PhaseTimeout:
                p["timeout"] = True
            except BaseException as exc:
                p["error"] = short_error(exc)
        self.save()
        return searcher

    @staticmethod
    def _stems(image_data, uuids):
        """File names (without .png) of the returned images, via get_filename (None if unavailable)."""
        try:
            out = []
            for u in uuids:
                f = image_data.get_filename(u)
                if not f:
                    return None
                out.append(os.path.splitext(os.path.basename(str(f)))[0])
            return out
        except Exception:
            return None

    def phase2_and_3(self, mods, image_data, image_id):
        p2, p3 = self.p["phase2"], self.p["phase3"]
        p2.update(ran=False, results=[], query_time=0.0, preprocess_time=0.0)
        p3.update(ran=False, results=[], time=0.0)
        if "RecommenderSystem" not in mods:
            p2["error"] = p3["error"] = "No es pot importar RecommenderSystem.py."
            self.save()
            return
        system = None
        with self.output("phase2"):
            t0 = time.perf_counter()
            try:
                with self.limit(self.cfg["limits"]["preprocess"]):
                    system = mods["RecommenderSystem"].RecommenderSystem(
                        self.cfg["vectors_path"], image_data, image_id)
                    system.preprocess()
            except PhaseTimeout:
                p2["error"] = "Temps esgotat ({} s) al constructor + preprocess()." \
                    .format(self.cfg["limits"]["preprocess"])
                system = None
            except BaseException as exc:
                p2["error"] = "Error al constructor o a preprocess():\n" + short_error(exc)
                system = None
            p2["preprocess_time"] = time.perf_counter() - t0
            self.save()
            if system is not None:
                p2["ran"] = True
                k = self.cfg.get("k", 10)
                try:
                    with self.limit(self.cfg["limits"]["phase2"]):
                        for n, q in enumerate(self.cfg["phase2_queries"], 1):
                            self.check()
                            r = {"query": q}
                            t1 = time.perf_counter()
                            try:
                                g = system.find_similar_images(q, k=k)
                                p2["query_time"] += time.perf_counter() - t1
                                r["ids"] = self._ids(g)[:k]
                            except PhaseTimeout:
                                raise
                            except Exception as exc:
                                p2["query_time"] += time.perf_counter() - t1
                                r["error"] = short_error(exc)
                            p2["results"].append(r)
                            if n % SAVE_EVERY == 0:
                                self.save()
                except PhaseTimeout:
                    p2["timeout"] = True
                    p2["query_time"] = float(self.cfg["limits"]["phase2"])
        self.save()
        gc.collect()
        if system is None:
            p3["error"] = "No s'ha pogut crear el RecommenderSystem (vegeu la Fase 2)."
            self.save()
            return
        with self.output("phase3"):
            p3["ran"] = True
            try:
                with self.limit(self.cfg["limits"]["phase3"]):
                    for tr in self.cfg["phase3_pairs"]:
                        self.check()
                        r = {"uuid_1": tr[0], "uuid_2": tr[1]}
                        t1 = time.perf_counter()
                        try:
                            res = system.find_transition_prompts(tr[0], tr[1])
                            r["prompts"] = [p if isinstance(p, str) else str(p) for p in (res or [])]
                        except PhaseTimeout:
                            raise
                        except Exception as exc:
                            r["error"] = short_error(exc)
                        r["time"] = time.perf_counter() - t1
                        p3["time"] += r["time"]
                        p3["results"].append(r)
                        self.save()
            except PhaseTimeout:
                p3["timeout"] = True
                p3["time"] = float(self.cfg["limits"]["phase3"])
                done = {(x["uuid_1"], x["uuid_2"]) for x in p3["results"]}
                for tr in self.cfg["phase3_pairs"]:
                    if (tr[0], tr[1]) not in done:
                        p3["results"].append({"uuid_1": tr[0], "uuid_2": tr[1],
                                              "error": "Temps esgotat ({} s per a la Fase 3)."
                                              .format(self.cfg["limits"]["phase3"])})
                        break
        self.save()

    @staticmethod
    def _ids(g):
        """Accept a Gallery-like object with .images, or a plain list/tuple."""
        if hasattr(g, "images"):
            g = g.images
        if g is None:
            return []
        return [str(x) for x in list(g)]

    def run_all(self):
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        mods = self.imports()
        image_data, image_id = self.setup(mods)
        self.phase1(mods, image_data)
        gc.collect()
        self.phase2_and_3(mods, image_data, image_id)
        self.p["finished"] = True
        self.save()
        return self.p


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        config = json.load(f)
    Runner(config, sys.argv[2]).run_all()
