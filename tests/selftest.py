#!/usr/bin/env python3
"""End-to-end self test for the ensemble rainfall probability API.

By default the script spawns TWO uvicorn processes sharing one SQLite file
(dual-instance deployment), waits for health, and exercises every key flow:

  * registration / member-count / threshold / datetime validation
  * idempotent resubmit (same revision+content) and conflicting resubmit rejection
  * revision ordering, grid-shape and non-negativity checks
  * publish blocked until every member is present
  * probability / denominator / all-missing null semantics and used revisions
  * correction immediately expires the old product while history stays queryable
  * two-instance interleaved publish vs. correction -> only consistent products
  * parallel duplicate publishes collapse to one idempotent product
  * injected compute failure leaves no half-written product (atomicity)
  * slow computation never blocks another forecast time's ingestion
  * Swagger UI / OpenAPI availability

Hourly verification coverage (sections 13-19):
  * first observation rejected while no current product exists; locking otherwise
  * observation revision ordering, idempotent resend and conflicting-content rejection
  * Brier MSE math per threshold with valid-cell filtering and null score at 0 cells
  * member corrections never change the locked product; observation corrections
    immediately invalidate the old verification while history stays queryable
  * dual-instance interleaved verification vs observation correction -> retries,
    only the latest observation revision has a current verification
  * parallel verifications collapse to one idempotent row
  * injected verification failure leaves no half-written verification
  * slow verification never blocks another forecast time's member ingestion

Only the Python standard library is required.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS_COUNT = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS_COUNT
    if not cond:
        raise AssertionError(f"FAIL: {name} {detail}")
    PASS_COUNT += 1
    print(f"  PASS  {name}")


def http(method: str, base: str, path: str, body: Optional[dict] = None) -> Tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"raw": raw}


def approx(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def matrix_approx(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for x, y in zip(ra, rb):
            if not approx(x, y):
                return False
    return True


# --------------------------------------------------------------------------- #
# dual-instance process management
# --------------------------------------------------------------------------- #
def spawn_servers() -> Tuple[list, str, str, str, str, str]:
    tmp = tempfile.mkdtemp(prefix="rain-selftest-")
    db_path = os.path.join(tmp, "rain.db")
    fail_marker = os.path.join(tmp, "fail.marker")
    verify_fail_marker = os.path.join(tmp, "verify-fail.marker")
    env = dict(os.environ)
    env.update(
        RAIN_DB_PATH=db_path,
        RAIN_TEST_PUBLISH_DELAY_S="0.6",
        RAIN_TEST_FAIL_MARKER=fail_marker,
        RAIN_TEST_VERIFY_DELAY_S="0.6",
        RAIN_TEST_VERIFY_FAIL_MARKER=verify_fail_marker,
        PYTHONPATH=ROOT,
    )
    procs = []
    for port in (8091, 8092):
        log = open(os.path.join(tmp, f"uvicorn-{port}.log"), "w")
        procs.append(
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app",
                 "--host", "127.0.0.1", "--port", str(port)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        )
    base_a, base_b = "http://127.0.0.1:8091", "http://127.0.0.1:8092"
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            s1, _ = http("GET", base_a, "/health")
            s2, _ = http("GET", base_b, "/health")
            if s1 == 200 and s2 == 200:
                return procs, base_a, base_b, fail_marker, verify_fail_marker, tmp
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("servers did not become healthy")


def shutdown(procs: list) -> None:
    for p in procs:
        p.send_signal(signal.SIGINT)
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def main() -> int:
    base_a = os.environ.get("RAIN_BASE_A")
    base_b = os.environ.get("RAIN_BASE_B")
    fail_marker = os.environ.get("RAIN_TEST_FAIL_MARKER", "")
    verify_fail_marker = os.environ.get("RAIN_TEST_VERIFY_FAIL_MARKER", "")
    spawned = False
    procs: list = []
    if not base_a or not base_b:
        procs, base_a, base_b, fail_marker, verify_fail_marker, _ = spawn_servers()
        spawned = True
        print(f"[setup] spawned dual instances {base_a} / {base_b}")
    hooks_enabled = bool(fail_marker)
    try:
        run(base_a, base_b, fail_marker, verify_fail_marker, hooks_enabled)
    finally:
        if spawned:
            shutdown(procs)
    print(f"\nALL TESTS PASSED ({PASS_COUNT} assertions)")
    return 0


def register(base: str, members: int = 3, rows: int = 2, cols: int = 2,
             thresholds=(5.0, 10.0), names=None) -> dict:
    if names is None:
        names = [f"m{i}" for i in range(1, members + 1)]
    body = {
        "forecast_time": "2026-11-16T00:00:00Z",
        "rows": rows, "cols": cols,
        "member_names": names,
        "thresholds": list(thresholds),
    }
    status, data = http("POST", base, "/groups", body)
    assert status == 201, (status, data)
    return data


def run(base_a: str, base_b: str, fail_marker: str,
        verify_fail_marker: str, hooks_enabled: bool) -> None:
    print("\n[1] validation on registration")
    bad = {"forecast_time": "2026-11-16T00:00:00Z", "rows": 2, "cols": 2,
           "member_names": ["only-one"], "thresholds": [5.0]}
    s, d = http("POST", base_a, "/groups", bad)
    check("1 member rejected (need 2-8)", s == 422, str(d))
    bad["member_names"] = [f"m{i}" for i in range(9)]
    s, d = http("POST", base_a, "/groups", bad)
    check("9 members rejected", s == 422)
    bad["member_names"] = ["a", "b"]
    bad["thresholds"] = [-1.0]
    s, d = http("POST", base_a, "/groups", bad)
    check("negative threshold rejected", s == 422)
    bad["thresholds"] = [5.0]
    bad["forecast_time"] = "2026-11-16T00:00:00"  # naive, no timezone
    s, d = http("POST", base_a, "/groups", bad)
    check("naive datetime rejected", s == 422)

    print("\n[2] happy-path group + member submission semantics")
    g = register(base_a, members=3, thresholds=[5.0, 10.0])
    gid = g["group_id"]
    check("registered 3 members", len(g["members"]) == 3)
    s, d = http("GET", base_b, f"/groups/{gid}")
    check("group visible from instance B", s == 200 and d["all_members_present"] is False)

    rain1 = [[0.0, 10.0], [20.0, 0.0]]
    s, d = http("PUT", base_a, f"/groups/{gid}/members/m1",
                {"rain": rain1, "revision": 1})
    check("m1 rev1 accepted", s == 200 and d["duplicate"] is False, str(d))
    s, d2 = http("PUT", base_b, f"/groups/{gid}/members/m1",
                 {"rain": rain1, "revision": 1})
    check("same revision+content resent is idempotent", s == 200 and d2["duplicate"] is True)
    check("idempotent hash stable", d["content_hash"] == d2["content_hash"])
    s, d = http("PUT", base_a, f"/groups/{gid}/members/m1",
                {"rain": [[0.0, 10.0], [20.0, 1.0]], "revision": 1})
    check("same revision different content rejected (异文拒收)",
          s == 409 and d["error"]["code"] == "REVISION_CONTENT_CONFLICT", str(d))
    s, d = http("PUT", base_a, f"/groups/{gid}/members/ghost",
                {"rain": rain1, "revision": 1})
    check("unknown member -> 404", s == 404 and d["error"]["code"] == "MEMBER_NOT_FOUND")
    s, d = http("PUT", base_a, "/groups/nope/members/m1", {"rain": rain1, "revision": 1})
    check("unknown group -> 404", s == 404 and d["error"]["code"] == "GROUP_NOT_FOUND")

    print("\n[3] revision ordering and grid validation on a second group")
    g2 = register(base_b, members=2, thresholds=[5.0])
    gid2 = g2["group_id"]
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0], [3.0, 4.0]], "revision": 1})
    check("g2 m1 rev1", s == 200)
    s, d = http("PUT", base_b, f"/groups/{gid2}/members/m1",
                {"rain": [[9.0, 9.0], [9.0, 9.0]], "revision": 2})
    check("g2 m1 rev2", s == 200)
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0], [3.0, 4.0]], "revision": 1})
    check("lower revision rejected", s == 409 and d["error"]["code"] == "REVISION_CONFLICT")
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "revision": 3})
    check("grid shape mismatch rejected", s == 422 and d["error"]["code"] == "GRID_MISMATCH")
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[-1.0, 2.0], [3.0, 4.0]], "revision": 3})
    check("negative rain rejected", s == 422 and d["error"]["code"] == "NEGATIVE_RAIN")
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0], [3.0, 4.0]], "missing": [[1, 0, 0]], "revision": 3})
    check("missing grid shape mismatch rejected",
          s == 422 and d["error"]["code"] == "GRID_MISMATCH", str(d))

    print("\n[4] publishing requires every member")
    s, d = http("POST", base_a, f"/groups/{gid}/publish")
    check("incomplete group cannot publish",
          s == 409 and d["error"]["code"] == "MEMBERS_INCOMPLETE", str(d))

    print("\n[5] complete group and verify probability math")
    rain2 = [[10.0, 10.0], [0.0, 0.0]]
    rain3 = [[5.0, 0.0], [99.0, 0.0]]
    miss3 = [[0, 0], [1, 1]]  # m3 missing on second row
    s, _ = http("PUT", base_b, f"/groups/{gid}/members/m2",
                {"rain": rain2, "revision": 1})
    check("m2 rev1 via instance B", s == 200)
    s, _ = http("PUT", base_a, f"/groups/{gid}/members/m3",
                {"rain": rain3, "missing": miss3, "revision": 1})
    check("m3 rev1 with missing grid", s == 200)

    s, p1 = http("POST", base_a, f"/groups/{gid}/publish")
    check("publish succeeds when complete", s == 200, str(p1))
    check("product marked current", p1["is_current"] is True)
    check("3 member revisions recorded",
          {m["member_id"]: m["revision"] for m in p1["members_used"]}
          == {"m1": 1, "m2": 1, "m3": 1})
    by_thr = {t["threshold"]: t for t in p1["thresholds"]}
    check("denominators exclude missing members",
          by_thr[5.0]["denominator"] == [[3, 3], [2, 2]],
          str(by_thr[5.0]["denominator"]))
    check("probability threshold=5",
          matrix_approx(by_thr[5.0]["probability"],
                        [[2 / 3, 2 / 3], [0.5, 0.0]]),
          str(by_thr[5.0]["probability"]))
    check("probability threshold=10",
          matrix_approx(by_thr[10.0]["probability"],
                        [[1 / 3, 2 / 3], [0.5, 0.0]]),
          str(by_thr[10.0]["probability"]))
    s, p1b = http("POST", base_b, f"/groups/{gid}/publish")
    check("republish identical revision set is idempotent",
          s == 200 and p1b["product_id"] == p1["product_id"])
    s, cur = http("GET", base_a, f"/groups/{gid}/products/current")
    check("current product query", s == 200 and cur["product_id"] == p1["product_id"])
    s, listing = http("GET", base_b, f"/groups/{gid}/products")
    check("one product in history listing", s == 200 and len(listing["products"]) == 1)
    s, d = http("GET", base_a, f"/groups/{gid2}/products/{p1['product_id']}")
    check("product from other group -> 404", s == 404)

    print("\n[6] all-missing cell yields null probability and zero denominator")
    g3 = register(base_a, members=2, rows=1, cols=2, thresholds=[5.0])
    gid3 = g3["group_id"]
    http("PUT", base_a, f"/groups/{gid3}/members/m1",
         {"rain": [[5.0, 9.0]], "missing": [[0, 1]], "revision": 1})
    http("PUT", base_b, f"/groups/{gid3}/members/m2",
         {"rain": [[10.0, 9.0]], "missing": [[0, 1]], "revision": 1})
    s, p3 = http("POST", base_a, f"/groups/{gid3}/publish")
    t3 = p3["thresholds"][0]
    check("all-missing cell null", t3["probability"] == [[1.0, None]], str(t3))
    check("all-missing cell denominator 0", t3["denominator"] == [[2, 0]])

    print("\n[7] correction expires old product immediately, history retained")
    s, _ = http("PUT", base_b, f"/groups/{gid}/members/m1",
                {"rain": [[0.0, 10.0], [20.0, 5.0]], "revision": 2})
    check("correction rev2 accepted", s == 200)
    s, d = http("GET", base_a, f"/groups/{gid}/products/current")
    check("no stale product served as current warning basis",
          s == 404 and d["error"]["code"] == "NO_CURRENT_PRODUCT", str(d))
    s, listing = http("GET", base_b, f"/groups/{gid}/products")
    check("old product retained but superseded",
          len(listing["products"]) == 1 and listing["products"][0]["is_current"] is False)
    s, hist = http("GET", base_a, f"/groups/{gid}/products/{p1['product_id']}")
    check("historical product still queryable", s == 200 and hist["is_current"] is False)
    s, p2 = http("POST", base_b, f"/groups/{gid}/publish")
    check("republish after correction", s == 200 and p2["product_id"] != p1["product_id"])
    check("new product uses rev2 for corrected member",
          {m["member_id"]: m["revision"] for m in p2["members_used"]}
          == {"m1": 2, "m2": 1, "m3": 1})
    s, cur = http("GET", base_b, f"/groups/{gid}/products/current")
    check("current points at new product", cur["product_id"] == p2["product_id"])

    print("\n[8] dual-instance interleaving: publish vs concurrent correction")
    g4 = register(base_a, members=3, rows=64, cols=64, thresholds=[5.0, 20.0])
    gid4 = g4["group_id"]
    for i, inst in enumerate((base_a, base_b, base_a)):
        name = f"m{i + 1}"
        grid = [[float((i + 1) * (r + c) % 25) for c in range(64)] for r in range(64)]
        s, _ = http("PUT", inst, f"/groups/{gid4}/members/{name}",
                    {"rain": grid, "revision": 1})
        assert s == 200

    result: Dict[str, Any] = {}

    def worker_publish() -> None:
        result["status"], result["body"] = http("POST", base_a, f"/groups/{gid4}/publish")

    t = threading.Thread(target=worker_publish)
    t.start()
    time.sleep(0.15)  # let the publish enter its slow compute phase
    t0 = time.time()
    s, corr = http("PUT", base_b, f"/groups/{gid4}/members/m2",
                   {"rain": [[7.0] * 64 for _ in range(64)], "revision": 2})
    write_elapsed = time.time() - t0
    check("correction write not blocked by in-flight compute", s == 200, str(corr))
    if hooks_enabled:
        check("ingest stayed fast while compute was running", write_elapsed < 0.5,
              f"{write_elapsed:.2f}s")
    t.join()
    check("in-flight publish eventually returns 200", result["status"] == 200, str(result))

    s, status4 = http("GET", base_a, f"/groups/{gid4}")
    latest = {m["member_id"]: m["latest_revision"] for m in status4["members"]}
    s, cur4 = http("GET", base_b, f"/groups/{gid4}/products/current")
    check("a current product exists after interleaving", s == 200)
    cur_map = {m["member_id"]: m["revision"] for m in cur4["members_used"]}
    check("current product matches latest member revisions (no mixed revision group)",
          cur_map == latest, f"{cur_map} vs {latest}")
    s, listing4 = http("GET", base_a, f"/groups/{gid4}/products")
    for prod in listing4["products"]:
        check(f"product {prod['product_id']} has complete revision set",
              len(prod["revision_map"]) == 3)
        if prod["is_current"]:
            check("only one current product",
                  sum(1 for x in listing4["products"] if x["is_current"]) == 1)
    check("current listing entry consistent",
          [x for x in listing4["products"] if x["is_current"]][0]["product_id"]
          == cur4["product_id"])

    print("\n[9] parallel publishes of the same revision set collapse to one product")
    g5 = register(base_b, members=2, rows=8, cols=8, thresholds=[5.0])
    gid5 = g5["group_id"]
    http("PUT", base_a, f"/groups/{gid5}/members/m1",
         {"rain": [[6.0] * 8 for _ in range(8)], "revision": 1})
    http("PUT", base_b, f"/groups/{gid5}/members/m2",
         {"rain": [[0.0] * 8 for _ in range(8)], "revision": 1})
    outs: Dict[str, Any] = {}

    def pub(instance: str) -> None:
        outs[instance] = http("POST", instance, f"/groups/{gid5}/publish")

    threads = [threading.Thread(target=pub, args=(base_a,)),
               threading.Thread(target=pub, args=(base_b,))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    ids = {body["product_id"] for st, body in outs.values() if st == 200}
    check("both parallel publishes succeed", all(st == 200 for st, _ in outs.values()))
    check("both return the same idempotent product", len(ids) == 1, str(ids))
    s, listing5 = http("GET", base_a, f"/groups/{gid5}/products")
    check("exactly one product row for the revision set", len(listing5["products"]) == 1)

    print("\n[10] compute failure leaves no half-written product")
    g6 = register(base_a, members=2, rows=4, cols=4, thresholds=[5.0])
    gid6 = g6["group_id"]
    http("PUT", base_a, f"/groups/{gid6}/members/m1",
         {"rain": [[6.0] * 4 for _ in range(4)], "revision": 1})
    http("PUT", base_b, f"/groups/{gid6}/members/m2",
         {"rain": [[4.0] * 4 for _ in range(4)], "revision": 1})
    if hooks_enabled:
        with open(fail_marker, "w") as fh:
            fh.write("fail")
        s, d = http("POST", base_a, f"/groups/{gid6}/publish")
        os.remove(fail_marker)
        check("injected failure surfaces as 500", s == 500 and d["error"]["code"] == "COMPUTE_FAILED")
        s, d = http("GET", base_b, f"/groups/{gid6}/products/current")
        check("no current product after failed compute", s == 404)
        s, listing6 = http("GET", base_a, f"/groups/{gid6}/products")
        check("no product rows at all after failed compute", listing6["products"] == [])
        s, p6 = http("POST", base_b, f"/groups/{gid6}/publish")
        check("publish succeeds after failure cleared", s == 200 and p6["is_current"])
    else:
        print("  SKIP  failure injection (RAIN_TEST_FAIL_MARKER not configured)")

    print("\n[11] slow compute on one time does not block another time's ingestion")
    g7 = register(base_a, members=2, rows=16, cols=16, thresholds=[5.0])
    gid7 = g7["group_id"]
    http("PUT", base_a, f"/groups/{gid7}/members/m1",
         {"rain": [[6.0] * 16 for _ in range(16)], "revision": 1})
    http("PUT", base_a, f"/groups/{gid7}/members/m2",
         {"rain": [[6.0] * 16 for _ in range(16)], "revision": 1})

    bg = {}

    def slow_publish() -> None:
        bg["result"] = http("POST", base_a, f"/groups/{gid7}/publish")

    th = threading.Thread(target=slow_publish)
    th.start()
    time.sleep(0.15)
    t0 = time.time()
    g8 = register(base_b, members=2, rows=3, cols=3, thresholds=[5.0])
    http("PUT", base_b, f"/groups/{g8['group_id']}/members/m1",
         {"rain": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], "revision": 1})
    s, _ = http("PUT", base_b, f"/groups/{g8['group_id']}/members/m2",
                {"rain": [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]], "revision": 1})
    ingest_elapsed = time.time() - t0
    check("other time registration+ingest succeeded during slow compute", s == 200)
    if hooks_enabled:
        check("other time ingest stayed fast", ingest_elapsed < 0.5, f"{ingest_elapsed:.2f}s")
    th.join()
    check("slow publish completed", bg["result"][0] == 200, str(bg))

    # ------------------------------------------------------------------ #
    # Hourly verification: observation locking, Brier scores, revisions
    # ------------------------------------------------------------------ #
    print("\n[13] observation upload gates and grid validation")
    gv = register(base_a, members=3, rows=2, cols=2, thresholds=[5.0, 10.0])
    gvid = gv["group_id"]
    # no product yet -> first observation must be rejected (nothing to lock)
    s, d = http("PUT", base_a, f"/groups/{gvid}/observations",
                {"rain": [[0.0, 10.0], [20.0, 0.0]], "revision": 1})
    check("first observation rejected without current product",
          s == 404 and d["error"]["code"] == "NO_CURRENT_PRODUCT", str(d))
    # verification without observation also rejected
    s, d = http("POST", base_a, f"/groups/{gvid}/verifications")
    check("verification without observation rejected",
          s == 409 and d["error"]["code"] == "NO_OBSERVATION", str(d))
    s, d = http("GET", base_a, f"/groups/{gvid}/verifications/current")
    check("no current verification -> 404",
          s == 404 and d["error"]["code"] == "NO_CURRENT_VERIFICATION")
    # build a product identical to section [5] so the Brier math is known
    http("PUT", base_a, f"/groups/{gvid}/members/m1",
         {"rain": [[0.0, 10.0], [20.0, 0.0]], "revision": 1})
    http("PUT", base_b, f"/groups/{gvid}/members/m2",
         {"rain": [[10.0, 10.0], [0.0, 0.0]], "revision": 1})
    http("PUT", base_a, f"/groups/{gvid}/members/m3",
         {"rain": [[5.0, 0.0], [99.0, 0.0]], "missing": [[0, 0], [1, 1]], "revision": 1})
    pv = http("POST", base_a, f"/groups/{gvid}/publish")[1]
    lock_id = pv["product_id"]
    s, d = http("PUT", base_a, f"/groups/{gvid}/observations",
                {"rain": [[3.0, 3.0]], "revision": 1})
    check("observation grid mismatch rejected",
          s == 422 and d["error"]["code"] == "GRID_MISMATCH", str(d))
    s, d = http("PUT", base_a, f"/groups/{gvid}/observations",
                {"rain": [[-1.0, 1.0], [1.0, 1.0]], "revision": 1})
    check("negative observation rain rejected",
          s == 422 and d["error"]["code"] == "NEGATIVE_RAIN")
    s, d = http("PUT", base_a, f"/groups/{gvid}/observations",
                {"rain": [[1.0, 1.0], [1.0, 1.0]], "missing": [[0, 0]], "revision": 1})
    check("observation missing-grid mismatch rejected", s == 422)
    s, d = http("PUT", base_a, "/groups/nope/observations",
                {"rain": [[1.0, 1.0], [1.0, 1.0]], "revision": 1})
    check("observation on unknown group -> 404",
          s == 404 and d["error"]["code"] == "GROUP_NOT_FOUND")

    print("\n[14] first observation locks current product; idempotency and 异文拒收")
    obs_rain = [[0.0, 10.0], [20.0, 0.0]]
    s, o1 = http("PUT", base_a, f"/groups/{gvid}/observations",
                 {"rain": obs_rain, "revision": 1})
    check("first observation accepted and locks current product",
          s == 200 and o1["locked_product_id"] == lock_id and o1["duplicate"] is False, str(o1))
    s, o1b = http("PUT", base_b, f"/groups/{gvid}/observations",
                  {"rain": obs_rain, "revision": 1})
    check("same observation revision+content resent is idempotent",
          s == 200 and o1b["duplicate"] is True
          and o1b["locked_product_id"] == lock_id and o1b["content_hash"] == o1["content_hash"])
    s, d = http("PUT", base_a, f"/groups/{gvid}/observations",
                {"rain": [[0.0, 10.0], [20.0, 1.0]], "revision": 1})
    check("same observation revision different content rejected (异文拒收)",
          s == 409 and d["error"]["code"] == "OBSERVATION_REVISION_CONTENT_CONFLICT", str(d))
    # ordering check needs a strictly lower revision than the latest one;
    # advance this group's observation to revision 2 on a separate fresh group
    go = register(base_b, members=2, rows=1, cols=2, thresholds=[5.0])
    goid = go["group_id"]
    http("PUT", base_a, f"/groups/{goid}/members/m1",
         {"rain": [[1.0, 1.0]], "revision": 1})
    http("PUT", base_b, f"/groups/{goid}/members/m2",
         {"rain": [[2.0, 2.0]], "revision": 1})
    http("POST", base_a, f"/groups/{goid}/publish")
    http("PUT", base_a, f"/groups/{goid}/observations",
         {"rain": [[1.0, 1.0]], "revision": 1})
    http("PUT", base_b, f"/groups/{goid}/observations",
         {"rain": [[2.0, 2.0]], "revision": 2})
    s, d = http("PUT", base_a, f"/groups/{goid}/observations",
                {"rain": [[7.0, 7.0]], "revision": 1})
    check("lower observation revision rejected",
          s == 409 and d["error"]["code"] == "OBSERVATION_REVISION_CONFLICT", str(d))

    print("\n[15] Brier score math, valid-cell filtering and null score")
    s, v1 = http("POST", base_a, f"/groups/{gvid}/verifications")
    check("verification created against locked product",
          s == 200 and v1["locked_product_id"] == lock_id and v1["obs_revision"] == 1, str(v1))
    # locked product (from section-5 math):
    #   thr 5  prob = [2/3, 2/3, 1/2, 0], thr 10 prob = [1/3, 2/3, 1/2, 0]
    # observed outcomes at rain [0,10,20,0]:
    #   thr 5  = [0,1,1,0]; thr 10 = [0,1,1,0]
    probs5 = [2 / 3, 2 / 3, 0.5, 0.0]
    probs10 = [1 / 3, 2 / 3, 0.5, 0.0]
    outcomes = [0.0, 1.0, 1.0, 0.0]
    brier5 = sum((p - o) ** 2 for p, o in zip(probs5, outcomes)) / 4
    brier10 = sum((p - o) ** 2 for p, o in zip(probs10, outcomes)) / 4
    vby = {t["threshold"]: t for t in v1["thresholds"]}
    check("brier threshold 5 value and valid cells",
          vby[5.0]["valid_cells"] == 4 and approx(vby[5.0]["brier_score"], brier5),
          str(vby[5.0]))
    check("brier threshold 10 value and valid cells",
          vby[10.0]["valid_cells"] == 4 and approx(vby[10.0]["brier_score"], brier10),
          str(vby[10.0]))
    s, v1again = http("POST", base_b, f"/groups/{gvid}/verifications")
    check("repeated verification is idempotent (same row)",
          s == 200 and v1again["verification_id"] == v1["verification_id"])
    s, cur = http("GET", base_a, f"/groups/{gvid}/verifications/current")
    check("current verification fetch", s == 200 and cur["verification_id"] == v1["verification_id"])

    # missing observed cells and null forecast cells are excluded from the mean
    gm = register(base_b, members=2, rows=1, cols=3, thresholds=[5.0])
    gmid = gm["group_id"]
    http("PUT", base_a, f"/groups/{gmid}/members/m1",
         {"rain": [[6.0, 6.0, 6.0]], "missing": [[0, 0, 1]], "revision": 1})
    http("PUT", base_b, f"/groups/{gmid}/members/m2",
         {"rain": [[0.0, 0.0, 0.0]], "missing": [[0, 0, 1]], "revision": 1})
    http("POST", base_a, f"/groups/{gmid}/publish")
    # product: p = [0.5, 0.5, null]; mark cell 0 missing in the observation,
    # cell 2 is null in the product -> only cell 1 contributes
    http("PUT", base_a, f"/groups/{gmid}/observations",
         {"rain": [[0.0, 9.0, 0.0]], "missing": [[1, 0, 0]], "revision": 1})
    vm = http("POST", base_b, f"/groups/{gmid}/verifications")[1]
    # only cell 1: p=0.5, observed outcome at thr5 = 1 -> (0.5-1)^2 = 0.25
    check("only valid observed + non-null forecast cells count",
          vm["thresholds"][0]["valid_cells"] == 1
          and approx(vm["thresholds"][0]["brier_score"], 0.25), str(vm))

    # every cell unusable -> empty (null) score, never a divide-by-zero
    gz = register(base_a, members=2, rows=1, cols=2, thresholds=[5.0])
    gzid = gz["group_id"]
    http("PUT", base_a, f"/groups/{gzid}/members/m1",
         {"rain": [[5.0, 9.0]], "missing": [[0, 1]], "revision": 1})
    http("PUT", base_b, f"/groups/{gzid}/members/m2",
         {"rain": [[10.0, 9.0]], "missing": [[0, 1]], "revision": 1})
    http("POST", base_a, f"/groups/{gzid}/publish")
    http("PUT", base_b, f"/groups/{gzid}/observations",
         {"rain": [[0.0, 0.0]], "missing": [[1, 1]], "revision": 1})
    vz = http("POST", base_a, f"/groups/{gzid}/verifications")[1]
    check("zero valid cells -> null brier score with valid_cells 0",
          vz["thresholds"][0]["brier_score"] is None
          and vz["thresholds"][0]["valid_cells"] == 0, str(vz))

    print("\n[16] member corrections cannot change the locked verification target")
    # correct a member, republish: a new current product exists, but the lock stays
    http("PUT", base_b, f"/groups/{gvid}/members/m1",
         {"rain": [[9.0, 9.0], [9.0, 9.0]], "revision": 2})
    pv2 = http("POST", base_b, f"/groups/{gvid}/publish")[1]
    check("new current product after member correction",
          pv2["product_id"] != lock_id and pv2["is_current"] is True)
    s, o2 = http("PUT", base_a, f"/groups/{gvid}/observations",
                 {"rain": [[1.0, 1.0], [1.0, 1.0]], "revision": 2})
    check("new observation revision inherits the original lock",
          s == 200 and o2["locked_product_id"] == lock_id, str(o2))
    s, d = http("GET", base_b, f"/groups/{gvid}/verifications/current")
    check("old verification invalidated immediately by observation correction",
          s == 404 and d["error"]["code"] == "NO_CURRENT_VERIFICATION", str(d))
    s, v2 = http("POST", base_a, f"/groups/{gvid}/verifications")
    check("recomputed verification still evaluates the locked (old) product",
          s == 200 and v2["locked_product_id"] == lock_id and v2["obs_revision"] == 2, str(v2))
    # outcomes all 0 at thr 5 (obs rain = 1): score against locked probs5
    brier5_obs2 = sum((p - 0.0) ** 2 for p in probs5) / 4
    v2by = {t["threshold"]: t for t in v2["thresholds"]}
    check("recomputed score matches locked product, not the new one",
          approx(v2by[5.0]["brier_score"], brier5_obs2), str(v2by[5.0]))
    s, listing = http("GET", base_b, f"/groups/{gvid}/verifications")
    check("verification history retained with exactly one current row",
          len(listing["verifications"]) == 2
          and sum(1 for x in listing["verifications"] if x["is_current"]) == 1, str(listing))
    check("all historical verifications reference the same locked product",
          all(x["locked_product_id"] == lock_id for x in listing["verifications"]))
    old_vid = next(x["verification_id"] for x in listing["verifications"] if not x["is_current"])
    s, hist = http("GET", base_a, f"/groups/{gvid}/verifications/{old_vid}")
    check("historical verification queryable by id",
          s == 200 and hist["is_current"] is False and hist["obs_revision"] == 1)
    s, d = http("GET", base_b, f"/groups/{gmid}/verifications/{old_vid}")
    check("verification id from another group -> 404",
          s == 404 and d["error"]["code"] == "VERIFICATION_NOT_FOUND")
    s, d = http("GET", base_a, f"/groups/{gvid}/verifications/999999")
    check("unknown verification id -> 404",
          s == 404 and d["error"]["code"] == "VERIFICATION_NOT_FOUND")
    st = http("GET", base_a, f"/groups/{gvid}")[1]
    check("group status exposes observation/verification state",
          st["latest_observation_revision"] == 2 and st["locked_product_id"] == lock_id
          and st["current_verification_id"] == v2["verification_id"], str(st))

    print("\n[17] interleaved observation correction vs in-flight verification")
    gw = register(base_a, members=3, rows=64, cols=64, thresholds=[5.0, 20.0])
    gwid = gw["group_id"]
    for i, inst in enumerate((base_a, base_b, base_a)):
        name = f"m{i + 1}"
        grid = [[float((i + 1) * (r + c) % 25) for c in range(64)] for r in range(64)]
        s, _ = http("PUT", inst, f"/groups/{gwid}/members/{name}",
                    {"rain": grid, "revision": 1})
        assert s == 200
    http("POST", base_a, f"/groups/{gwid}/publish")
    http("PUT", base_b, f"/groups/{gwid}/observations",
         {"rain": [[3.0] * 64 for _ in range(64)], "revision": 1})
    vres: Dict[str, Any] = {}

    def slow_verify() -> None:
        vres["result"] = http("POST", base_a, f"/groups/{gwid}/verifications")

    t = threading.Thread(target=slow_verify)
    t.start()
    time.sleep(0.15)  # let verification enter its slow compute phase
    t0 = time.time()
    s, corr = http("PUT", base_b, f"/groups/{gwid}/observations",
                   {"rain": [[8.0] * 64 for _ in range(64)], "revision": 2})
    write_elapsed = time.time() - t0
    check("observation correction accepted during in-flight verification", s == 200, str(corr))
    if hooks_enabled:
        check("observation write stayed fast while verification computed",
              write_elapsed < 0.5, f"{write_elapsed:.2f}s")
    t.join()
    check("in-flight verification still returns 200 (retried on new revision)",
          vres["result"][0] == 200, str(vres["result"]))
    curw = http("GET", base_b, f"/groups/{gwid}/verifications/current")[1]
    check("current verification matches latest observation revision",
          curw["obs_revision"] == 2 and curw["is_current"] is True, str(curw))
    listw = http("GET", base_a, f"/groups/{gwid}/verifications")[1]["verifications"]
    check("only one current verification after interleaving",
          sum(1 for x in listw if x["is_current"]) == 1
          and [x for x in listw if x["is_current"]][0]["verification_id"] == curw["verification_id"])
    check("no stale revision-1 verification left current",
          all(x["is_current"] is False for x in listw if x["obs_revision"] == 1))

    print("\n[18] verification compute failure leaves no half result; parallel idempotency")
    gf = register(base_a, members=2, rows=8, cols=8, thresholds=[5.0, 10.0])
    gfid = gf["group_id"]
    http("PUT", base_a, f"/groups/{gfid}/members/m1",
         {"rain": [[6.0] * 8 for _ in range(8)], "revision": 1})
    http("PUT", base_b, f"/groups/{gfid}/members/m2",
         {"rain": [[4.0] * 8 for _ in range(8)], "revision": 1})
    http("POST", base_a, f"/groups/{gfid}/publish")
    http("PUT", base_b, f"/groups/{gfid}/observations",
         {"rain": [[5.0] * 8 for _ in range(8)], "revision": 1})
    if hooks_enabled:
        with open(verify_fail_marker, "w") as fh:
            fh.write("fail")
        s, d = http("POST", base_a, f"/groups/{gfid}/verifications")
        os.remove(verify_fail_marker)
        check("injected verification failure surfaces as 500 COMPUTE_FAILED",
              s == 500 and d["error"]["code"] == "COMPUTE_FAILED", str(d))
        s, d = http("GET", base_b, f"/groups/{gfid}/verifications/current")
        check("no current verification after failed compute", s == 404)
        s, listing = http("GET", base_a, f"/groups/{gfid}/verifications")
        check("no verification rows at all after failed compute",
              listing["verifications"] == [])
        s, vf = http("POST", base_b, f"/groups/{gfid}/verifications")
        check("verification succeeds after failure cleared",
              s == 200 and vf["is_current"] is True)
    else:
        print("  SKIP  verification failure injection (RAIN_TEST_VERIFY_FAIL_MARKER not configured)")

    # parallel verifications of the same observation revision collapse to one row
    gp = register(base_b, members=2, rows=8, cols=8, thresholds=[5.0])
    gpid = gp["group_id"]
    http("PUT", base_a, f"/groups/{gpid}/members/m1",
         {"rain": [[6.0] * 8 for _ in range(8)], "revision": 1})
    http("PUT", base_b, f"/groups/{gpid}/members/m2",
         {"rain": [[0.0] * 8 for _ in range(8)], "revision": 1})
    http("POST", base_a, f"/groups/{gpid}/publish")
    http("PUT", base_a, f"/groups/{gpid}/observations",
         {"rain": [[1.0] * 8 for _ in range(8)], "revision": 1})
    vouts: Dict[str, Any] = {}

    def verify(instance: str) -> None:
        vouts[instance] = http("POST", instance, f"/groups/{gpid}/verifications")

    threads = [threading.Thread(target=verify, args=(base_a,)),
               threading.Thread(target=verify, args=(base_b,))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    vids = {body["verification_id"] for st, body in vouts.values() if st == 200}
    check("both parallel verifications succeed",
          all(st == 200 for st, _ in vouts.values()))
    check("parallel verifications collapse to one idempotent row",
          len(vids) == 1, str(vids))
    check("exactly one verification row for the observation revision",
          len(http("GET", base_a, f"/groups/{gpid}/verifications")[1]["verifications"]) == 1)

    print("\n[19] slow verification does not block another time's member ingestion")
    gq = register(base_a, members=2, rows=16, cols=16, thresholds=[5.0])
    gqid = gq["group_id"]
    http("PUT", base_a, f"/groups/{gqid}/members/m1",
         {"rain": [[6.0] * 16 for _ in range(16)], "revision": 1})
    http("PUT", base_a, f"/groups/{gqid}/members/m2",
         {"rain": [[6.0] * 16 for _ in range(16)], "revision": 1})
    http("POST", base_a, f"/groups/{gqid}/publish")
    http("PUT", base_a, f"/groups/{gqid}/observations",
         {"rain": [[6.0] * 16 for _ in range(16)], "revision": 1})
    bgv = {}

    def slow_verify_other() -> None:
        bgv["result"] = http("POST", base_a, f"/groups/{gqid}/verifications")

    th = threading.Thread(target=slow_verify_other)
    th.start()
    time.sleep(0.15)
    t0 = time.time()
    gr = register(base_b, members=2, rows=3, cols=3, thresholds=[5.0])
    http("PUT", base_b, f"/groups/{gr['group_id']}/members/m1",
         {"rain": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], "revision": 1})
    s, _ = http("PUT", base_b, f"/groups/{gr['group_id']}/members/m2",
                {"rain": [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]], "revision": 1})
    ingest_elapsed = time.time() - t0
    check("other time ingestion succeeded during slow verification", s == 200)
    if hooks_enabled:
        check("other time ingestion stayed fast during verification",
              ingest_elapsed < 0.5, f"{ingest_elapsed:.2f}s")
    th.join()
    check("slow verification completed", bgv["result"][0] == 200, str(bgv))

    print("\n[20] Swagger / OpenAPI")
    s, openapi = http("GET", base_a, "/openapi.json")
    check("openapi.json served", s == 200 and "paths" in openapi)
    for path in ("/groups", "/groups/{group_id}/members/{member_id}",
                 "/groups/{group_id}/publish",
                 "/groups/{group_id}/products/current",
                 "/groups/{group_id}/observations",
                 "/groups/{group_id}/verifications",
                 "/groups/{group_id}/verifications/current",
                 "/groups/{group_id}/verifications/{verification_id}"):
        check(f"path {path} documented", path in openapi["paths"])
    req = urllib.request.Request(base_a + "/docs")
    with urllib.request.urlopen(req, timeout=10) as resp:
        check("Swagger UI served", resp.status == 200 and b"swagger" in resp.read().lower())


if __name__ == "__main__":
    sys.exit(main())
