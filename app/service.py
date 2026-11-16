"""Business logic: registration, member submission and product publishing.

Concurrency model (two API processes sharing one SQLite file)
----------------------------------------------------------------
* Every write happens inside a short ``BEGIN IMMEDIATE`` transaction, so only
  one writer exists at a time. Conflicting writers get ``SQLITE_BUSY`` after
  the busy timeout and receive HTTP 409/503 instead of corrupting state.
* Publishing reads a snapshot, runs the (potentially slow) NumPy computation
  *outside* any write transaction, then re-validates the revision set inside
  an immediate write transaction. The slow CPU work therefore never blocks
  other forecast times from ingesting members.
* A product row and its threshold rows are inserted in one transaction; any
  failure rolls the whole thing back, so no half product can ever be visible.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import timezone
from typing import Any, Dict, List, Tuple

from .db import get_conn
from .errors import ApiError
from .schemas import RegisterRequest, SubmitRequest
from .storage import (
    compute_probabilities,
    content_hash,
    decode_float,
    decode_int,
    decode_mask,
    encode_blob,
    matrix_to_json,
    to_arrays,
)

# --- test hooks (disabled unless explicitly enabled; never on by default) ---
PUBLISH_DELAY_S = float(os.environ.get("RAIN_TEST_PUBLISH_DELAY_S", "0"))
FAIL_MARKER = os.environ.get("RAIN_TEST_FAIL_MARKER", "")


def _is_busy(exc: Exception) -> bool:
    return isinstance(exc, Exception) and "SQLITE_BUSY" in str(exc)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def register_group(payload: RegisterRequest) -> Dict[str, Any]:
    group_id = uuid.uuid4().hex
    thresholds_json = json.dumps(payload.thresholds)
    with get_conn() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO groups (group_id, forecast_time, rows, cols, thresholds)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    group_id,
                    payload.forecast_time.astimezone(timezone.utc).isoformat(),
                    payload.rows,
                    payload.cols,
                    thresholds_json,
                ),
            )
            conn.executemany(
                "INSERT INTO members (group_id, member_idx, member_id) VALUES (?, ?, ?)",
                [(group_id, idx, name) for idx, name in enumerate(payload.member_names)],
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return {
        "group_id": group_id,
        "forecast_time": payload.forecast_time.astimezone(timezone.utc).isoformat(),
        "rows": payload.rows,
        "cols": payload.cols,
        "members": [
            {"member_idx": idx, "member_id": name}
            for idx, name in enumerate(payload.member_names)
        ],
        "thresholds": payload.thresholds,
    }


def _get_group(conn, group_id: str) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM groups WHERE group_id = ?", (group_id,)
    ).fetchone()
    if row is None:
        raise ApiError(404, "GROUP_NOT_FOUND", f"group {group_id!r} is not registered")
    d = dict(row)
    d["thresholds"] = json.loads(d["thresholds"])
    return d


def _get_member(conn, group_id: str, member_id: str) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT member_idx, member_id FROM members WHERE group_id = ? AND member_id = ?",
        (group_id, member_id),
    ).fetchone()
    if row is None:
        raise ApiError(404, "MEMBER_NOT_FOUND", f"member {member_id!r} is not part of this group")
    return dict(row)


# --------------------------------------------------------------------------- #
# Member submission
# --------------------------------------------------------------------------- #
def submit_member(group_id: str, member_id: str, payload: SubmitRequest) -> Dict[str, Any]:
    with get_conn() as conn:
        group = _get_group(conn, group_id)
        member = _get_member(conn, group_id, member_id)
        rain_flat, miss_flat = to_arrays(
            payload.rain, payload.missing, group["rows"], group["cols"]
        )
        digest = content_hash(rain_flat, miss_flat)
        rain_blob = encode_blob(rain_flat)
        miss_blob = encode_blob(miss_flat)
        idx = member["member_idx"]

        try:
            conn.execute("BEGIN IMMEDIATE")
            prev = conn.execute(
                "SELECT revision, content_hash FROM member_revisions"
                " WHERE group_id = ? AND member_idx = ?"
                " ORDER BY revision DESC LIMIT 1",
                (group_id, idx),
            ).fetchone()
            duplicate = False
            if prev is not None:
                if payload.revision < prev["revision"]:
                    conn.execute("ROLLBACK")
                    raise ApiError(
                        409,
                        "REVISION_CONFLICT",
                        f"revision {payload.revision} is not greater than latest {prev['revision']}",
                    )
                if payload.revision == prev["revision"]:
                    if prev["content_hash"] == digest:
                        duplicate = True  # 同版同文重送：幂等，不重复入库
                    else:
                        conn.execute("ROLLBACK")
                        raise ApiError(
                            409,
                            "REVISION_CONTENT_CONFLICT",
                            f"revision {payload.revision} already exists with different content (异文拒收)",
                        )
            if not duplicate:
                conn.execute(
                    "INSERT INTO member_revisions"
                    " (group_id, member_idx, revision, content_hash, rain_blob, miss_blob)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (group_id, idx, payload.revision, digest, rain_blob, miss_blob),
                )
                # 更正成员：旧产品立即过期（历史行保留可查）
                conn.execute(
                    "UPDATE products SET is_current = 0 WHERE group_id = ? AND is_current = 1",
                    (group_id,),
                )
            conn.execute("COMMIT")
        except ApiError:
            raise
        except Exception:
            conn.execute("ROLLBACK")
            raise

    return {
        "group_id": group_id,
        "member_id": member_id,
        "member_idx": idx,
        "revision": payload.revision,
        "content_hash": digest,
        "duplicate": duplicate,
    }


# --------------------------------------------------------------------------- #
# Publishing
# --------------------------------------------------------------------------- #
def _read_snapshot(conn, group_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[int, int]]:
    group = _get_group(conn, group_id)
    member_rows = conn.execute(
        "SELECT member_idx, member_id FROM members WHERE group_id = ? ORDER BY member_idx",
        (group_id,),
    ).fetchall()
    latest: List[Dict[str, Any]] = []
    revision_map: Dict[int, int] = {}
    for m in member_rows:
        rev = conn.execute(
            "SELECT revision FROM member_revisions"
            " WHERE group_id = ? AND member_idx = ? ORDER BY revision DESC LIMIT 1",
            (group_id, m["member_idx"]),
        ).fetchone()
        if rev is None:
            raise ApiError(
                409,
                "MEMBERS_INCOMPLETE",
                f"member {m['member_id']!r} has no submission; all members must be present before publishing",
            )
        revision_map[int(m["member_idx"])] = int(rev["revision"])
        latest.append({"member_idx": int(m["member_idx"]), "member_id": m["member_id"]})
    return group, latest, revision_map


def _canonical_revision_map(revision_map: Dict[int, int]) -> str:
    return json.dumps(revision_map, sort_keys=True, separators=(",", ":"))


def publish_product(group_id: str) -> Dict[str, Any]:
    # Bounded outer loop: each iteration is one consistent revision-set attempt.
    for attempt in range(1, 4):
        # 1) snapshot under a read transaction (no write lock held during compute)
        with get_conn() as conn:
            try:
                conn.execute("BEGIN")
                group, members, revision_map = _read_snapshot(conn, group_id)
                revs = conn.execute(
                    "SELECT group_id, member_idx, revision, rain_blob, miss_blob"
                    " FROM member_revisions WHERE group_id = ?",
                    (group_id,),
                ).fetchall()
                existing = conn.execute(
                    "SELECT product_id FROM products"
                    " WHERE group_id = ? AND revision_map = ? AND is_current = 1",
                    (group_id, _canonical_revision_map(revision_map)),
                ).fetchone()
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        if existing is not None:
            return _load_product(existing["product_id"])

        wanted = {
            (int(r["member_idx"]), int(r["revision"])): (
                decode_float(r["rain_blob"], group["rows"] * group["cols"]),
                decode_mask(r["miss_blob"], group["rows"] * group["cols"]),
            )
            for r in revs
        }
        rains = [wanted[(idx, revision_map[idx])][0] for idx in sorted(revision_map)]
        misses = [wanted[(idx, revision_map[idx])][1] for idx in sorted(revision_map)]

        # 2) slow compute happens OUTSIDE the write lock: other groups keep ingesting
        if PUBLISH_DELAY_S > 0:
            time.sleep(PUBLISH_DELAY_S)
        if FAIL_MARKER and os.path.exists(FAIL_MARKER):
            # deterministic failure injection for atomicity self-tests
            raise ApiError(500, "COMPUTE_FAILED", "injected computation failure (test hook)")
        try:
            per_threshold = compute_probabilities(rains, misses, group["thresholds"])
        except ApiError:
            raise
        except Exception as exc:  # calculation failure must not leave half a result
            raise ApiError(500, "COMPUTE_FAILED", f"probability computation failed: {exc}") from exc

        canonical = _canonical_revision_map(revision_map)

        # 3) re-validate under the write lock and persist atomically
        write_attempts = 0
        should_retry_snapshot = False
        with get_conn() as conn:
            while True:
                write_attempts += 1
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    latest_map: Dict[int, int] = {}
                    for m in members:
                        rev = conn.execute(
                            "SELECT revision FROM member_revisions"
                            " WHERE group_id = ? AND member_idx = ? ORDER BY revision DESC LIMIT 1",
                            (group_id, m["member_idx"]),
                        ).fetchone()
                        if rev is None:
                            conn.execute("ROLLBACK")
                            raise ApiError(409, "MEMBERS_INCOMPLETE", "member disappeared before publish commit")
                        latest_map[int(m["member_idx"])] = int(rev["revision"])

                    if latest_map != revision_map:
                        # a correction landed while we were computing: this product
                        # would be stale; retry from the new consistent revision set
                        conn.execute("ROLLBACK")
                        if attempt >= 3:
                            raise ApiError(
                                409,
                                "REVISION_CHANGED",
                                "member revisions changed during publish; retry the request",
                            )
                        should_retry_snapshot = True
                        break

                    row = conn.execute(
                        "SELECT product_id, is_current FROM products"
                        " WHERE group_id = ? AND revision_map = ?",
                        (group_id, canonical),
                    ).fetchone()
                    if row is not None and int(row["is_current"]) == 1:
                        conn.execute("ROLLBACK")
                        return _load_product(int(row["product_id"]))

                    # supersede every previous product of this group (history retained)
                    conn.execute(
                        "UPDATE products SET is_current = 0 WHERE group_id = ? AND is_current = 1",
                        (group_id,),
                    )
                    if row is not None:  # same revision map existed but was superseded: revive it
                        product_id = int(row["product_id"])
                        conn.execute(
                            "UPDATE products SET is_current = 1 WHERE product_id = ?", (product_id,)
                        )
                    else:
                        cur = conn.execute(
                            "INSERT INTO products (group_id, revision_map, is_current)"
                            " VALUES (?, ?, 1)",
                            (group_id, canonical),
                        )
                        product_id = int(cur.lastrowid)
                        conn.executemany(
                            "INSERT INTO product_thresholds"
                            " (product_id, threshold, probability_blob, denom_blob)"
                            " VALUES (?, ?, ?, ?)",
                            [
                                (
                                    product_id,
                                    float(thr),
                                    encode_blob(per_threshold[float(thr)][0]),
                                    encode_blob(per_threshold[float(thr)][1]),
                                )
                                for thr in group["thresholds"]
                            ],
                        )
                    conn.execute("COMMIT")
                    return _load_product(product_id)
                except ApiError:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    raise
                except Exception as exc:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    if _is_busy(exc) and write_attempts < 3:
                        time.sleep(0.05 * write_attempts)
                        continue
                    if _is_busy(exc):
                        raise ApiError(503, "DATABASE_BUSY", "database locked by another writer; retry") from exc
                    raise
        if should_retry_snapshot:
            continue

    raise ApiError(409, "REVISION_CHANGED", "member revisions kept changing during publish; retry the request")


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def _load_product(product_id: int) -> Dict[str, Any]:
    with get_conn() as conn:
        prow = conn.execute(
            "SELECT * FROM products WHERE product_id = ?", (product_id,)
        ).fetchone()
        if prow is None:
            raise ApiError(404, "PRODUCT_NOT_FOUND", f"product {product_id} not found")
        group = _get_group(conn, prow["group_id"])
        rows, cols = group["rows"], group["cols"]
        n = rows * cols
        revision_map = {int(k): v for k, v in json.loads(prow["revision_map"]).items()}
        member_rows = conn.execute(
            "SELECT member_idx, member_id FROM members WHERE group_id = ? ORDER BY member_idx",
            (prow["group_id"],),
        ).fetchall()
        thresholds_out: List[Dict[str, Any]] = []
        for trow in conn.execute(
            "SELECT threshold, probability_blob, denom_blob"
            " FROM product_thresholds WHERE product_id = ? ORDER BY threshold",
            (product_id,),
        ):
            prob = decode_float(trow["probability_blob"], n)
            denom = decode_int(trow["denom_blob"], n)
            thresholds_out.append(
                {
                    "threshold": float(trow["threshold"]),
                    "probability": matrix_to_json(prob, rows, cols),
                    "denominator": denom.reshape(rows, cols).tolist(),
                }
            )
        member_revisions = [
            {
                "member_idx": int(m["member_idx"]),
                "member_id": m["member_id"],
                "revision": int(revision_map[int(m["member_idx"])]),
            }
            for m in member_rows
        ]
        return {
            "product_id": int(prow["product_id"]),
            "group_id": prow["group_id"],
            "forecast_time": group["forecast_time"],
            "is_current": bool(prow["is_current"]),
            "created_at": prow["created_at"],
            "revision_map": revision_map,
            "members_used": member_revisions,
            "thresholds": thresholds_out,
        }


def get_current_product(group_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        _get_group(conn, group_id)  # 404 on unknown group
        row = conn.execute(
            "SELECT product_id FROM products WHERE group_id = ? AND is_current = 1",
            (group_id,),
        ).fetchone()
        if row is None:
            raise ApiError(
                404,
                "NO_CURRENT_PRODUCT",
                "no current product for this group (publish first, or the previous product was superseded)",
            )
        product_id = int(row["product_id"])
    return _load_product(product_id)


def list_products(group_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        _get_group(conn, group_id)
        rows = conn.execute(
            "SELECT product_id, revision_map, is_current, created_at"
            " FROM products WHERE group_id = ? ORDER BY product_id",
            (group_id,),
        ).fetchall()
        return {
            "group_id": group_id,
            "products": [
                {
                    "product_id": int(r["product_id"]),
                    "revision_map": {int(k): v for k, v in json.loads(r["revision_map"]).items()},
                    "is_current": bool(r["is_current"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ],
        }


def get_product(group_id: str, product_id: int) -> Dict[str, Any]:
    product = _load_product(product_id)
    if product["group_id"] != group_id:
        raise ApiError(404, "PRODUCT_NOT_FOUND", f"product {product_id} does not belong to this group")
    return product


def get_group_status(group_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        group = _get_group(conn, group_id)
        members = conn.execute(
            "SELECT member_idx, member_id FROM members WHERE group_id = ? ORDER BY member_idx",
            (group_id,),
        ).fetchall()
        out_members = []
        complete = True
        for m in members:
            rev = conn.execute(
                "SELECT revision, content_hash, created_at FROM member_revisions"
                " WHERE group_id = ? AND member_idx = ? ORDER BY revision DESC LIMIT 1",
                (group_id, m["member_idx"]),
            ).fetchone()
            if rev is None:
                complete = False
                out_members.append(
                    {"member_idx": int(m["member_idx"]), "member_id": m["member_id"], "latest_revision": None}
                )
            else:
                out_members.append(
                    {
                        "member_idx": int(m["member_idx"]),
                        "member_id": m["member_id"],
                        "latest_revision": int(rev["revision"]),
                        "content_hash": rev["content_hash"],
                        "updated_at": rev["created_at"],
                    }
                )
        current = conn.execute(
            "SELECT product_id FROM products WHERE group_id = ? AND is_current = 1",
            (group_id,),
        ).fetchone()
        return {
            "group_id": group_id,
            "forecast_time": group["forecast_time"],
            "rows": group["rows"],
            "cols": group["cols"],
            "thresholds": group["thresholds"],
            "members": out_members,
            "all_members_present": complete,
            "current_product_id": int(current["product_id"]) if current else None,
        }


def list_groups() -> Dict[str, Any]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT group_id, forecast_time, rows, cols, thresholds, created_at"
            " FROM groups ORDER BY created_at"
        ).fetchall()
        return {
            "groups": [
                {
                    "group_id": r["group_id"],
                    "forecast_time": r["forecast_time"],
                    "rows": r["rows"],
                    "cols": r["cols"],
                    "thresholds": json.loads(r["thresholds"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        }
