from __future__ import annotations
import json
import os
import threading
import uuid
from typing import Dict, Any


class StatsStore:
    def __init__(self, path: str = "data/user_stats.json"):
        self.path = path
        self.lock = threading.RLock()
        d = os.path.dirname(self.path)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        self.data: Dict[str, Any] = self._load()
        # Normalize shape to {users, usernames, stats}
        if not isinstance(self.data, dict):
            self.data = {}
        if "users" not in self.data or "usernames" not in self.data or "stats" not in self.data:
            # Backwards-compat: old shape had top-level user_id -> stats
            old = self.data if self.data else {}
            self.data = {
                "users": {},
                "usernames": {},
                "stats": old if all(isinstance(v, dict) and set(v.keys()) & {"total", "correct", "wrong"} == set() for v in old.values()) else {},
                "items": {}
            }
        # Ensure new top-level keys
        self.data.setdefault("items", {})
        # users map may also hold optional per-user extras (e.g., codebooks)

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def _now_iso(self) -> str:
        import datetime as _dt
        return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def _parse_iso(self, ts: str):
        import datetime as _dt
        try:
            if ts.endswith("Z"):
                ts = ts[:-1]
            return _dt.datetime.fromisoformat(ts)
        except Exception:
            return None

    def _half_life_decay(self, mastery: float, last_seen_at: str | None, half_life_days: float = 7.0) -> float:
        if mastery <= 0:
            return 0.0
        if not last_seen_at:
            return mastery
        import math, datetime as _dt
        t0 = self._parse_iso(last_seen_at)
        if not t0:
            return mastery
        dt_days = ( _dt.datetime.utcnow() - t0 ).total_seconds() / 86400.0
        if dt_days <= 0:
            return mastery
        k = math.log(2.0) / max(half_life_days, 0.1)
        decayed = mastery * math.exp(-k * dt_days)
        return max(0.0, min(1.0, decayed))

    def record(self, user_id: str, skill_id: str, category_id: str, correct: bool, misconception_tag: str | None = None, template_id: str | None = None) -> Dict[str, Any]:
        with self.lock:
            stats = self.data.setdefault("stats", {})
            u = stats.setdefault(user_id, {"per_skill": {}, "per_category": {}})
            # Per-skill
            sk = u["per_skill"].setdefault(
                skill_id,
                {"correct": 0, "wrong": 0, "total": 0, "mastery": 0.0, "last_seen_at": None, "misconceptions": {}, "template_counts": {}}
            )
            if correct:
                sk["correct"] += 1
            else:
                sk["wrong"] += 1
            sk["total"] += 1
            if template_id:
                tc = sk.setdefault("template_counts", {})
                tc[template_id] = int(tc.get(template_id, 0)) + 1
            # Update mastery with decay + EMA update
            prev_m = float(sk.get("mastery", 0.0) or 0.0)
            prev_seen = sk.get("last_seen_at")
            decayed = self._half_life_decay(prev_m, prev_seen)
            result = 1.0 if correct else 0.0
            alpha = 0.7  # weight on previous mastery
            new_m = alpha * decayed + (1.0 - alpha) * result
            sk["mastery"] = max(0.0, min(1.0, new_m))
            sk["last_seen_at"] = self._now_iso()
            # Misconception tracking (only count on wrong answers)
            if (not correct) and misconception_tag:
                m = sk.setdefault("misconceptions", {})
                m[misconception_tag] = int(m.get(misconception_tag, 0)) + 1
            # Per-category
            cat = u["per_category"].setdefault(category_id, {"correct": 0, "wrong": 0, "total": 0})
            if correct:
                cat["correct"] += 1
            else:
                cat["wrong"] += 1
            cat["total"] += 1
            self._save()
            return u

    def get(self, user_id: str) -> Dict[str, Any]:
        with self.lock:
            return self.data.get("stats", {}).get(user_id, {"per_skill": {}, "per_category": {}})

    def upsert_user(self, username: str) -> Dict[str, str]:
        """Create or fetch a user_id for the given username (case-insensitive)."""
        if not username or not username.strip():
            raise ValueError("username required")
        uname = username.strip()
        key = uname.lower()
        with self.lock:
            users = self.data.setdefault("users", {})
            usernames = self.data.setdefault("usernames", {})
            if key in usernames:
                uid = usernames[key]
                # ensure stored username reflects latest casing
                users.setdefault(uid, {})["username"] = uname
                self._save()
                return {"user_id": uid, "username": uname}
            # create new
            uid = str(uuid.uuid4())
            usernames[key] = uid
            users[uid] = {"username": uname}
            # ensure stats entry exists
            self.data.setdefault("stats", {}).setdefault(uid, {"per_skill": {}, "per_category": {}})
            self._save()
            return {"user_id": uid, "username": uname}

    def profile(self, user_id: str) -> Dict[str, Any]:
        with self.lock:
            uname = self.data.get("users", {}).get(user_id, {}).get("username")
            stats = self.get(user_id)
            return {"user_id": user_id, "username": uname, "stats": stats}

    def get_by_name(self, username: str) -> Dict[str, Any] | None:
        key = username.strip().lower()
        with self.lock:
            uid = self.data.get("usernames", {}).get(key)
            if not uid:
                return None
            return self.profile(uid)

    # Convenience APIs for adaptivity
    def decayed_mastery(self, user_id: str, skill_id: str, half_life_days: float = 7.0) -> float:
        with self.lock:
            u = self.data.get("stats", {}).get(user_id)
            if not u:
                return 0.0
            sk = u.get("per_skill", {}).get(skill_id)
            if not sk:
                return 0.0
            return self._half_life_decay(float(sk.get("mastery", 0.0) or 0.0), sk.get("last_seen_at"), half_life_days)

    # ---------- Item-level aggregation ----------
    def record_item_event(
        self,
        item_id: str,
        skill_id: str,
        correct: bool | None = None,
        confidence: int | None = None,
        time_ms: int | None = None,
        prompt_id: str | None = None,
        seed: int | None = None,
    ) -> Dict[str, Any]:
        now = self._now_iso()
        with self.lock:
            items = self.data.setdefault("items", {})
            it = items.setdefault(item_id, {
                "skill_id": skill_id,
                "created_at": now,
                "last_seen_at": now,
                "delivered": 0,
                "correct": 0,
                "wrong": 0,
                "confidence_sum": 0,
                "confidence_n": 0,
                "time_sum_ms": 0,
                "time_n": 0,
                "flags": 0,
            })
            it["last_seen_at"] = now
            it["delivered"] += 1
            if correct is True:
                it["correct"] += 1
            elif correct is False:
                it["wrong"] += 1
            if isinstance(confidence, int) and 1 <= confidence <= 5:
                it["confidence_sum"] += confidence
                it["confidence_n"] += 1
            if isinstance(time_ms, int) and time_ms >= 0:
                it["time_sum_ms"] += time_ms
                it["time_n"] += 1
            if prompt_id is not None:
                it["prompt_id"] = prompt_id
            if seed is not None:
                it["seed"] = seed
            # Derived metrics
            delivered = int(it.get("delivered", 0))
            correct_n = int(it.get("correct", 0))
            flags = int(it.get("flags", 0))
            if delivered > 0:
                it["p_value"] = correct_n / max(1, delivered)
                it["flag_rate"] = flags / max(1, delivered)
            # Retirement rule (thresholds): after 100 deliveries, retire if flag_rate > 1% or p outside [0.1, 0.9]
            if delivered >= 100 and not it.get("retired"):
                p = float(it.get("p_value", 0.0) or 0.0)
                fr = float(it.get("flag_rate", 0.0) or 0.0)
                if fr > 0.01 or p < 0.1 or p > 0.9:
                    it["retired"] = True
                    it["retired_reason"] = "quality_guardrails"
            self._save()
            return it

    def flag_item(self, item_id: str) -> Dict[str, Any]:
        now = self._now_iso()
        with self.lock:
            it = self.data.setdefault("items", {}).setdefault(item_id, {
                "skill_id": None,
                "created_at": now,
                "last_seen_at": now,
                "delivered": 0,
                "correct": 0,
                "wrong": 0,
                "confidence_sum": 0,
                "confidence_n": 0,
                "time_sum_ms": 0,
                "time_n": 0,
                "flags": 0,
            })
            it["flags"] = int(it.get("flags", 0)) + 1
            it["last_seen_at"] = now
            self._save()
            return it

    # ---------- Per-user extras (e.g., anonymization codebooks) ----------
    def _user_rec(self, user_id: str) -> Dict[str, Any]:
        users = self.data.setdefault("users", {})
        return users.setdefault(user_id, {})

    def get_or_create_codebook(self, user_id: str, vocab: list[str]) -> Dict[str, str]:
        import random, hashlib
        with self.lock:
            u = self._user_rec(user_id)
            cb = u.get("codebook")
            if isinstance(cb, dict) and cb:
                return cb
            # Derive a deterministic seed per user_id
            h = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
            seed = int(h[:8], 16)
            rng = random.Random(seed)
            def code_token() -> str:
                # e.g., NARO-17, ZAB-04
                letters = ''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(4))
                num = rng.randrange(1, 99)
                return f"{letters}-{num:02d}"
            mapping: Dict[str, str] = {}
            for term in sorted(set(vocab)):
                mapping[term] = code_token()
            u["codebook"] = mapping
            self._save()
            return mapping

    def extend_codebook(self, user_id: str, tokens: list[str]) -> Dict[str, str]:
        import random, hashlib
        with self.lock:
            u = self._user_rec(user_id)
            cb: Dict[str, str] = u.setdefault("codebook", {})
            h = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
            seed = int(h[:8], 16)
            rng = random.Random(seed)
            def code_token() -> str:
                letters = ''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(4))
                num = rng.randrange(1, 99)
                return f"{letters}-{num:02d}"
            added = False
            for t in sorted(set(tokens)):
                if t and t not in cb:
                    cb[t] = code_token()
                    added = True
            if added:
                self._save()
            return cb

    def get_or_create_numeric_key(self, user_id: str) -> Dict[str, int]:
        import random, hashlib
        with self.lock:
            u = self._user_rec(user_id)
            nk = u.get("numeric_key")
            if isinstance(nk, dict) and {"a","b"} <= set(nk.keys()):
                return nk
            h = hashlib.sha256((user_id+"/num").encode("utf-8")).hexdigest()
            seed = int(h[:8], 16)
            rng = random.Random(seed)
            a = rng.randrange(2, 9)
            b = rng.randrange(1, 9)
            nk = {"a": a, "b": b}
            u["numeric_key"] = nk
            self._save()
            return nk
