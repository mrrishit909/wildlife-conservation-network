"""Tenants, demo tokens and the acoustic classifier (fitted on one set of synthetic sounds, scored on another, then
registered with its alert threshold). The reserve itself is loaded by POST /v1/scenarios/month.

    python -m wl.seed [--if-empty]
"""
import hashlib
import pickle
import sys
import uuid

from psycopg.types.json import Jsonb

from core import db

from . import engine, world

TENANTS = {"reserve": ("Mara Ridge Reserve", 13), "other": ("Second Reserve", 29)}       # name, world seed
ROLES = ["researcher", "ranger", "warden"]


def token(tenant, role):
    """Demo credentials for local use only."""
    return f"{tenant}-{role}-demo"


def tenant_id(key):
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"{key}.wildlife-network.example")


def train():
    test = world.sounds(202, 30000)                      # large, so that a false-alarm rate of one in ten thousand is measurable
    model, thr, metrics = engine.train_sounds(world.sounds(101), test)
    rule = [k for x, k in test if engine.loud_and_short(x)]
    metrics["baseline"] = {"name": "short and loud", "precision": round(rule.count("gunshot") / max(1, len(rule)), 4), "recall": round(rule.count("gunshot") / sum(k == "gunshot" for _, k in test), 4)}
    return model, thr, metrics


def main(if_empty=False):
    db.migrate()
    with db.tx() as c:
        if if_empty and c.execute("SELECT 1 FROM tenants LIMIT 1").fetchone():
            return print("already seeded")
    model, thr, metrics = train()
    out = {"metrics": metrics}
    with db.tx() as c:
        c.execute("INSERT INTO model_registry (name, version, data_snapshot, metrics, baseline, artifact) VALUES ('acoustic-classifier',%s,%s,%s,%s,%s)",
                  [engine.VERSION, "synthetic sound features, seeds 101 (fit) and 202 (score)", Jsonb(metrics), Jsonb(metrics["baseline"]), pickle.dumps((model, thr))])
        for key, (name, _) in TENANTS.items():
            t = tenant_id(key)
            c.execute("INSERT INTO tenants (id, name) VALUES (%s,%s)", [t, name])
            for role in ROLES:
                c.execute("INSERT INTO api_tokens VALUES (%s,%s,%s,%s,%s)", [hashlib.sha256(token(key, role).encode()).hexdigest(), t, uuid.uuid5(t, role), f"{role} ({name})", role])
            out[key] = t
    print("classifier registered:", metrics)
    print("demo tokens:", ", ".join(token(t, r) for t in TENANTS for r in ROLES))
    return out


if __name__ == "__main__":
    main("--if-empty" in sys.argv)
