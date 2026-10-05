"""Public API (modular monolith). Blueprint services map to: edge-gateway + sync-service (numbered offline batches),
acoustic-inference (sound classifier), tracking-service (collar fixes), habitat-service (occupancy, corridor), risk-service
(night-vehicle clusters, risk tiles) and patrol-optimizer. vision-inference and media-ingestion are not built: camera
detections arrive as the label and confidence an on-device classifier would send.

    uvicorn wl.api:app          python -m core.jobs wl.api      # the worker
"""
import datetime
import json
import pickle
import uuid
from typing import Literal

import numpy as np
from fastapi import Depends, Header
from fastapi.encoders import jsonable_encoder
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import engine, world
from .seed import TENANTS

READ = {"reserve:read", "jobs:read"}
FIELD = READ | {"sensors:sync", "patrols:plan", "media:infer"}
PERMISSIONS = {"researcher": READ, "ranger": FIELD, "warden": FIELD | {"scenario:control", "simulation:read", "audit:read"}}
app = create_app("wildlife-conservation-network", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
PERIODS = {"before": range(0, world.LOSS_DAY), "after": range(world.LOSS_DAY, world.DAYS)}
_cache = {}


def sounds():
    if "sounds" not in _cache:
        with db.tx() as c:
            row = c.execute("SELECT artifact, metrics FROM model_registry WHERE name = 'acoustic-classifier' AND state = 'production'").fetchone()
        _cache["sounds"] = (*pickle.loads(row["artifact"]), row["metrics"])          # our own training output
    return _cache["sounds"]


def classify(features):
    model, thr, _ = sounds()
    proba = model.predict_proba(np.array(features, dtype=float).reshape(-1, 5))
    gi, ei = list(model.classes_).index("gunshot"), list(model.classes_).index("engine")
    return [(str(model.classes_[p.argmax()]), float(p[gi]), float(p[ei]), dict(zip(model.classes_, (round(float(v), 4) for v in p)))) for p in proba], thr


def started(c):
    row = c.execute("SELECT started_at FROM reserve").fetchone()
    if not row:
        raise Problem(409, "no_reserve", "load the scenario first: POST /v1/scenarios/month")
    return row["started_at"]


# ---------------------------------------------------------------- edge-gateway + sync-service

class Obs(BaseModel):
    kind: Literal["animal", "vehicle", "sound"]
    day: int = Field(ge=0, le=400)
    hour: float = Field(ge=0, lt=24)
    label: str | None = Field(None, max_length=40)
    confidence: float | None = Field(None, ge=0, le=1)
    count: int = Field(1, ge=1, le=500)
    features: list[float] | None = Field(None, min_length=5, max_length=5, description="Sounds only: duration s, rise s, peak Hz, bandwidth Hz, level dB")


class SyncIn(BaseModel):
    seq: int = Field(ge=0, le=100000, description="The sensor's own batch number; a batch is stored once however often it is sent")
    observations: list[Obs] = Field(max_length=500)


def ingest_batch(c, t, sensor, seq, obs, t0, species):
    """Store one batch exactly once. -> (status, stored, arrived out of order)"""
    late = seq < sensor["last_seq"]
    if not c.execute("INSERT INTO sync_batches (sensor_id, seq, tenant_id, observations, arrived_out_of_order) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING seq", [sensor["id"], seq, t, len(obs), late]).fetchone():
        return "duplicate", 0, False
    snd = [o for o in obs if o["kind"] == "sound"]
    labels = iter(classify([o["features"] for o in snd])[0]) if snd else iter(())
    rows = []
    for k, o in enumerate(obs):
        lab, pg, pe, _ = next(labels) if o["kind"] == "sound" else (None, None, None, None)
        rows.append((uuid.uuid5(sensor["id"], f"{seq}-{k}"), t, sensor["id"], t0 + datetime.timedelta(days=o["day"], hours=o["hour"]), species.get(o["label"]), o["confidence"], o["count"], f"edge://{sensor['name']}/{seq}/{k}",
                     sensor["x"], sensor["y"], o["kind"], o["label"], o["day"], o["hour"], seq, json.dumps(o["features"]) if o["features"] else None, lab, pg if lab != "engine" else pe))
    db.many(c, """INSERT INTO observations (id, tenant_id, sensor_id, observed_at, species_id, confidence, count, media_uri, x, y, kind, label, day, hour, batch_seq, features, sound_label, sound_p)
                  VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", rows)       # plain inserts: a month is thousands of small batches in one transaction
    c.execute("UPDATE sensors SET last_seq = greatest(last_seq, %s) WHERE id = %s", [seq, sensor["id"]])
    sensor["last_seq"] = max(sensor["last_seq"], seq)
    return "stored", len(rows), late


@app.post("/v1/sensors/{sensor_id}/sync", tags=["sensors"], summary="Upload one numbered batch from a sensor. Safe to resend and to send out of order: each batch is stored once")
def sync(sensor_id: uuid.UUID, body: SyncIn, ctx: Ctx = Depends(auth("sensors:sync")), idem: str | None = IdemKey):
    def work(c):
        t0 = started(c)
        sensor = c.execute("SELECT id, name, sensor_type, x, y, last_seq FROM sensors WHERE id = %s FOR UPDATE", [sensor_id]).fetchone()
        if not sensor:
            raise Problem(404, "sensor_not_found")
        if any((o.kind == "sound") != (sensor["sensor_type"] == "acoustic") or (o.kind == "sound") != (o.features is not None) for o in body.observations):
            raise Problem(422, "wrong_kind_for_sensor", "cameras send animal and vehicle detections; microphones send sounds with five features")
        species = {r["name"]: r["id"] for r in c.execute("SELECT id, name FROM species")}
        status, stored, late = ingest_batch(c, ctx.tenant_id, sensor, body.seq, [o.model_dump() for o in body.observations], t0, species)
        have = {r["seq"] for r in c.execute("SELECT seq FROM sync_batches WHERE sensor_id = %s", [sensor_id])}
        return 200, {"sensor": sensor["name"], "seq": body.seq, "status": status, "stored": stored, "arrived_out_of_order": late, "missing_batches": [s for s in range(max(have) + 1) if s not in have][:50]}
    return run(ctx, idem, body, work)


@app.post("/v1/scenarios/month", status_code=202, tags=["scenario"], summary="(+) Load the synthetic reserve and replay a month of sensor uploads in the order they arrived, late and duplicate batches included (job)")
def load(ctx: Ctx = Depends(auth("scenario:control")), idem: str | None = IdemKey):
    return run(ctx, idem, {}, lambda c: (202, jobs.enqueue(c, ctx, "scenario.load", {})))


@jobs.handler("scenario.load")
def load_job(c, job):
    t = job["tenant_id"]
    name = c.execute("SELECT name FROM tenants WHERE id = %s", [t]).fetchone()["name"]
    w = world.build(next(s for n, s in TENANTS.values() if n == name))
    for table in ("patrol_events", "tracks", "animals", "observations", "sync_batches", "sensors", "species", "reserve"):
        c.execute(f"DELETE FROM {table}")                             # row-level security limits this to the tenant
    t0 = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=world.DAYS)).replace(hour=0, minute=0, second=0, microsecond=0)
    c.execute("INSERT INTO reserve VALUES (%s,%s,%s)", [t, t0, w["seed"]])
    species = {n: uuid.uuid5(t, f"species-{n}") for n in world.SPECIES}
    db.load(c, "species", ["id", "tenant_id", "name"], [(i, t, n) for n, i in species.items()])
    sensors = [{"id": uuid.uuid5(t, f"sensor-{i}"), "name": s["name"], "sensor_type": s["type"], "x": s["xy"][0], "y": s["xy"][1], "last_seq": -1} for i, s in enumerate(w["sensors"])]
    db.load(c, "sensors", ["id", "tenant_id", "sensor_type", "x", "y", "deployed_at", "name", "ordinal"], [(s["id"], t, s["sensor_type"], s["x"], s["y"], t0, s["name"], i) for i, s in enumerate(sensors)])
    aid = [uuid.uuid5(t, f"animal-{a['tag']}") for a in w["animals"]]
    db.load(c, "animals", ["id", "tenant_id", "species_id", "tag_id"], [(aid[i], t, species[a["species"]], a["tag"]) for i, a in enumerate(w["animals"])])
    db.load(c, "tracks", ["animal_id", "recorded_at", "x", "y", "source", "tenant_id", "day"], [(aid[x["animal"]], t0 + datetime.timedelta(days=x["day"], hours=x["hour"]), *x["xy"], "gps-collar", t, x["day"]) for x in w["tracks"]])
    batches = {}
    for o in w["observations"]:
        batches.setdefault((o["sensor"], o["day"]), []).append({k: o.get(k) for k in ("kind", "day", "hour", "label", "confidence", "count", "features")})
    out = {"batches_received": 0, "stored": 0, "duplicates_ignored": 0, "arrived_out_of_order": 0, "observations": 0}
    for s, day in w["uploads"]:
        status, n, late = ingest_batch(c, t, sensors[s], day, batches.get((s, day), []), t0, species)
        out["batches_received"] += 1
        out["stored" if status == "stored" else "duplicates_ignored"] += 1
        out["arrived_out_of_order"] += late
        out["observations"] += n
    kinds = {r["kind"]: r["n"] for r in c.execute("SELECT kind, count(*) AS n FROM observations GROUP BY 1")}
    gaps = c.execute("SELECT count(*) AS n FROM sensors s WHERE (SELECT count(*) FROM sync_batches b WHERE b.sensor_id = s.id) < s.last_seq + 1").fetchone()["n"]
    audit.record(c, Ctx(t, uuid.UUID(job["payload"]["actor_id"]), "worker", "system"), "scenario.loaded", "tenant", t, {k: out[k] for k in ("batches_received", "duplicates_ignored", "observations")})
    return {**out, "cameras": w["n_cameras"], "microphones": len(sensors) - w["n_cameras"], "days": world.DAYS, "collared_animals": len(aid), "collar_fixes": len(w["tracks"]), "by_kind": kinds, "sensors_with_missing_batches": gaps,
            "longest_offline": {"sensor": sensors[w["offline_sensor"]]["name"], "days": 5, "delivered_on_day": 20}}


# ---------------------------------------------------------------- acoustic-inference

class InferIn(BaseModel):
    kind: Literal["acoustic"] = Field(description="Only sounds: no image is analysed in this build")
    features: list[float] = Field(min_length=5, max_length=5, description="duration s, rise s, peak Hz, bandwidth Hz, level dB")


@app.post("/v1/media/infer", tags=["inference"], summary="Classify one sound from its five summary features, and say whether it would raise a gunshot alert")
def infer(body: InferIn, ctx: Ctx = Depends(auth("media:infer"))):
    if min(body.features) <= 0:
        raise Problem(422, "bad_features", "all five features must be positive")
    rows, thr = classify(body.features)
    label, pg, _, proba = rows[0]
    m = sounds()[2]
    return {"label": label, "probabilities": proba, "gunshot_probability": round(pg, 4), "alert_threshold": thr, "raises_alert": pg >= thr,
            "why_so_strict": f"A gunshot is assumed to be {m['assumed_gunshot_prevalence']:.1%} of sounds. At a 0.5 threshold about {m['at_0.5']['precision_expected_in_field']:.0%} of alerts would be real; at {thr}, about {m['chosen']['precision_expected_in_field']:.0%}, keeping {m['chosen']['recall']:.0%} of gunshots.",
            "model_version": engine.VERSION}


# ---------------------------------------------------------------- analysis over the month on record

def month(c, t):
    """Everything derived from the stored observations; recomputed only when a new batch has arrived."""
    version = c.execute("SELECT count(*) AS n, max(received_at) AS at FROM sync_batches").fetchone()
    if _cache.get(t, (None,))[0] == (version["n"], version["at"]):
        return _cache[t][1]
    cams = c.execute("SELECT id, name, x, y, ordinal FROM sensors WHERE sensor_type = 'camera' ORDER BY ordinal").fetchall()
    idx = {s["id"]: k for k, s in enumerate(cams)}
    xy = [(s["x"], s["y"]) for s in cams]
    obs = c.execute("SELECT sensor_id, kind, label, confidence, day, hour, x, y, sound_label, sound_p FROM observations").fetchall()
    thr = sounds()[1]
    heard = [((o["x"], o["y"]), engine.night_of(o["day"], o["hour"]), o["sound_label"], o["sound_p"], o["day"], o["hour"]) for o in obs
             if o["kind"] == "sound" and ((o["sound_label"] == "gunshot" and o["sound_p"] >= thr) or (o["sound_label"] == "engine" and o["sound_p"] >= 0.9 and engine.is_night(o["hour"])))]
    clusters, counts = engine.vehicle_clusters([(idx[o["sensor_id"]], o["day"], o["hour"], o["confidence"]) for o in obs if o["kind"] == "vehicle"], xy, [h[:3] for h in heard])
    for cl in clusters:
        cl["camera_names"] = [cams[k]["name"] for k in cl["cameras"]]
    herd = [(r["x"], r["y"]) for r in c.execute("SELECT x, y FROM tracks WHERE day >= %s ORDER BY recorded_at", [world.LOSS_DAY])][::8]
    risk = engine.risk_tiles(clusters, [h[0] for h in heard if h[2] == "gunshot"], herd)
    out = {"cams": cams, "idx": idx, "xy": xy, "obs": obs, "heard": heard, "clusters": clusters, "counts": counts, "risk": risk}
    _cache[t] = ((version["n"], version["at"]), out)
    return out


@app.get("/v1/species", tags=["habitat"], summary="(+) Species on record")
def species_list(ctx: Ctx = Depends(auth("reserve:read"))):
    with db.tx(ctx.tenant_id) as c:
        return jsonable_encoder({"items": c.execute("SELECT s.id, s.name, (SELECT count(*) FROM observations o WHERE o.species_id = s.id) AS detections FROM species s ORDER BY s.name").fetchall()})


@app.get("/v1/species/{species_id}/distribution", tags=["habitat"], summary="Which camera sites a species uses, before and after day 15: plain counts and an occupancy model that allows for missed detections")
def distribution(species_id: uuid.UUID, ctx: Ctx = Depends(auth("reserve:read"))):
    with db.tx(ctx.tenant_id) as c:
        sp = c.execute("SELECT name FROM species WHERE id = %s", [species_id]).fetchone()
        if not sp:
            raise Problem(404, "species_not_found")
        m = month(c, ctx.tenant_id)
    if ("distribution", sp["name"]) in m:                         # 60 bootstrap fits per period: computed once per data version
        return m["distribution", sp["name"]]
    dets = [(m["idx"][o["sensor_id"]], o["day"], o["confidence"]) for o in m["obs"] if o["kind"] == "animal" and o["label"] == sp["name"]]
    out = m["distribution", sp["name"]] = {"species": sp["name"], "site_confidence": engine.SITE_CONFIDENCE, "periods": {}}
    for name, days in PERIODS.items():
        y = engine.histories([(s, d) for s, d, conf in dets if conf >= engine.SITE_CONFIDENCE], len(m["cams"]), days)
        any_conf = engine.histories([(s, d) for s, d, _ in dets], len(m["cams"]), days)
        est = engine.occupancy(y)
        out["periods"][name] = {"days": [days[0], days[-1]], **est, "share_with_any_label_at_any_confidence": round(float((any_conf.sum(axis=1) > 0).mean()), 3),
                                "sites": [{"sensor": cam["name"], "x": cam["x"], "y": cam["y"], "days_detected": int(n)} for cam, n in zip(m["cams"], y.sum(axis=1)) if n]}
    return out


@app.get("/v1/corridor/shift", tags=["habitat"], summary="(+) Did the herd's crossing point move after the habitat loss? From camera detections, with a permutation test, and from the GPS collars")
def corridor(species: str = "elephant", ctx: Ctx = Depends(auth("reserve:read"))):
    with db.tx(ctx.tenant_id) as c:
        m = month(c, ctx.tenant_id)
        fixes = c.execute("SELECT t.day >= %s AS after, avg(t.x) AS x, count(*) AS n FROM tracks t JOIN animals a ON a.id = t.animal_id JOIN species s ON s.id = a.species_id WHERE s.name = %s GROUP BY 1", [world.LOSS_DAY, species]).fetchall()
    dets = [(o["x"], o["y"], o["day"]) for o in m["obs"] if o["kind"] == "animal" and o["label"] == species and o["confidence"] >= engine.MIN_CONFIDENCE]
    if ("shift", species) not in m:                               # 4,000 permutations: computed once per data version
        m["shift", species] = engine.shift([x for x, _, d in dets if d < world.LOSS_DAY], [x for x, _, d in dets if d >= world.LOSS_DAY])
    cam = m["shift", species]
    collar = {("after" if r["after"] else "before"): round(float(r["x"]), 2) for r in fixes}
    return jsonable_encoder({"species": species, "habitat_lost": {"day": world.LOSS_DAY, "block_km": {"x": world.LOSS[:2], "y": world.LOSS[2:]}}, "from_cameras": cam,
                             "from_collars": {**collar, "shift_km": round(collar["after"] - collar["before"], 2) if len(collar) == 2 else None, "fixes": sum(r["n"] for r in fixes)},
                             "detections": {"before": [[x, y] for x, y, d in dets if d < world.LOSS_DAY], "after": [[x, y] for x, y, d in dets if d >= world.LOSS_DAY]}})


@app.get("/v1/animals", tags=["tracking"], summary="(+) Collared animals")
def animals(ctx: Ctx = Depends(auth("reserve:read"))):
    with db.tx(ctx.tenant_id) as c:
        return jsonable_encoder({"items": c.execute("SELECT a.id, a.tag_id, s.name AS species, (SELECT count(*) FROM tracks t WHERE t.animal_id = a.id) AS fixes FROM animals a JOIN species s ON s.id = a.species_id ORDER BY a.tag_id").fetchall()})


@app.get("/v1/animals/{animal_id}/track", tags=["tracking"], summary="A collared animal's GPS fixes for a range of days")
def track(animal_id: uuid.UUID, from_day: int = 0, to_day: int = world.DAYS, ctx: Ctx = Depends(auth("reserve:read"))):
    with db.tx(ctx.tenant_id) as c:
        a = c.execute("SELECT a.tag_id, s.name AS species FROM animals a JOIN species s ON s.id = a.species_id WHERE a.id = %s", [animal_id]).fetchone()
        if not a:
            raise Problem(404, "animal_not_found")
        pts = c.execute("SELECT day, x, y, recorded_at FROM tracks WHERE animal_id = %s AND day BETWEEN %s AND %s ORDER BY recorded_at", [animal_id, from_day, to_day]).fetchall()
    return jsonable_encoder({**a, "fixes": len(pts), "points": pts})


@app.get("/v1/alerts/suspicious", tags=["risk"], summary="(+) Night-vehicle clusters that are far more than chance, with corroborating sounds, and the gunshot alerts")
def suspicious(ctx: Ctx = Depends(auth("reserve:read"))):
    with db.tx(ctx.tenant_id) as c:
        started(c)
        m = month(c, ctx.tenant_id)
        resend = c.execute("SELECT id FROM sensors WHERE sensor_type = 'acoustic' ORDER BY ordinal LIMIT 1").fetchone()["id"]      # a microphone id the demo uses to show a duplicate upload
    return jsonable_encoder({"clusters": m["clusters"], "false_positive_controls": {**m["counts"], "min_confidence": engine.MIN_CONFIDENCE, "rule": "a camera is flagged only for confident night-time vehicles on at least two nights with a chance probability under 0.001"},
                             "gunshot_alerts": [{"x": h[0][0], "y": h[0][1], "day": h[4], "hour": h[5], "probability": round(h[3], 3)} for h in m["heard"] if h[2] == "gunshot"],
                             "night_engine_sounds": sum(h[2] == "engine" for h in m["heard"]), "gunshot_threshold": sounds()[1], "resend_sensor": resend})


@app.get("/v1/risk/tiles", tags=["risk"], summary="Relative risk on a 1 km grid, from night-vehicle clusters, gunshot alerts and where the herd nears the boundary")
def tiles(ctx: Ctx = Depends(auth("reserve:read"))):
    with db.tx(ctx.tenant_id) as c:
        started(c)
        m = month(c, ctx.tenant_id)
    return {"size_km": world.SIZE, "tile_km": 1, "risk": [[round(float(v), 3) for v in row] for row in m["risk"]], "weights": {"night_vehicle_clusters": 0.5, "gunshot_alerts": 0.3, "herd_near_boundary": 0.2},
            "note": "The weights are a stated judgement, not learnt from outcomes: there is no record of incidents to learn from."}


# ---------------------------------------------------------------- patrol-optimizer

class PatrolIn(BaseModel):
    teams: int = Field(3, ge=1, le=8)
    hours: float = Field(8.0, ge=2, le=14)


@app.post("/v1/patrols/optimize", tags=["patrols"], summary="Routes for the ranger teams that cover the most risk within their hours, beside a fixed boundary patrol")
def optimize(body: PatrolIn, ctx: Ctx = Depends(auth("patrols:plan")), idem: str | None = IdemKey):
    def work(c):
        started(c)
        m = month(c, ctx.tenant_id)
        routes, base = engine.patrols(m["risk"], teams=body.teams, hours=body.hours), engine.perimeter(body.teams, body.hours)
        c.execute("DELETE FROM patrol_events WHERE event_type = 'planned_waypoint'")
        plan = uuid.uuid7()
        for k, path in enumerate(routes):
            for n, (x, y) in enumerate(path[1:-1]):
                c.execute("INSERT INTO patrol_events (id, tenant_id, ranger_team, event_type, x, y, metadata) VALUES (%s,%s,%s,'planned_waypoint',%s,%s,%s)", [uuid.uuid7(), ctx.tenant_id, f"Team {k + 1}", x, y, Jsonb({"plan": str(plan), "order": n + 1})])
        audit.record(c, ctx, "patrols.planned", "plan", plan, {"teams": body.teams, "hours": body.hours, "engine": engine.VERSION})
        describe = lambda paths, dwell: [{"team": f"Team {k + 1}", "waypoints": len(p) - 2, "km": round(sum(float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(p, p[1:])), 1), "hours": round(engine.path_hours(p, dwell), 2), "path": [[round(float(x), 2), round(float(y), 2)] for x, y in p]} for k, p in enumerate(paths)]      # noqa: E731
        return 200, {"plan_id": plan, "risk_covered": round(engine.covered(m["risk"], routes), 4), "routes": describe(routes, engine.DWELL_H),
                     "baseline": {"name": "each team follows a stretch of the boundary track", "risk_covered": round(engine.covered(m["risk"], base), 4), "routes": describe(base, 0)},
                     "assumptions": {"speed_kmh": engine.SPEED_KMH, "stop_at_each_waypoint_h": engine.DWELL_H, "a_patrol_covers_km": engine.COVER_KM}, "status": "proposed: for the warden to approve, not dispatched"}
    return run(ctx, idem, body, work)


@app.get("/v1/simulation/truth", tags=["simulation"], summary="(+) Demo only: what the simulator planted. Possible only because the reserve is synthetic")
def truth(ctx: Ctx = Depends(auth("simulation:read"))):
    with db.tx(ctx.tenant_id) as c:
        seed = c.execute("SELECT seed FROM reserve").fetchone()
    if not seed:
        raise Problem(409, "no_reserve", "load the scenario first: POST /v1/scenarios/month")
    w = world.build(seed["seed"])
    return {"night_vehicles": {"centre": world.HOTSPOT, "nights": world.VEHICLE_NIGHTS, "cameras": [s["name"] for s in w["sensors"][:3]]}, "gunshots": sum(o["truth"].get("sound") == "gunshot" for o in w["observations"]),
            "corridor_km_east": {"before": world.CORRIDOR[0], "after": world.CORRIDOR[1]},
            "sites_used": {sp: {"before": round(len(w["occupied"][sp, 0]) / w["n_cameras"], 3), "after": round(len(w["occupied"][sp, 1]) / w["n_cameras"], 3)} for sp in world.SPECIES}}
