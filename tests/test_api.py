"""Integration, end-to-end and security tests against a real Postgres."""
import pytest

from core import db, jobs, scenario
from wl import engine, world

from .conftest import bearer

WARDEN, RANGER, RESEARCH, OTHER = bearer("reserve", "warden"), bearer("reserve", "ranger"), bearer("reserve", "researcher"), bearer("other", "warden")
ZERO = "00000000-0000-0000-0000-000000000000"


def test_nothing_works_before_the_reserve_is_loaded(client):
    assert client.get("/v1/risk/tiles", headers=RANGER).json()["code"] == "no_reserve"


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


@pytest.fixture(scope="module")
def truth():
    return world.build(13)


def one(sql, params=()):
    with db.tx() as c:
        return c.execute(sql, params).fetchone()


def test_demo_journey_matches_the_generated_reserve(demo, truth):
    load = demo["load"]["result"]
    batches = {(s, d) for s, d in truth["uploads"]}
    assert load["observations"] == len(truth["observations"]) and load["stored"] == len(batches) and load["duplicates_ignored"] == len(truth["uploads"]) - len(batches)
    assert load["arrived_out_of_order"] > 30 and load["sensors_with_missing_batches"] == 0 and load["by_kind"]["sound"] == sum(o["kind"] == "sound" for o in truth["observations"])
    leo = demo["leopard"]
    assert leo["species"] == "leopard"
    for name, per in (("before", 0), ("after", 1)):
        p, true = leo["periods"][name], len(truth["occupied"]["leopard", per]) / 60
        assert p["share_with_any_label_at_any_confidence"] > 0.7 > true and p["naive"] <= p["psi"] and p["low"] <= p["psi"] <= p["high"] and abs(p["naive"] - true) < 0.15
        assert demo["truth"]["sites_used"]["leopard"][name] == round(true, 3)
    sh = demo["shift"]
    assert sh["from_cameras"]["p_value"] < 0.01 and abs(sh["from_cameras"]["shift_km"] - 6.5) < 1.5 and abs(sh["from_collars"]["shift_km"] - 6.5) < 0.5
    sus = demo["sus"]
    assert len(sus["clusters"]) == 1
    cl = sus["clusters"][0]
    assert set(cl["camera_names"]) <= {"CAM-01", "CAM-02", "CAM-03"} and len(set(cl["nights"]) & set(world.VEHICLE_NIGHTS)) >= 4 and len(set(cl["nights"]) - set(world.VEHICLE_NIGHTS)) <= 1 and cl["confidence"] == "high"      # one stray confident mislabel is allowed
    assert abs(cl["centre"][0] - world.HOTSPOT[0]) < 2 and abs(cl["centre"][1] - world.HOTSPOT[1]) < 2 and cl["corroborating_sounds"]
    assert sus["false_positive_controls"]["cameras_with_any"] > 10 * sus["false_positive_controls"]["cameras_flagged"] > 0
    assert all(g["day"] in (21, 22, 25, 26) for g in sus["gunshot_alerts"]) and len(sus["gunshot_alerts"]) <= demo["truth"]["gunshots"] == 3
    tiles = demo["tiles"]["risk"]
    assert len(tiles) == 24 and max(max(r) for r in tiles) == 1.0 and tiles[2][21] > 0.8 > tiles[20][3]
    pat = demo["patrol"]
    assert pat["risk_covered"] > pat["baseline"]["risk_covered"] and all(r["hours"] <= 8.01 for r in pat["routes"] + pat["baseline"]["routes"]) and "proposed" in pat["status"]
    assert demo["sensors"] == {"sensor": "MIC-01", "seq": 4, "status": "duplicate", "stored": 0, "arrived_out_of_order": False, "missing_batches": []}
    assert demo["infer"]["label"] == "branch" and not demo["infer"]["raises_alert"] and demo["infer"]["alert_threshold"] > 0.8
    assert demo["audit"]["chain_valid"] and {"scenario.loaded", "patrols.planned"} <= {e["action"] for e in demo["audit"]["events"]}


def test_a_batch_is_stored_once_whatever_order_it_arrives_in(client, demo):
    cam = one("SELECT s.id, s.last_seq FROM sensors s JOIN tenants t ON t.id = s.tenant_id WHERE s.sensor_type = 'camera' AND t.name LIKE 'Mara%%' ORDER BY s.ordinal LIMIT 1")
    before = one("SELECT count(*) AS n FROM observations WHERE sensor_id = %s", [cam["id"]])["n"]
    obs = [{"kind": "animal", "day": 31, "hour": 6.5, "label": "zebra", "confidence": 0.93, "count": 4}]
    ahead = client.post(f"/v1/sensors/{cam['id']}/sync", headers=RANGER, json={"seq": 33, "observations": obs}).json()
    assert ahead["status"] == "stored" and ahead["missing_batches"] == [30, 31, 32] and not ahead["arrived_out_of_order"]         # the gap is reported, not hidden
    late = client.post(f"/v1/sensors/{cam['id']}/sync", headers=RANGER, json={"seq": 31, "observations": obs}).json()
    assert late["status"] == "stored" and late["arrived_out_of_order"] and late["missing_batches"] == [30, 32]
    again = client.post(f"/v1/sensors/{cam['id']}/sync", headers=RANGER, json={"seq": 31, "observations": obs * 3}).json()
    assert again["status"] == "duplicate" and again["stored"] == 0
    assert one("SELECT count(*) AS n FROM observations WHERE sensor_id = %s", [cam["id"]])["n"] == before + 2
    wrong = client.post(f"/v1/sensors/{cam['id']}/sync", headers=RANGER, json={"seq": 40, "observations": [{"kind": "sound", "day": 31, "hour": 2, "features": [0.2, 0.01, 900, 3000, 80]}]})
    assert wrong.json()["code"] == "wrong_kind_for_sensor"


def test_new_evidence_changes_the_alerts(client, demo):
    cam = one("SELECT s.id FROM sensors s JOIN tenants t ON t.id = s.tenant_id WHERE s.name = 'CAM-30' AND t.name LIKE 'Mara%%'")["id"]
    for seq, day in ((41, 20), (42, 22), (43, 24), (44, 26)):
        client.post(f"/v1/sensors/{cam}/sync", headers=RANGER, json={"seq": seq, "observations": [{"kind": "vehicle", "day": day, "hour": 23.4, "label": "vehicle", "confidence": 0.95}]})
    names = [n for c in client.get("/v1/alerts/suspicious", headers=RANGER).json()["clusters"] for n in c["camera_names"]]
    assert "CAM-30" in names and "CAM-01" in names


def test_tracks_and_species_listing(client, demo):
    animals = client.get("/v1/animals", headers=RESEARCH).json()["items"]
    assert len(animals) == 8 and all(a["fixes"] == 120 for a in animals)
    tr = client.get(f"/v1/animals/{animals[0]['id']}/track", headers=RESEARCH, params={"from_day": 15, "to_day": 29}).json()
    assert tr["fixes"] == 60 and abs(sum(p["x"] for p in tr["points"]) / 60 - world.CORRIDOR[1]) < 1
    assert [s["name"] for s in demo["species"]["items"]] == sorted(world.SPECIES)


@pytest.mark.parametrize("headers,method,path,body,expected", [
    ({}, "GET", "/v1/species", None, 401),
    (RESEARCH, "GET", "/v1/species", None, 200),
    (RESEARCH, "POST", "/v1/patrols/optimize", {}, 403),
    (RESEARCH, "POST", f"/v1/sensors/{ZERO}/sync", {"seq": 1, "observations": []}, 403),
    (RESEARCH, "POST", "/v1/media/infer", {"kind": "acoustic", "features": [1, 1, 1, 1, 1]}, 403),
    (RANGER, "POST", "/v1/scenarios/month", None, 403),
    (RANGER, "GET", "/v1/simulation/truth", None, 403),
    (RANGER, "GET", "/v1/audit", None, 403),
    (RANGER, "POST", f"/v1/sensors/{ZERO}/sync", {"seq": 1, "observations": []}, 404),
    (RANGER, "POST", "/v1/media/infer", {"kind": "image", "features": [1, 1, 1, 1, 1]}, 422),
    (RANGER, "POST", "/v1/media/infer", {"kind": "acoustic", "features": [1, 1, 0, 1, 1]}, 422),
    (RANGER, "POST", "/v1/media/infer", {"kind": "acoustic", "features": [1, 1, 1]}, 422),
    (RANGER, "POST", "/v1/patrols/optimize", {"teams": 50}, 422),
    (RESEARCH, "GET", f"/v1/species/{ZERO}/distribution", None, 404),
    (RESEARCH, "GET", f"/v1/animals/{ZERO}/track", None, 404),
])
def test_authorization_and_validation(client, demo, headers, method, path, body, expected):
    assert client.request(method, path, headers=headers, json=body).status_code == expected


def test_one_reserve_cannot_see_or_feed_another(client, demo):
    mic = demo["sus"]["resend_sensor"]
    mine = one("SELECT count(*) AS n FROM observations o JOIN tenants t ON t.id = o.tenant_id WHERE t.name LIKE 'Mara%%'")["n"]
    assert client.get("/v1/alerts/suspicious", headers=OTHER).json()["code"] == "no_reserve"
    job = client.post("/v1/scenarios/month", headers=OTHER).json()              # their load deletes "everything": only their own rows
    jobs.drain()
    assert client.get(job["status_url"], headers=OTHER).json()["result"]["cameras"] == 60
    assert client.post(f"/v1/sensors/{mic}/sync", headers=OTHER, json={"seq": 99, "observations": []}).status_code == 404
    assert client.get(f"/v1/species/{demo['species']['items'][0]['id']}/distribution", headers=OTHER).status_code == 404
    theirs = {a["id"] for a in client.get("/v1/animals", headers=OTHER).json()["items"]}
    assert theirs and not theirs & {a["id"] for a in client.get("/v1/animals", headers=RESEARCH).json()["items"]}
    assert one("SELECT count(*) AS n FROM observations o JOIN tenants t ON t.id = o.tenant_id WHERE t.name LIKE 'Mara%%'")["n"] == mine
