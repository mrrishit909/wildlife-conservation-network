"""A deterministic synthetic reserve and one month of sensor data. Nothing here is real: there are no photographs and
no recordings. A camera "detection" is what an on-device classifier would output (a label and a confidence), and an
acoustic "event" is five summary features of a sound. Everything carries the truth it was generated from."""
import functools
import math
import random

SIZE, DAYS, LOSS_DAY = 24, 30, 15          # a 24 x 24 km reserve; habitat is lost in the west on day 15
HQ = (12.0, 12.0)
LOSS = (5, 10, 6, 18)                      # x0, x1, y0, y1 of the cleared block
CORRIDOR = (7.5, 14.0)                     # where the herd crosses, before and after the loss (km east)
HOTSPOT = (21.0, 3.0)                      # where the night vehicles appear
VEHICLE_NIGHTS = [17, 19, 21, 23, 25, 27]
GUNSHOT_NIGHTS = [21, 25]
STORM_DAYS = [6, 20, 26]
# species -> (occupancy intercept, slope on habitat quality, daily detection probability when present, nocturnal)
SPECIES = {"elephant": (0, 0, 0.45, False), "zebra": (-1.0, 3.0, 0.35, False), "antelope": (-0.2, 2.5, 0.5, False), "leopard": (-2.2, 2.0, 0.12, True), "warthog": (-0.8, 1.5, 0.3, False), "hyena": (-1.5, 1.2, 0.2, True)}
# sound class -> mean of (duration s, rise time s, peak Hz, bandwidth Hz, level dB)
SOUNDS = {"gunshot": (0.15, 0.005, 900, 3000, 85), "thunder": (2.5, 0.2, 120, 400, 80), "branch": (0.14, 0.007, 1500, 3600, 74), "engine": (8.0, 1.0, 150, 300, 65), "call": (1.2, 0.2, 1200, 800, 55)}
SOUND_FEATURES = ["duration_s", "rise_s", "peak_hz", "bandwidth_hz", "level_db"]


def quality(x, y, day=0):
    """Habitat quality 0..1: three good patches, minus the block cleared on LOSS_DAY."""
    q = 0.15 + sum(a * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / s) for a, cx, cy, s in ((0.8, 7, 12, 30), (0.7, 16, 16, 40), (0.6, 15, 6, 25)))
    if day >= LOSS_DAY and LOSS[0] <= x <= LOSS[1] and LOSS[2] <= y <= LOSS[3]:
        q *= 0.2
    return min(1.0, q)


def sound(r, kind):
    return [round(m * r.lognormvariate(0, 0.35 if i < 4 else 0.07), 4) for i, m in enumerate(SOUNDS[kind])]


def sounds(seed, n=3000):
    """Labelled sound features for fitting and scoring the acoustic classifier."""
    r = random.Random(seed)
    return [(sound(r, k), k) for k in (r.choice(list(SOUNDS)) for _ in range(n))]


@functools.lru_cache(maxsize=4)
def build(seed=13, n_cameras=60):
    r = random.Random(seed)
    cams = [(20.5, 2.5), (21.5, 3.5), (22.0, 2.0)]                # three cameras cover the south-east boundary track
    while len(cams) < n_cameras:
        p = (round(r.uniform(1, SIZE - 1), 1), round(r.uniform(1, SIZE - 1), 1))
        if all(math.dist(p, q) > 1.6 for q in cams):
            cams.append(p)
    mics = [(3.0 + 6 * i, 3.0 + 6 * j) for j in range(4) for i in range(4)]
    sensors = [{"type": "camera", "name": f"CAM-{i + 1:02d}", "xy": p} for i, p in enumerate(cams)] + [{"type": "acoustic", "name": f"MIC-{i + 1:02d}", "xy": p} for i, p in enumerate(mics)]
    names = list(SPECIES)
    obs, occupied = [], {}                                        # occupied[(species, period)] = set of camera indices truly used

    def detect(s, day, hour, label, conf, kind="animal", **truth):
        obs.append({"sensor": s, "day": day, "hour": round(hour, 2), "kind": kind, "label": label, "confidence": round(conf, 3), "count": truth.pop("count", 1), "truth": truth})

    for period, (d0, d1) in enumerate(((0, LOSS_DAY), (LOSS_DAY, DAYS))):
        for sp, (a, b, p, noct) in SPECIES.items():
            occ = set()
            for i, (x, y) in enumerate(cams):
                q = quality(x, y, d0)
                psi = (0.9 if abs(x - CORRIDOR[period]) < 3 else 0.04) if sp == "elephant" else 1 / (1 + math.exp(-(a + b * q)))
                if r.random() < psi:
                    occ.add(i)
                    for day in range(d0, d1):
                        if r.random() < p:
                            for _ in range(1 + (r.random() < 0.4)):
                                hour = r.uniform(19, 29) % 24 if noct else r.uniform(5.5, 18.5)
                                wrong = r.random() < 0.07
                                detect(i, day, hour, r.choice([n for n in names if n != sp]) if wrong else sp, r.betavariate(5, 3) if wrong else r.betavariate(9, 1.5), species=sp, count=r.randint(2, 9) if sp in ("elephant", "zebra", "antelope") else 1)
            occupied[sp, period] = occ
    for i, (x, y) in enumerate(cams):
        near_road = abs(y - HQ[1]) < 1.0
        for day in range(DAYS):
            for _ in range(sum(r.random() < 0.25 for _ in range(2))):                       # wind, grass: the classifier still names something, unsure
                detect(i, day, r.uniform(0, 24), r.choice(names), r.betavariate(2, 4), species=None)
            if near_road:
                for _ in range(sum(r.random() < 0.4 for _ in range(3))):
                    detect(i, day, r.uniform(7, 18), "vehicle", r.betavariate(9, 1.5), kind="vehicle", vehicle="authorised")
            if r.random() < 0.02:                                                             # an animal at night mistaken for a vehicle
                detect(i, day, r.uniform(21, 28) % 24, "vehicle", r.betavariate(3, 3), kind="vehicle", vehicle=None)
    for day in VEHICLE_NIGHTS:
        for i in range(3):
            if r.random() < 0.8:
                h = r.uniform(23, 27)                                  # late on that night: after midnight it is already the next calendar day
                detect(i, day + (h >= 24), h % 24, "vehicle", r.betavariate(9, 1.5), kind="vehicle", vehicle="intruder")
    corner = len(cams) + 3                                        # MIC-04 at (21, 3)
    for m in range(len(mics)):
        s = len(cams) + m
        for day in range(DAYS):
            for kind, k in (("call", r.randint(2, 6)), ("branch", r.randint(0, 2)), ("thunder", r.randint(4, 8) if day in STORM_DAYS else 0), ("engine", 1 if s == corner and day in VEHICLE_NIGHTS and r.random() < 0.75 else 0),
                            ("gunshot", (2 if day == GUNSHOT_NIGHTS[0] else 1) if s == corner and day in GUNSHOT_NIGHTS else 0)):
                for _ in range(k):
                    h = r.uniform(23, 27) if kind in ("engine", "gunshot") else r.uniform(0, 24)
                    obs.append({"sensor": s, "day": day + (h >= 24), "hour": round(h % 24, 2), "kind": "sound", "label": None, "confidence": None, "count": 1, "features": sound(r, kind), "truth": {"sound": kind}})
    # collared elephants: four fixes a day, drifting north and south along the corridor
    animals, tracks = [], []
    for k in range(8):
        animals.append({"tag": f"EL-{k + 1:03d}", "species": "elephant"})
        y, heading = r.uniform(3, 21), r.choice([-1, 1])
        for day in range(DAYS):
            for fix in range(4):
                y += heading * r.uniform(0.4, 1.4)
                if not 2 < y < SIZE - 2:
                    heading, y = -heading, min(SIZE - 2, max(2, y))
                tracks.append({"animal": k, "day": day, "hour": 6 * fix, "xy": (round(CORRIDOR[day >= LOSS_DAY] + r.gauss(0, 1.3), 2), round(y, 2))})
    # uploads: (arrival order, sensor, batch = day). Most upload daily; five save up a week; one is offline for five days; a few resend.
    weekly, offline = set(r.sample(range(3, n_cameras), 5)), r.randrange(3, n_cameras)
    uploads = []
    for s in range(len(sensors)):
        for day in range(DAYS):
            arrive = day + (6 - day % 7 if s in weekly else 0) + (20 - day if s == offline and 9 <= day <= 13 else 0) + r.random()
            uploads.append((arrive, s, day))
            if r.random() < 0.03:
                uploads.append((arrive + r.uniform(0.1, 2), s, day))
    uploads.sort()
    return {"seed": seed, "sensors": sensors, "n_cameras": n_cameras, "observations": obs, "occupied": occupied, "animals": animals, "tracks": tracks, "uploads": [(s, d) for _, s, d in uploads], "offline_sensor": offline}
