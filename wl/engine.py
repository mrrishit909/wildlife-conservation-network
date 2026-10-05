"""Occupancy estimation, corridor shift, acoustic classification, night-vehicle anomaly detection, risk tiles and
patrol routing. Each takes plain lists so it can be scored offline against the simulator's truth."""
import math

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import world

VERSION = "wl-1.0"
MIN_CONFIDENCE = 0.8             # camera detections below this are stored but never raise an alert
SITE_CONFIDENCE = 0.9            # stricter still for "this species uses this site": one wrong label would count as presence
GUNSHOT_PREVALENCE = 0.001       # assumed share of all sounds that are gunshots (the training set has them at one in five)
FIELD_PRECISION = 0.25           # the alert threshold is the lowest at which a quarter of alerts would be real at that prevalence
COVER_KM = 1.5                   # a patrol covers what lies within this distance of its path
SPEED_KMH, DWELL_H = 12.0, 0.33


# ---------------------------------------------------------------- occupancy

def histories(detections, n_sites, days):
    """detections: [(site, day)] -> 0/1 matrix [site, day in `days`]"""
    y = np.zeros((n_sites, len(days)), dtype=int)
    col = {d: k for k, d in enumerate(days)}
    for s, d in detections:
        if d in col:
            y[s, col[d]] = 1
    return y


def _fit(d, k):
    def nll(t):
        psi, p = 1 / (1 + math.exp(-t[0])), 1 / (1 + math.exp(-t[1]))
        like = psi * p ** d * (1 - p) ** (k - d) + (1 - psi) * (d == 0)
        return -np.log(np.maximum(like, 1e-300)).sum()
    t = minimize(nll, [0.0, -1.0], method="Nelder-Mead").x
    return 1 / (1 + math.exp(-t[0])), 1 / (1 + math.exp(-t[1]))


def occupancy(y, boot=60, seed=0):
    """Single-season occupancy model: a site with no detections is either unused, or used and missed every day.
    Maximum likelihood for psi (share of sites used) and p (daily chance of detecting the species where it is).
    -> {naive, psi, p, low, high}: naive is the share of sites with any detection; low/high an 80% bootstrap range."""
    d, k = y.sum(axis=1), y.shape[1]
    if not d.sum():
        return {"sites": len(d), "sites_with_detections": 0, "naive": 0.0, "psi": 0.0, "p": None, "low": 0.0, "high": 0.0}
    psi, p = _fit(d, k)
    rng = np.random.default_rng(seed)
    bs = sorted(_fit(d[rng.integers(0, len(d), len(d))], k)[0] for _ in range(boot)) or [psi]
    return {"sites": len(d), "sites_with_detections": int((d > 0).sum()), "naive": round(float((d > 0).mean()), 3), "psi": round(psi, 3), "p": round(p, 3), "low": round(bs[int(0.1 * len(bs))], 3), "high": round(bs[int(0.9 * len(bs)) - 1], 3)}


# ---------------------------------------------------------------- corridor

def shift(before, after, perms=4000, seed=0):
    """Did the herd's east-west position change? before/after: lists of x (km east) of detections or fixes.
    -> shift in km and a permutation p-value (how often shuffling the period labels gives a shift this large)."""
    a, b = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if len(a) < 5 or len(b) < 5:
        return {"before_km_east": None, "after_km_east": None, "shift_km": None, "p_value": None, "n_before": len(a), "n_after": len(b)}
    obs = b.mean() - a.mean()
    rng, pool = np.random.default_rng(seed), np.concatenate([a, b])
    hits = sum(abs((q := rng.permutation(pool))[len(a):].mean() - q[:len(a)].mean()) >= abs(obs) for _ in range(perms))
    return {"before_km_east": round(float(a.mean()), 2), "after_km_east": round(float(b.mean()), 2), "shift_km": round(float(obs), 2), "p_value": round((hits + 1) / (perms + 1), 4), "n_before": len(a), "n_after": len(b)}


# ---------------------------------------------------------------- sounds

def train_sounds(pairs, holdout=None):
    """Multinomial logistic regression on log features. With a holdout, also picks the gunshot alert threshold.
    The training sounds are balanced across classes but in the field a gunshot is about one sound in a thousand, so
    precision is recomputed at that prevalence (Bayes' rule on the held-out hit and false-alarm rates) and the
    threshold is the lowest at which at least FIELD_PRECISION of alerts would be real. -> (model, threshold, metrics)"""
    model = make_pipeline(FunctionTransformer(np.log), StandardScaler(), LogisticRegression(max_iter=2000, C=5)).fit(np.array([x for x, _ in pairs]), [k for _, k in pairs])
    if not holdout:
        return model, 0.5, {}
    x, y = np.array([x for x, _ in holdout]), np.array([k for _, k in holdout])
    proba = model.predict_proba(x)
    g, real = proba[:, list(model.classes_).index("gunshot")], y == "gunshot"

    def at(thr):
        flag = g >= thr
        tpr, fpr = flag[real].mean(), flag[~real].mean()
        field = tpr * GUNSHOT_PREVALENCE / max(1e-12, tpr * GUNSHOT_PREVALENCE + fpr * (1 - GUNSHOT_PREVALENCE))
        return {"threshold": round(float(thr), 3), "recall": round(float(tpr), 4), "false_alarm_rate": round(float(fpr), 5), "precision_on_balanced_test": round(float(real[flag].mean()), 4) if flag.any() else None, "precision_expected_in_field": round(float(field), 4)}
    grid = [at(t) for t in np.arange(0.5, 0.9995, 0.005)]
    pick = next((r for r in grid if r["precision_expected_in_field"] >= FIELD_PRECISION and r["recall"] > 0), grid[-1])
    return model, pick["threshold"], {"accuracy": round(float((model.classes_[proba.argmax(axis=1)] == y).mean()), 4), "n_test": len(y), "assumed_gunshot_prevalence": GUNSHOT_PREVALENCE, "chosen": pick, "at_0.5": grid[0]}


def loud_and_short(features):
    """Baseline gunshot rule: short and loud."""
    return features[0] < 0.4 and features[4] > 72


# ---------------------------------------------------------------- night vehicles

def night_of(day, hour):
    """Detections after midnight belong to the night that started the evening before."""
    return day if hour >= 12 else day - 1


def is_night(hour):
    return hour >= 21 or hour < 5


def vehicle_clusters(dets, cams, sounds=(), n_nights=world.DAYS):
    """dets: [(camera index, day, hour, confidence)] vehicle detections; sounds: [(xy, night, label)] confident engine or
    gunshot sounds. A camera is flagged when confident night-time vehicles on at least two different nights are far
    more than chance (Poisson tail against the reserve-wide rate). Flagged cameras within 2.5 km form one cluster.
    -> (clusters, counts) """
    night = [(c, night_of(d, h), conf) for c, d, h, conf in dets if is_night(h)]
    sure = [(c, n) for c, n, conf in night if conf >= MIN_CONFIDENCE]
    by = {}
    for c, n in sure:
        by.setdefault(c, set()).add(n)
    counts = sorted(len(v) for v in by.values())
    rate = max(0.005, sum(counts[:max(1, int(0.9 * len(cams)))]) / (max(1, int(0.9 * len(cams))) * n_nights))          # background rate, ignoring the busiest tenth
    flagged = {c: (nights, float(poisson.sf(len(nights) - 1, rate * n_nights))) for c, nights in by.items() if len(nights) >= 2}
    flagged = {c: v for c, v in flagged.items() if v[1] < 0.001}
    clusters, left = [], set(flagged)
    while left:
        group, frontier = set(), [left.pop()]
        while frontier:
            c = frontier.pop()
            group.add(c)
            near = {o for o in left if math.dist(cams[c], cams[o]) <= 2.5}
            left -= near
            frontier += near
        group |= {c for c in by if c not in flagged and any(math.dist(cams[c], cams[g]) <= 2.5 for g in group)}        # neighbours that saw it too
        nights = sorted(set().union(*(by[c] for c in group)))
        centre = tuple(round(float(v), 2) for v in np.mean([cams[c] for c in group], axis=0))
        heard = sorted({(n, label) for xy, n, label in sounds if n in nights and math.dist(xy, centre) <= 4})
        clusters.append({"centre": centre, "cameras": sorted(group), "nights": nights, "detections": sum(c in group for c, _ in sure), "chance_p_value": min(flagged[c][1] for c in group if c in flagged),
                         "corroborating_sounds": [{"night": n, "sound": label} for n, label in heard], "confidence": "high" if len(group) >= 2 or heard else "medium"})
    return clusters, {"night_vehicle_detections": len(night), "confident": len(sure), "cameras_with_any": len({c for c, _, _ in night}), "cameras_flagged": len(flagged), "background_rate_per_camera_night": round(rate, 4)}


# ---------------------------------------------------------------- risk and patrols

def risk_tiles(clusters, gunshots, herd_xy, size=world.SIZE):
    """A size x size grid (1 km tiles) of relative risk: night-vehicle clusters, confident gunshots, and where the herd is
    close to the boundary. The weights are a stated judgement, not learnt."""
    gx, gy = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    bump = lambda pts, s: sum((np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * s * s)) for x, y in pts), np.zeros_like(gx))      # noqa: E731
    edge = np.exp(-np.minimum.reduce([gx, gy, size - gx, size - gy]) / 3)
    herd = bump(herd_xy, 2.0)
    r = 0.5 * bump([c["centre"] for c in clusters], 2.0) + 0.3 * bump(gunshots, 2.5) + 0.2 * (herd / herd.max() if herd.max() else herd) * edge
    return r / r.max() if r.max() else r


def covered(risk, paths):
    """Share of total risk lying within COVER_KM of any patrol path (paths are sampled every 0.5 km)."""
    size = risk.shape[0]
    gx, gy = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    seen = np.zeros_like(risk, dtype=bool)
    for path in paths:
        for a, b in zip(path, path[1:]):
            for t in np.linspace(0, 1, max(2, int(math.dist(a, b) / 0.5) + 1)):
                seen |= (gx - (a[0] + t * (b[0] - a[0]))) ** 2 + (gy - (a[1] + t * (b[1] - a[1]))) ** 2 <= COVER_KM ** 2
    return float(risk[seen].sum() / risk.sum()) if risk.sum() else 0.0


def path_hours(path, dwell=DWELL_H):
    return sum(math.dist(a, b) for a, b in zip(path, path[1:])) / SPEED_KMH + dwell * max(0, len(path) - 2)


def patrols(risk, hq=world.HQ, teams=3, hours=8.0, ms=1500):
    """Routes for `teams` ranger teams leaving and returning to headquarters within `hours`, visiting the waypoints
    that collect the most risk (prize-collecting vehicle routing). -> [[(x, y), ...]] one closed path per team"""
    size = risk.shape[0]
    order = np.dstack(np.unravel_index(np.argsort(-risk, axis=None), risk.shape))[0]
    pts = []
    for iy, ix in order:
        p = (ix + 0.5, iy + 0.5)
        if risk[iy, ix] > 0.02 and all(math.dist(p, q) >= 2.0 for q in pts):
            pts.append(p)
        if len(pts) == 36:
            break
    if not pts:
        return [[hq, hq] for _ in range(teams)]
    gx, gy = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    prize = [float(risk[(gx - x) ** 2 + (gy - y) ** 2 <= COVER_KM ** 2].sum()) for x, y in pts]
    nodes = [hq] + pts
    t = [[int(3600 * (math.dist(a, b) / SPEED_KMH + (DWELL_H if j else 0))) for j, b in enumerate(nodes)] for a in nodes]
    mgr = pywrapcp.RoutingIndexManager(len(nodes), teams, 0)
    model = pywrapcp.RoutingModel(mgr)
    cb = model.RegisterTransitCallback(lambda a, b: t[mgr.IndexToNode(a)][mgr.IndexToNode(b)])
    model.SetArcCostEvaluatorOfAllVehicles(cb)
    model.AddDimension(cb, 0, int(3600 * hours), True, "time")
    for k, pz in enumerate(prize):
        model.AddDisjunction([mgr.NodeToIndex(k + 1)], int(200000 * pz / max(prize)))          # skipping a waypoint costs its share of risk
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromMilliseconds(ms)
    sol = model.SolveWithParameters(p)
    out = []
    for v in range(teams):
        idx, path = model.Start(v), [hq]
        while sol and not model.IsEnd(idx):
            idx = sol.Value(model.NextVar(idx))
            path.append(nodes[mgr.IndexToNode(idx)])
        out.append(path if len(path) > 1 else [hq, hq])
    return out


def perimeter(teams=3, hours=8.0, hq=world.HQ, size=world.SIZE, inset=2.0):
    """Baseline: each team drives to its stretch of the boundary track, follows it as far as time allows, and returns."""
    corners = [(inset, inset), (size - inset, inset), (size - inset, size - inset), (inset, size - inset)]
    loop = [tuple(float(v) for v in np.array(a) + (np.array(b) - np.array(a)) * t) for a, b in zip(corners, corners[1:] + corners[:1]) for t in np.linspace(0, 1, 11)[:-1]]
    out = []
    for k in range(teams):
        start = k * len(loop) // teams
        path = [hq, loop[start], hq]
        for j in range(1, len(loop) // teams + 1):
            nxt = loop[(start + j) % len(loop)]
            trial = path[:-1] + [nxt, hq]
            if path_hours(trial, dwell=0) > hours:
                break
            path = trial
        out.append(path)
    return out
