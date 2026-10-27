"""Compact a SportVU 2015-16 game log into a small npz (N2 alignment + later Stage 3/4 truth).
  python src/stage1/sportvu_compact.py <MM.DD.YYYY.AWY.at.HOM> [out_dir]
Fetches the .7z from the public mirror (github.com/linouk23/NBA-Player-Movements) into a temp dir, dedupes moments
across events by timestamp and writes data_meta/sportvu/<game>.npz:
  t_ms (unix ms), period, game_clock (s), shot_clock (s, nan if none), xyz (N, 11, 3) ft (row 0 = ball, nan if absent),
  team (N, 11), pid (N, 11), plus roster json (playerid -> name, jersey, team)."""
import json, os, sys, tempfile, urllib.request
import numpy as np, py7zr

MIRROR = "https://github.com/linouk23/NBA-Player-Movements/raw/master/data/2016.NBA.Raw.SportVU.Game.Logs/{}.7z"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compact(game, out_dir):
    with tempfile.TemporaryDirectory() as td:
        arc = os.path.join(td, "g.7z")
        urllib.request.urlretrieve(MIRROR.format(game), arc)
        with py7zr.SevenZipFile(arc) as z: z.extractall(td)
        js = [f for f in os.listdir(td) if f.endswith(".json")][0]
        d = json.load(open(os.path.join(td, js)))
    roster, rows = {}, {}
    for ev in d["events"]:
        for side in ("home", "visitor"):
            for p in ev[side]["players"]:
                roster[p["playerid"]] = dict(name=f'{p["firstname"]} {p["lastname"]}', jersey=p["jersey"], team=ev[side]["abbreviation"])
        for m in ev["moments"]: rows.setdefault(m[1], m)
    ts = sorted(rows); n = len(ts)
    xyz = np.full((n, 11, 3), np.nan, np.float32); team = np.zeros((n, 11), np.int64); pid = np.zeros((n, 11), np.int64)
    per = np.zeros(n, np.int8); gc = np.zeros(n, np.float32); sc = np.full(n, np.nan, np.float32)
    for i, t in enumerate(ts):
        m = rows[t]; per[i], gc[i] = m[0], m[2]
        if m[3] is not None: sc[i] = m[3]
        ents = sorted(m[5], key=lambda r: r[0] != -1)[:11]   # ball first when present
        if not ents or ents[0][0] != -1: ents = [None] + ents[:10]
        for j, r in enumerate(ents):
            if r is None: continue
            team[i, j], pid[i, j] = r[0], r[1]; xyz[i, j] = r[2:5]
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{game}.npz"), t_ms=np.array(ts, np.int64), period=per, game_clock=gc, shot_clock=sc, xyz=xyz, team=team, pid=pid)
    json.dump(dict(gameid=d["gameid"], gamedate=d["gamedate"], roster={str(k): v for k, v in roster.items()}), open(os.path.join(out_dir, f"{game}.json"), "w"), indent=1)
    print(game, d["gameid"], n, "moments")


if __name__ == "__main__":
    compact(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "data", "sportvu"))
