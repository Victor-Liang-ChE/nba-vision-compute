"""N2: align a broadcast clip to SportVU by the scorebug game clock.
  python src/stage1/align_sportvu.py <video> <sportvu game> [--t0 SECONDS_INTO_FULL_VIDEO] [--fps 10]
1. Locate the scorebug clock: RapidOCR on 1 frame / 2 s, keep boxes reading M:SS or SS.s with a period word (1st..4th, OT)
   next to them; the clock box = median of those boxes (no per-broadcast config).
2. Read the clock crop at --fps; clock ticks = frames where the reading drops by exactly 1 s.
3. Each tick maps to the SportVU moment where the displayed clock (ceil of game_clock) makes the same drop.
   offset = SportVU unix ms - video ms; live stretches share one offset (breaks / edits change it).
4. Report: ticks, offset segments, residual |tick - segment median| (pass rule: median < 500 ms) -> notes/align/<clip>.json."""
import argparse, json, os, re
import cv2, numpy as np
from rapidocr import RapidOCR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLOCK = re.compile(r"^(\d{1,2}):(\d{2})$|^(\d{1,2})\.(\d)$")
PERIOD = re.compile(r"^(1st|2nd|3rd|4th|OT\d?)$", re.I)


def parse_clock(s):
    m = CLOCK.match(s.replace(" ", ""))
    if not m: return None
    return int(m[1]) * 60 + int(m[2]) if m[1] else int(m[3]) + int(m[4]) / 10


def frames(cap, step):
    i = 0
    while True:
        ok, f = cap.read()
        if not ok: return
        if i % step == 0: yield i, f
        i += 1


def locate(video, eng):
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS); hits, per, pboxes = [], [], []
    for i, f in frames(cap, int(round(fps * 2))):
        r = eng(f)
        if r.txts is None: continue
        P = [(t.strip(), b) for t, b in zip(r.txts, r.boxes) if PERIOD.match(t.strip())]
        for t, b in zip(r.txts, r.boxes):
            if parse_clock(t) is None or not P: continue
            c = np.array(b).mean(0); pc = np.array(P[0][1]).mean(0)
            if abs(c[1] - pc[1]) < 40 and abs(c[0] - pc[0]) < 300:   # same scorebug row as the period word
                hits.append(np.array(b)); per.append(PERIOD.match(P[0][0])[1].lower()); pboxes.append(np.array(P[0][1]))
        if len(hits) >= 25: break
    cap.release()
    if not hits: raise SystemExit("no scorebug clock found")
    b = np.median(np.stack(hits), 0); x0, y0 = b.min(0); x1, y1 = b.max(0)
    pad = 0.35 * (y1 - y0)
    period = max(set(per), key=per.count)
    pb = np.median(np.stack(pboxes), 0); px0, py0 = pb.min(0); px1, py1 = pb.max(0)
    locate.period_box = (int(px0 - pad), int(py0 - pad), int(px1 + pad), int(py1 + pad))
    return (int(x0 - 0.5 * pad), int(y0 - pad), int(x1 + 0.5 * pad), int(y1 + pad)), period, len(hits)


def read_clock(video, box, eng, fps_read):
    """Every native frame is checked, but OCR (recognition only, no detection) runs only when the clock crop changes: one read per
    displayed value instead of per frame, and the change frame gives the tick to the frame (33 ms at 30 fps). fps_read is unused."""
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS); out = []; x0, y0, x1, y1 = box; last = None; v = None; n_ocr = 0
    pbx = getattr(locate, "period_box", None); per_now = None; per_log = []
    for i, f in frames(cap, 1):
        if pbx and i % int(round(fps * 5)) == 0:   # period word every 5 s (full games cross periods)
            a0, b0, a1, b1 = pbx; rr = eng(cv2.resize(f[max(0, b0):b1, max(0, a0):a1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), use_det=False, use_cls=False)
            mm = re.search(r"(1ST|2ND|3RD|4TH|OT\d?)", "".join(rr.txts or []).upper().replace(" ", ""))
            if mm: per_now = mm[1].lower(); per_log.append((round(i / fps, 1), per_now))
        g = cv2.cvtColor(f[max(0, y0):y1, max(0, x0):x1], cv2.COLOR_BGR2GRAY)
        if last is None or np.abs(g.astype(np.int16) - last).mean() > 6:
            r = eng(cv2.resize(f[max(0, y0):y1, max(0, x0):x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), use_det=False, use_cls=False); n_ocr += 1
            t = "".join(r.txts) if r.txts is not None else ""
            m = re.search(r"(\d{1,2}):(\d{2})|(?<![\d:])(\d{1,2})\.(\d)(?!\d)", t.replace(" ", ""))   # the crop may also hold the period / shot clock
            v = (int(m[1]) * 60 + int(m[2]) if m[1] else int(m[3]) + int(m[4]) / 10) if m else None; last = g.astype(np.int16)
        out.append((i / fps, v, per_now))
        if i % 1800 == 0: print("frame", i, "ocr calls", n_ocr, "clock in", sum(x is not None for _, x, _p in out), flush=True)
    cap.release(); read_clock.per_log = per_log; return out


def sportvu_ticks(game, period):
    d = np.load(f"{ROOT}/data/sportvu/{game}.npz"); p = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}.get(period, 5)
    sel = d["period"] == p; gc, t = d["game_clock"][sel], d["t_ms"][sel]
    disp = np.ceil(gc - 1e-6).astype(int); ticks = {}
    for k in np.nonzero(np.diff(disp) == -1)[0]:   # displayed value drops from disp[k] to disp[k+1] at moment k+1
        ticks.setdefault(int(disp[k + 1]), t[k + 1])   # first crossing (clock stopped / restarted keeps the first)
    return ticks


def main():
    a = argparse.ArgumentParser(); a.add_argument("video"); a.add_argument("game"); a.add_argument("--t0", type=float, default=0); a.add_argument("--fps", type=float, default=10)
    o = a.parse_args(); eng = RapidOCR()
    box, period, nhit = locate(o.video, eng); print("scorebug clock at", box, "period", period, "hits", nhit, flush=True)
    reads = read_clock(o.video, box, eng, o.fps)
    PN = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
    vt = [(t, v, pp) for t, v, pp in reads if v is not None and v >= 60]   # whole-second display only (tenths under 1:00 skipped for now)
    ticks_v = [(t1, int(v1), p1 or period) for (t0, v0, _), (t1, v1, p1) in zip(vt, vt[1:]) if v0 - v1 == 1 and t1 - t0 < 0.1]   # consecutive native frames
    # each tick matches SportVU in the period read off the scorebug (every 5 s; the located period until the first read)
    SV = {p: sportvu_ticks(o.game, {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(p, "OT")) for p in range(1, 6)}
    off = [(tv, val, SV[PN.get(pp, 5)][val] - (o.t0 + tv) * 1000) for tv, val, pp in ticks_v if val in SV[PN.get(pp, 5)]]
    ticks_v = [(tv, val) for tv, val, pp in ticks_v]
    # segments: consecutive ticks whose offsets agree within 300 ms
    segs, cur = [], []
    for x in off:
        if cur and abs(x[2] - np.median([c[2] for c in cur])) > 300: segs.append(cur); cur = []
        cur.append(x)
    if cur: segs.append(cur)
    res = [abs(x[2] - np.median([c[2] for c in s])) for s in segs if len(s) >= 5 for x in s]
    rep = dict(video=os.path.basename(o.video), game=o.game, t0=o.t0, period=period, clock_box=box, locate_hits=nhit,
               frames_read=len(reads), frames_with_clock=sum(r[1] is not None for r in reads), video_ticks=len(ticks_v), matched_ticks=len(off),
               segments=[dict(start_s=s[0][0], end_s=s[-1][0], n=len(s), offset_ms=float(np.median([c[2] for c in s]))) for s in segs],
               residual_ms_median=float(np.median(res)) if res else None, residual_ms_p90=float(np.percentile(res, 90)) if res else None,
               ticks_in_segments_ge5=len(res), ticks=[[round(t, 3), v] for t, v in ticks_v], period_log=getattr(read_clock, "per_log", []))
    rep["pass"] = rep["residual_ms_median"] is not None and rep["residual_ms_median"] < 500
    os.makedirs(f"{ROOT}/notes/align", exist_ok=True)
    json.dump(rep, open(f"{ROOT}/notes/align/{os.path.splitext(rep['video'])[0]}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "segments"}, indent=1)); print("segments:", [(round(s["start_s"]), round(s["end_s"]), s["n"]) for s in rep["segments"]])


if __name__ == "__main__":
    main()
