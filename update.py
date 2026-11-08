"""Market Pulse: holt Daten, berechnet Scores, schreibt data/latest.json + data/history.json."""
import io, json, os, sys, time, warnings
import numpy as np, pandas as pd, requests, yfinance as yf
warnings.filterwarnings("ignore")
CFG = json.load(open("config.json")); W = pd.Series(CFG["weights"]); AM = CFG["ampel"]
UA = {"User-Agent": "Mozilla/5.0"}
MK = {"SPX": ("S&P 500", "^GSPC"), "NDX": ("Nasdaq 100", "^NDX"),
      "DAX": ("DAX", "^GDAXI"), "MSCI": ("MSCI World (URTH)", "URTH")}
WIKI = {"SPX": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
        "NDX": ("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker"),
        "DAX": ("https://en.wikipedia.org/wiki/DAX", "Ticker")}
warn = []
os.makedirs("data", exist_ok=True)
def log(*a): print(*a, file=sys.stderr)

def dl(tickers, period):
    for _ in range(3):
        try:
            d = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, threads=True)["Close"]
            if isinstance(d, pd.Series): d = d.to_frame(list(tickers)[0])
            d = d.dropna(how="all")
            if len(d): d.index = pd.to_datetime(d.index).tz_localize(None); return d
        except Exception as e: log("dl error", e)
        time.sleep(5)
    return pd.DataFrame()

def members():
    cache = json.load(open("data/members.json")) if os.path.exists("data/members.json") else {}
    for k, (url, col) in WIKI.items():
        try:
            html = requests.get(url, headers=UA, timeout=30).text
            for t in pd.read_html(io.StringIO(html)):
                cs = [x for x in t.columns if str(x).lower() in ("ticker", "symbol", "ticker symbol")]
                if cs and len(t) > 20:
                    s = [str(x).strip() for x in t[cs[0]]]
                    s = [x if k == "DAX" else x.replace(".", "-") for x in s]
                    if k == "DAX": s = [x if "." in x else x + ".DE" for x in s]
                    cache[k] = s; break
            else: warn.append(f"Mitglieder {k}: keine Tabelle gefunden")
        except Exception as e: warn.append(f"Mitglieder {k}: {e}")
    json.dump(cache, open("data/members.json", "w"))
    return cache

def fred(i):
    for n in range(3):
        try:
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={i}", headers=UA, timeout=60)
            s = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True).iloc[:, 0]
            return pd.to_numeric(s, errors="coerce").dropna()
        except Exception:
            if n == 2: raise
            time.sleep(10)

def prank(s, win): return s.rolling(win, min_periods=win // 3).apply(lambda a: (a[:-1] < a[-1]).mean() * 100, raw=True)
def above(r, n):
    m = r.rolling(n).mean(); return ((r > m) * 100.0).where(m.notna())
def reidx(s, idx): return s.reindex(s.index.union(idx)).ffill().reindex(idx)

def trend(c):
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    return (40 * (c > s200) + 25 * (c > s50) + 25 * (s50 > s200) + 10 * (s200 > s200.shift(20))).where(s200.notna())
def mom(c): return pd.concat([prank(c.pct_change(h), 2520) for h in (21, 63, 126, 252)], axis=1).mean(axis=1)
def breadth(C):
    out = []
    for n in (200, 50):
        m = C.rolling(n).mean(); v = m.notna() & C.notna()
        out.append(((C > m) & v).sum(axis=1) / v.sum(axis=1).replace(0, np.nan) * 100)
    return (out[0] + out[1]) / 2
def label(s): return "sehr pessimistisch" if s < 20 else "pessimistisch" if s < 40 else "neutral" if s < 60 else "optimistisch" if s < 80 else "sehr optimistisch"

# ---- Daten laden ----
base = ["^GSPC", "^NDX", "^GDAXI", "URTH", "^VIX", "^VIX3M", "HYG", "LQD", "SPY", "TLT"]
D = dl(base, "12y")
if D.empty: sys.exit("Keine Kursdaten erhalten")
mem = members(); C = {}
for k, syms in mem.items():
    parts = []
    for i in range(0, len(syms), 100):
        parts.append(dl(syms[i:i + 100], "2y"))
    if parts:
        df = pd.concat([p for p in parts if len(p)], axis=1)
        C[k] = df.loc[:, df.notna().sum() > 250]; log(k, C[k].shape[1], "Werte")

# ---- Globale Reihen: Volatilität, Risikoappetit ----
vp = []
if "^VIX" in D: vp.append(100 - prank(D["^VIX"].dropna(), 1260))
if "^VIX" in D and "^VIX3M" in D: vp.append(((1.1 - D["^VIX"] / D["^VIX3M"]) / 0.2 * 100).clip(0, 100))
vol = pd.concat(vp, axis=1).ffill().mean(axis=1) if vp else None
rp = []
try:
    hy, d10, d3 = fred("BAMLH0A0HYM2"), fred("DGS10"), fred("DGS3MO")
    rp += [100 - prank(hy, 1260), ((d10 - d3 + 1) / 3 * 100).clip(0, 100)]
except Exception as e: warn.append(f"FRED: {e}")
if {"HYG", "LQD"} <= set(D): rp.append(above(D["HYG"] / D["LQD"], 50))
if {"SPY", "TLT"} <= set(D): rp.append(above(D["SPY"] / D["TLT"], 50))
risk = pd.concat(rp, axis=1).ffill().mean(axis=1) if rp else None

# ---- Scores je Markt ----
B = {k: breadth(v) for k, v in C.items()}
AR, SC = {}, {}
for k, (name, t) in MK.items():
    if t not in D: warn.append(f"{name}: keine Daten"); continue
    c = D[t].dropna(); ix = c.index
    if k == "MSCI" and B: b = pd.concat([reidx(x, ix) for x in B.values()], axis=1).mean(axis=1)
    else: b = reidx(B[k], ix) if k in B else None
    f = pd.DataFrame({"trend": trend(c), "momentum": mom(c),
                      "vol": reidx(vol, ix) if vol is not None else np.nan,
                      "breadth": b if b is not None else np.nan,
                      "risk": reidx(risk, ix) if risk is not None else np.nan})
    SC[k] = (f * W).sum(axis=1, min_count=1) / (f.notna() * W).sum(axis=1).replace(0, np.nan)
    AR[k] = f
allsc = pd.concat(SC, axis=1).ffill().mean(axis=1).dropna()

# ---- Ausgabe ----
last = lambda s: round(float(s.dropna().iloc[-1]), 1)
markets = {k: {"name": MK[k][0], "score": last(SC[k]), "label": label(last(SC[k])),
               "areas": {a: round(float(AR[k][a].dropna().iloc[-1]), 1) for a in W.index if AR[k][a].notna().any()}} for k in SC}
areas_all = {a: round(float(np.mean([m["areas"][a] for m in markets.values() if a in m["areas"]])), 1)
             for a in W.index if any(a in m["areas"] for m in markets.values())}
markets = {"ALL": {"name": "Gesamt", "score": last(allsc), "label": label(last(allsc)), "areas": areas_all}, **markets}

crosses = {}
for k, (name, t) in MK.items():
    if t in D:
        c = D[t].dropna(); d = (c.rolling(50).mean() > c.rolling(200).mean())[c.rolling(200).mean().notna()]
        ch = d[d != d.shift()].iloc[1:]
        if len(ch): crosses[k] = {"type": "Golden Cross" if ch.iloc[-1] else "Death Cross", "date": ch.index[-1].strftime("%Y-%m-%d")}

# Ampel (angelehnt an LetMoneyWork, Schwellen in config.json)
ampel = {"status": "unbekannt", "items": {}, "reasons": []}
try:
    c = D["^GSPC"].dropna(); v = D["^VIX"].dropna()
    ema21, s50, s200 = c.ewm(span=21).mean().iloc[-1], c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
    px = c.iloc[-1]; vx = v.iloc[-1]; vch = (vx / v.iloc[-2] - 1) * 100
    fg = None
    try:
        hd = {**UA, "Referer": "https://edition.cnn.com/", "Origin": "https://edition.cnn.com"}
        fg = float(requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata/" + pd.Timestamp.utcnow().strftime("%Y-%m-%d"), headers=hd, timeout=20).json()["fear_and_greed"]["score"])
    except Exception as e: warn.append(f"Fear&Greed nicht abrufbar: {e}")
    t_ok = bool(px > ema21 > s50 > s200); v_ok = bool(vx < AM["vix_max"] and vch < AM["vix_rise_max_pct"])
    f_ok = None if fg is None else bool(fg > AM["fear_greed_min"])
    ampel = {"status": "green" if t_ok and v_ok and f_ok is not False else "red",
             "checks": {"trend": t_ok, "vix": v_ok, "fear_greed": f_ok},
             "items": {"S&P 500": round(px, 1), "EMA 21": round(ema21, 1), "SMA 50": round(s50, 1), "SMA 200": round(s200, 1),
                       "VIX": round(vx, 1), "VIX Änderung %": round(vch, 1), "Fear & Greed": None if fg is None else round(fg)}}
except Exception as e: warn.append(f"Ampel: {e}")

date = allsc.index[-1].strftime("%Y-%m-%d")
json.dump({"date": date, "updated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "markets": markets,
           "ampel": ampel, "crosses": crosses, "warnings": warn}, open("data/latest.json", "w"), ensure_ascii=False, indent=1)

hp = "data/history.json"
H = {r["d"]: r for r in json.load(open(hp))} if os.path.exists(hp) else {}
al = pd.concat({"ALL": allsc, **SC}, axis=1).ffill()
for d, row in al.iloc[-400:].iterrows():
    H[d.strftime("%Y-%m-%d")] = {"d": d.strftime("%Y-%m-%d"), **{k: round(float(x), 1) for k, x in row.items() if pd.notna(x)}}
json.dump([H[k] for k in sorted(H)], open(hp, "w"))
log("Fertig", date, markets["ALL"]["score"], warn)
