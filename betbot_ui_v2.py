# betbot_ui_v2.py — UI for train_nfl_models.py bundle with pick logging + feedback calibration.
# Includes realism guardrails (blend/clamp/shrink/deviation cap/underdog floor),
# favorite/dog balancing, and "Last 5 matchups" table (score, site, who covered).
# Adds a ProbEnsemble compatibility shim so bundles unpickle cleanly.

import os, math, pickle, requests, pandas as pd, numpy as np
from datetime import datetime, timedelta, date, time
from collections import defaultdict

# --- robust TZ ---
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        ET = ZoneInfo("America/New_York")
    except ZoneInfoNotFoundError:
        import tzdata  # noqa
        ET = ZoneInfo("America/New_York")
except Exception:
    from dateutil.tz import gettz
    ET = gettz("America/New_York")

import streamlit as st

# --- compatibility shim for pickled ensemble from trainer ---
class ProbEnsemble:
    # during unpickle, `self.models` is populated automatically
    def __init__(self, models=None):
        self.models = models or []
    def predict_proba(self, X):
        import numpy as np
        if hasattr(self.models, "predict_proba"):  # single model case
            return self.models.predict_proba(X)
        ps = [m.predict_proba(X) for m in self.models]
        return np.mean(ps, axis=0)

SPORT_KEY = "americanfootball_nfl"
BOOK_ALLOW = {"DraftKings","FanDuel"}
DEFAULT_ODDS_API_KEY = "32036e70a91adcd434de3515fe5f161d"
PICKS_LOG = "picks_log.csv"
FEEDBACK_PATH = "model_feedback.pkl"

TEAM_ABBR = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
    "Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS"
}
DIVISION = {
    "BUF":"AFCE","MIA":"AFCE","NE":"AFCE","NYJ":"AFCE","BAL":"AFCN","CIN":"AFCN","CLE":"AFCN","PIT":"AFCN",
    "HOU":"AFCS","IND":"AFCS","JAX":"AFCS","TEN":"AFCS","DEN":"AFCW","KC":"AFCW","LV":"AFCW","LAC":"AFCW",
    "DAL":"NFCE","NYG":"NFCE","PHI":"NFCE","WAS":"NFCE","CHI":"NFCN","DET":"NFCN","GB":"NFCN","MIN":"NFCN",
    "ATL":"NFCS","CAR":"NFCS","NO":"NFCS","TB":"NFCS","ARI":"NFCW","LAR":"NFCW","SEA":"NFCW","SF":"NFCW",
}

# ---------- odds + math helpers ----------
def american_to_decimal(a:int)->float: return 1 + (a/100.0 if a>0 else 100.0/abs(a))
def pretty_line(x): return f"{x:+.1f}"
def to_et(iso):
    try: return datetime.fromisoformat(iso.replace("Z","+00:00")).astimezone(ET)
    except: return None

def logit(p):
    p=max(1e-6,min(1-1e-6,p))
    return math.log(p/(1-p))
def sigmoid(z): 
    return 1/(1+math.exp(-z))

def blend_prob_clamped(p_mkt, p_model, w=0.15, clamp_pp=0.12):
    """Blend in log-odds then clamp to stay within market ± clamp_pp (probability points)."""
    if p_mkt is None: 
        return p_model
    z = w*logit(p_mkt) + (1-w)*logit(p_model)
    p = sigmoid(z)
    return min(max(p, p_mkt - clamp_pp), p_mkt + clamp_pp)

def shrink_toward_half(p, shrink):
    """shrink in [0,1]: 1 = raw, 0 = 0.5"""
    return 0.5 + shrink*(p - 0.5)

def fetch_odds(api_key, markets="spreads"):
    url=f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
    r=requests.get(url, params={"apiKey":api_key,"regions":"us","markets":markets,"oddsFormat":"american","dateFormat":"iso"}, timeout=25)
    r.raise_for_status(); return r.json()

def offers_by_point(sp_ev, home, away):
    by_point=defaultdict(list)
    for b in sp_ev.get("bookmakers",[]) or []:
        title=b.get("title",b.get("key","?"))
        if title not in BOOK_ALLOW: continue
        for m in b.get("markets",[]):
            if m.get("key")!="spreads": continue
            hp=ap=None; pt=None
            for o in m.get("outcomes",[]):
                nm,pr,po=o.get("name"),o.get("price"),o.get("point")
                if isinstance(pr,int):
                    if nm==home: hp=pr; pt=po
                    elif nm==away: ap=pr; pt=po if pt is None else pt
            if hp is not None and ap is not None and pt is not None:
                by_point[float(pt)].append({"book":title,"home_price":hp,"away_price":ap})
    return by_point

def best_price(offers_at_point, side):
    key="home_price" if side=="home" else "away_price"
    best=None
    for o in offers_at_point:
        if (best is None) or (o[key]>best["price"]): best={"price":o[key], "book":o["book"]}
    return best

def consensus_prob(offers_at_point):
    vals=[]
    for off in offers_at_point:
        da, db = american_to_decimal(off["home_price"]), american_to_decimal(off["away_price"])
        s = 1/da + 1/db
        ph = (1/da)/s
        vals.append(ph)
    return float(np.mean(vals)) if vals else None

# ---------- schedule + features (live, mirrors trainer) ----------
def tz_aware_kick(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    out["game_date"]=pd.to_datetime(out.get("game_date", out.get("gameday")))
    hhmm = out.get("game_time_eastern", out.get("gametime", pd.Series(["13:00"]*len(out)))).fillna("13:00")
    def hh(x):
        try: return int(str(x).split(":")[0])
        except: return 13
    naive=pd.to_datetime(out["game_date"].dt.date.astype(str)+" "+hhmm.apply(hh).astype(str)+":00", errors="coerce")
    out["kickoff_et"]=naive.dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
    return out

def derive_home_line(row):
    for cand in ("spread_line","home_spread"):
        if cand in row and pd.notna(row[cand]):
            try: return float(row[cand])
            except: pass
    if "spread_favorite" in row and "spread_close" in row and pd.notna(row["spread_close"]):
        try:
            ln=float(row["spread_close"]); fav=row["spread_favorite"]
            return -abs(ln) if str(fav)==str(row["home_team"]) else abs(ln)
        except: return None
    return None

def build_history_state(up_to_dt: datetime, h2h_years:int):
    try:
        import nfl_data_py as nfl
    except Exception:
        return None
    df = nfl.import_schedules(list(range(2010, up_to_dt.year+1)))
    df = df[df.get("game_type","REG").fillna("REG")=="REG"].copy()
    df = tz_aware_kick(df)
    df = df[df["kickoff_et"] < up_to_dt].copy().sort_values("kickoff_et")

    ratings=defaultdict(lambda:1500.0)
    last_played={}
    form_pd=defaultdict(list)
    ats_all=defaultdict(list); ats_home=defaultdict(list); ats_away=defaultdict(list)
    mvs_all=defaultdict(list)
    h2h_any=defaultdict(list); h2h_host=defaultdict(list)

    for _,r in df.iterrows():
        ht,at,ko = r["home_team"], r["away_team"], r["kickoff_et"]
        H=TEAM_ABBR.get(ht,ht); A=TEAM_ABBR.get(at,at)

        hs=pd.to_numeric(r.get("home_score")); as_=pd.to_numeric(r.get("away_score"))
        line=derive_home_line(r)
        if pd.notna(hs) and pd.notna(as_) and line is not None:
            elo_diff=(ratings[ht]+55.0)-ratings[at]
            home_win=int(hs>as_)
            pdiff=int(hs)-int(as_)
            exp = 1/(1+10**(-elo_diff/400))
            mov = math.log(max(abs(pdiff),1)+1)*(2.2/((abs((ratings[ht]-ratings[at])))*0.001+2.2))
            delta=20.0*mov*(home_win-exp)
            ratings[ht]+=delta; ratings[at]-=delta
            last_played[ht]=ko; last_played[at]=ko

            cover_home = 1 if ((hs-as_)+line)>0 else (0 if ((hs-as_)+line)<0 else None)
            if cover_home is not None:
                form_pd[ht].append(pdiff); form_pd[at].append(-pdiff)
                ats_all[ht].append(cover_home); ats_all[at].append(1-cover_home)
                ats_home[ht].append(cover_home); ats_away[at].append(1-cover_home)
                mvs_all[ht].append((hs-as_)+line); mvs_all[at].append(-((hs-as_)+line))
                ts = ko
                h2h_any[(H,A)].append((cover_home, ts))
                h2h_any[(A,H)].append((1-cover_home, ts))
                h2h_host[(H,A)].append((cover_home, ts))

    return {"ratings":ratings,"last_played":last_played,"form_pd":form_pd,
            "ats_all":ats_all,"ats_home":ats_home,"ats_away":ats_away,"mvs_all":mvs_all,
            "h2h_any":h2h_any,"h2h_host":h2h_host,"h2h_years":h2h_years}

def build_spread_feats(order, home, away, line_home, when_dt, state):
    ht=TEAM_ABBR.get(home,home); at=TEAM_ABBR.get(away,away)
    ratings=state["ratings"]; HFA=55.0
    lp=state["last_played"]; form=state["form_pd"]; ats_all=state["ats_all"]; ats_home=state["ats_home"]; ats_away=state["ats_away"]; mvs=state["mvs_all"]
    h2h_any=state["h2h_any"]; h2h_host=state["h2h_host"]; yrs=state["h2h_years"]

    rest_h = (when_dt - lp.get(home, when_dt - timedelta(days=7))).days if isinstance(lp.get(home), datetime) else 7
    rest_a = (when_dt - lp.get(away, when_dt - timedelta(days=7))).days if isinstance(lp.get(away), datetime) else 7

    def avg_tail(lst,n): return float(np.mean(lst[-n:])) if lst else 0.0
    def rate_tail(lst,n): return float(np.mean(lst[-n:])) if lst else 0.0

    form3_h=avg_tail(form[home],3); form3_a=avg_tail(form[away],3)
    form5_h=avg_tail(form[home],5); form5_a=avg_tail(form[away],5)
    ats10_h=rate_tail(ats_all[home],10); ats10_a=rate_tail(ats_all[away],10)
    ats10_home_h=rate_tail(ats_home[home],10); ats10_away_a=rate_tail(ats_away[away],10)
    mvs5_h=avg_tail(mvs[home],5); mvs5_a=avg_tail(mvs[away],5)

    cutoff = when_dt - timedelta(days=365*yrs)
    def list_rate(vlist):
        v=[v for v,t in vlist if t>=cutoff]
        return (float(np.mean(v)), len(v)) if v else (0.5,0)
    h2h_any_rate, h2h_any_n = list_rate(h2h_any[(ht,at)])
    h2h_host_rate, h2h_host_n = list_rate(h2h_host[(ht,at)])

    elo_diff=(ratings[home]+HFA)-ratings[away]
    elo_spread=elo_diff/28.0
    value_vs_elo=line_home-elo_spread
    abs_line=abs(line_home)
    is_half=1.0 if abs(line_home-round(line_home))>1e-6 else 0.0
    def keynum(k): 
        return 1.0 if abs(round(line_home))==k and abs(line_home-round(line_home))<1e-9 else 0.0

    feats = {
        "line_home": float(line_home), "abs_line": abs_line, "is_half": is_half, "home_fav": 1.0 if line_home<0 else 0.0,
        "key3": keynum(3),"key6": keynum(6),"key7": keynum(7),"key10": keynum(10),"key14": keynum(14),
        "divisional_game": 1.0 if DIVISION.get(ht)==DIVISION.get(at) else 0.0,
        "rest_days_diff": float(rest_h-rest_a),
        "short_week_home": 1.0 if rest_h<=5 else 0.0, "short_week_away": 1.0 if rest_a<=5 else 0.0,
        "post_bye_home": 1.0 if rest_h>=13 else 0.0, "post_bye_away": 1.0 if rest_a>=13 else 0.0,
        "home_last3_pd": form3_h, "away_last3_pd": form3_a, "home_last5_pd": form5_h, "away_last5_pd": form5_a,
        "team_h_ats10": ats10_h, "team_a_ats10": ats10_a, "team_h_ats10_home": ats10_home_h, "team_a_ats10_away": ats10_away_a,
        "team_h_mvs5": mvs5_h, "team_a_mvs5": mvs5_a,
        "h2h_any_rate": h2h_any_rate, "h2h_any_n": float(h2h_any_n),
        "h2h_host_rate": h2h_host_rate, "h2h_host_n": float(h2h_host_n),
        "elo_diff_home_minus_away": float(elo_diff), "elo_spread": float(elo_spread), "value_vs_elo": float(value_vs_elo),
    }
    vec=np.array([feats.get(f,0.0) for f in order], dtype=float)
    return vec, feats

def build_ml_feats(order, home, away, when_dt, state):
    ratings=state["ratings"]; HFA=55.0
    lastp=state["last_played"]; form=state["form_pd"]
    rest_h=(when_dt - lastp.get(home, when_dt - timedelta(days=7))).days if isinstance(lastp.get(home), datetime) else 7
    rest_a=(when_dt - lastp.get(away, when_dt - timedelta(days=7))).days if isinstance(lastp.get(away), datetime) else 7
    def avg_tail(lst,n): return float(np.mean(lst[-n:])) if lst else 0.0
    form3_h=avg_tail(form[home],3); form3_a=avg_tail(form[away],3)
    elo_diff=(ratings[home]+HFA)-ratings[away]
    feats={"elo_diff_home_minus_away":float(elo_diff),
           "rest_days_diff":float(rest_h-rest_a),
           "divisional_game":1.0 if DIVISION.get(TEAM_ABBR.get(home,home))==DIVISION.get(TEAM_ABBR.get(away,away)) else 0.0,
           "home_last3_pd":float(form3_h),"away_last3_pd":float(form3_a)}
    vec=np.array([feats.get(f,0.0) for f in order], dtype=float)
    return vec, feats

# trainer-compatible bucket cal
def bucket_edges_from_bundle(bundle):
    return (bundle.get("spread",{}).get("pack",{}) or {}).get("bucket_edges") or [2.5,3.5,6.5,9.5,13.5]
def apply_bucket_cal_single(p, abs_line, fav_flag, edges, cals):
    b=0
    for j,e in enumerate(edges):
        if abs_line>e: b=j+1
        else: break
    iso=(cals or {}).get((b,int(fav_flag)))
    return float(iso.predict([float(p)])[0]) if iso is not None else float(p)

def load_feedback():
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH,"rb") as f: return pickle.load(f)
        except Exception: return {}
    return {}

def apply_feedback_cal(p, market, fav_flag, abs_line, fb):
    try:
        if market=="Moneyline":
            key=("moneyline", 1 if fav_flag else 0)
            iso=(fb.get(key) or {}).get("iso")
            return float(iso.predict([float(p)])[0]) if iso else float(p)
        else:
            edges=[2.5,3.5,6.5,9.5,13.5]
            b=0
            for j,e in enumerate(edges):
                if abs_line>e: b=j+1
                else: break
            key=("spread", b, 1 if fav_flag else 0)
            iso=(fb.get(key) or {}).get("iso")
            return float(iso.predict([float(p)])[0]) if iso else float(p)
    except Exception:
        return float(p)

# ----- Last 5 matchups (score, site, ATS winner) -----
def build_hist_small(up_to_dt):
    """Return a compact dataframe of historical games with spreads & outcomes up to up_to_dt."""
    try:
        import nfl_data_py as nfl
    except Exception:
        return pd.DataFrame()

    df = nfl.import_schedules(list(range(2010, up_to_dt.year + 1)))
    df = df[df.get("game_type", "REG").fillna("REG") == "REG"].copy()
    df = tz_aware_kick(df)

    # Map abbr + derive closing home line
    df["home_abbr"] = df["home_team"].map(TEAM_ABBR).fillna(df["home_team"])
    df["away_abbr"] = df["away_team"].map(TEAM_ABBR).fillna(df["away_team"])
    df["line_home"] = df.apply(derive_home_line, axis=1)

    # Only games before our window end and with complete info
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df[(pd.notna(df["kickoff_et"])) & (df["kickoff_et"] < up_to_dt)]
    df = df[pd.notna(df["line_home"]) & pd.notna(df["home_score"]) & pd.notna(df["away_score"])].copy()

    # Margin vs spread (home perspective) + ATS winner label
    df["mvs_home"] = (df["home_score"] - df["away_score"]) + df["line_home"]
    df = df[pd.notna(df["mvs_home"])].copy()

    def _ats_winner(row):
        if row["mvs_home"] > 0:  # home covered
            return row["home_abbr"]
        elif row["mvs_home"] < 0:  # away covered
            return row["away_abbr"]
        return "PUSH"

    df["ATS winner"] = df.apply(_ats_winner, axis=1)
    df["date"] = df["kickoff_et"].dt.date.astype(str)
    df["final"] = (
        df["away_abbr"].astype(str) + " @ " + df["home_abbr"].astype(str) + "  " +
        df["away_score"].astype(int).astype(str) + "-" + df["home_score"].astype(int).astype(str)
    )

    keep = ["season", "week", "date", "final", "line_home", "ATS winner", "home_abbr", "away_abbr", "kickoff_et"]
    return df[keep].sort_values("kickoff_et").reset_index(drop=True)

def last5_table(df_small: pd.DataFrame, home_abbr: str, away_abbr: str) -> pd.DataFrame:
    """Filter last 5 head-to-head (regular season) games for the two teams."""
    if df_small is None or df_small.empty:
        return pd.DataFrame()
    mask = (
        ((df_small["home_abbr"] == home_abbr) & (df_small["away_abbr"] == away_abbr)) |
        ((df_small["home_abbr"] == away_abbr) & (df_small["away_abbr"] == home_abbr))
    )
    d = df_small[mask].copy().sort_values("kickoff_et")
    if d.empty:
        return pd.DataFrame()
    d = d.tail(5)  # last 5
    out = d[["season", "week", "date", "final", "line_home", "ATS winner"]].copy()
    out = out.rename(columns={"line_home": "line(home)"})
    return out.reset_index(drop=True)

# ---------- UI ----------
st.set_page_config(page_title="BetBot — NFL (ATS + H2H)", page_icon="🏈", layout="wide")
st.title("🏈 BetBot — NFL (ATS + H2H) · DK + FD")

with st.sidebar:
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = os.getenv("ODDS_API_KEY") or DEFAULT_ODDS_API_KEY
    api_key = st.text_input("The Odds API key", key="api_key")

    bet_type = st.radio("Bet type", ["Spread","Moneyline"], index=0)
    top_n = st.number_input("Top N picks", 1, 32, 10, 1)
    min_edge = st.slider("Only include EV ≥ (%)", 0.0, 10.0, 0.0, 0.1)/100.0

    st.markdown("#### Reality checks")
    stability_w = st.slider("Blend toward market (%)", 0, 40, 15, 5)/100.0
    clamp_pp = st.slider("Clamp shift (pp) around market", 0, 25, 12, 1)/100.0
    shrink = st.slider("Aggression (0=cautious, 1=raw)", 0.0, 1.0, 0.8, 0.05)
    max_dev_pp = st.slider("Max deviation from consensus (pp)", 0, 30, 15, 1)/100.0
    min_dog_p = st.slider("Underdog floor: min model P", 0.50, 0.70, 0.54, 0.01)

    st.markdown("#### Balance (after ranking)")
    min_favs = st.slider("Min favorites in list", 0, 10, 3, 1)
    max_dogs = st.slider("Max underdogs in list", 1, 20, 7, 1)
    interleave = st.toggle("Interleave favs/dogs", value=True)

    st.markdown("#### Week window")
    mode = st.radio("", ["Around today","Season+Week"], index=0, horizontal=True)
    def nfl_week1_start(season:int)->datetime:
        sept1=date(season,9,1); labor = sept1 + timedelta(days=(0-sept1.weekday())%7)
        return datetime.combine(labor + timedelta(days=3), time(0,0), tzinfo=ET)
    if mode=="Around today":
        off = st.slider("Week offset", -4, 18, 0, 1)
        now=datetime.now(ET)
        start = (now - timedelta(days=(now.weekday()-3)%7) + timedelta(weeks=off)).replace(hour=0,minute=0,second=0,microsecond=0)
        end = start + timedelta(days=5, hours=23, minutes=59, seconds=59)
        wlabel = f"Week around {start.date()} (Thu→Mon)"
    else:
        yr = st.number_input("Season", 2010, 2035, datetime.now(ET).year, 1)
        wk = st.number_input("Week", 1, 18, 1, 1)
        start = nfl_week1_start(int(yr)) + timedelta(days=7*(wk-1))
        end   = start + timedelta(days=5, hours=23, minutes=59, seconds=59)
        wlabel = f"Season {yr}, Week {wk} (Thu→Mon)"

    st.markdown("#### Stat key (plain English)")
    st.table(pd.DataFrame({
        "Stat":["Elo gap","Elo→spread","Value vs Elo","Rest edge","ATS form","H2H (last 5y)","Key number"],
        "Meaning":[
            "How strong the home side looks vs away (includes home edge)",
            "We turn Elo gap into points (≈28 Elo ≈ 1 point)",
            "How far the market line is from Elo-based line",
            "More days off helps; short week hurts",
            "Recent against-the-spread results; margin vs spread",
            "Head-to-head recently (overall & same stadium)",
            "3, 6, 7, 10, 14 matter for covers"
        ]
    }))

    scan = st.button("Scan Now")
    learn = st.button("Apply feedback from picks_log.csv")

# load bundle
def load_bundle(path="model_nfl_bundle.pkl"):
    if not os.path.exists(path): 
        st.error("Missing model bundle. Run `python train_nfl_models.py` first."); st.stop()
    try:
        with open(path,"rb") as f: pack=pickle.load(f)
    except Exception as e:
        st.error("Failed to load bundle (likely sklearn mismatch). Run Streamlit with the same venv used to train.")
        st.exception(e); st.stop()
    return pack, path, datetime.fromtimestamp(os.path.getmtime(path), tz=ET)
bundle, bundle_path, bundle_mtime = load_bundle()
edges = bucket_edges_from_bundle(bundle)

# feedback (optional)
def load_feedback_safe():
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH,"rb") as f: return pickle.load(f)
        except Exception: pass
    return {}
fb = load_feedback_safe()

# button: learn now
if learn:
    st.info("Run `python learn_from_picks.py` in your project folder. This script grades your past picks and updates model_feedback.pkl. The UI reads it automatically on reload.")

if not st.session_state.get("api_key"): st.stop()
if not scan: st.stop()
api_key = st.session_state["api_key"]

# pull odds
try:
    data_sp = fetch_odds(api_key, markets="spreads")
    data_ml = fetch_odds(api_key, markets="h2h")
except Exception as e:
    st.error(f"Odds fetch error: {e}"); st.stop()

# filter events in window
def to_dt(ev): 
    t=to_et(ev.get("commence_time"))
    return t if t and (start<=t<=end) else None
events_sp = [ev for ev in data_sp if to_dt(ev)]
events_ml = [ev for ev in data_ml if to_dt(ev)]
sp_map = {(ev["home_team"], ev["away_team"]): ev for ev in events_sp}

# history state (for features)
h2h_years = int((bundle.get("metadata") or {}).get("h2h_years", 5))
state = build_history_state(end, h2h_years) or {}
if not state:
    st.error("nfl_data_py not installed. `pip install nfl-data-py pyarrow`"); st.stop()

# Precompute compact H2H history table once (for last-5)
df_hist_small = build_hist_small(end)

rows=[]

# ---- SPREAD ----
if bet_type=="Spread":
    sp_obj=bundle.get("spread", {})
    if not sp_obj:
        st.error("Spread model not in bundle."); st.stop()
    sp_pack=sp_obj.get("pack") or {}; sp_feats=sp_obj.get("features", [])
    base=sp_pack.get("base")
    if base is None:
        st.error("Spread base model missing."); st.stop()

    for ev in events_ml:
        home,away = ev["home_team"], ev["away_team"]
        when = to_et(ev["commence_time"])
        sp_ev = sp_map.get((home,away))
        if not sp_ev: continue
        by_point = offers_by_point(sp_ev, home, away)
        if not by_point: continue

        for line_home, offers in by_point.items():
            # model prob
            X, feats = build_spread_feats(sp_feats, home, away, float(line_home), when, state)
            p_raw = float(base.predict_proba(X.reshape(1,-1))[0,1])
            p_raw = float(np.clip(p_raw, 0.05, 0.95))
            p_cal = apply_bucket_cal_single(p_raw, feats["abs_line"], feats["home_fav"], edges, sp_pack.get("bucket_cals", {}))
            # feedback cal
            fav_flag = 1 if feats["home_fav"]==1.0 else 0
            p_cal = apply_feedback_cal(p_cal, "Spread", fav_flag, feats["abs_line"], fb)

            # consensus
            p_cons = consensus_prob(offers)
            # blend + clamp, then shrink
            p_used_home = blend_prob_clamped(p_cons, p_cal, stability_w, clamp_pp)
            p_used_home = shrink_toward_half(p_used_home, shrink)
            p_used_away = 1 - p_used_home

            # dev cap (drop if too far from market)
            if p_cons is not None:
                dev_home = abs(p_used_home - p_cons)
                dev_away = abs(p_used_away - (1 - p_cons))
                if dev_home > max_dev_pp: p_used_home = None
                if dev_away > max_dev_pp: p_used_away = None

            best_h=best_price(offers, "home"); best_a=best_price(offers, "away")
            if not best_h or not best_a: continue
            dec_h, dec_a = american_to_decimal(best_h["price"]), american_to_decimal(best_a["price"])

            # push credit on 3 / 7
            push = 0.02 if feats["key3"]==1.0 else (0.01 if feats["key7"]==1.0 else 0.0)

            # classify fav/dog using market line
            side_home_is_fav = True if float(line_home)<0 else False

            # HOME pick
            if p_used_home is not None:
                ev_h = p_used_home*dec_h - 1 + push
                _side = "fav" if side_home_is_fav else "dog"
                if (_side=="dog") and (p_used_home < min_dog_p): ev_h = -9.99
                rows.append({
                    "Time (ET)": when.strftime("%Y-%m-%d %I:%M %p ET"), "Match": f"{away} @ {home}",
                    "Pick": f"{home} {pretty_line(line_home)} @ {best_h['book']} {best_h['price']:+}",
                    "P(used)": round(p_used_home,3), "EV %": round(ev_h*100,2), "Line": pretty_line(line_home),
                    "_side": _side,
                    "_detail":{"type":"Spread","home":home,"away":away,"p_used":p_used_home,"ev_pct":ev_h*100,
                               "book":best_h["book"],"price":best_h["price"],"line_for_pick":float(line_home),
                               "market_home_line": float(line_home), "pick_side":"home","feats":feats}
                })

            # AWAY pick
            if p_used_away is not None:
                ev_a = p_used_away*dec_a - 1 + push
                _side = "fav" if (not side_home_is_fav) else "dog"
                if (_side=="dog") and (p_used_away < min_dog_p): ev_a = -9.99
                rows.append({
                    "Time (ET)": when.strftime("%Y-%m-%d %I:%M %p ET"), "Match": f"{away} @ {home}",
                    "Pick": f"{away} {pretty_line(-line_home)} @ {best_a['book']} {best_a['price']:+}",
                    "P(used)": round(p_used_away,3), "EV %": round(ev_a*100,2), "Line": pretty_line(-line_home),
                    "_side": _side,
                    "_detail":{"type":"Spread","home":home,"away":away,"p_used":p_used_away,"ev_pct":ev_a*100,
                               "book":best_a["book"],"price":best_a["price"],"line_for_pick":float(-line_home),
                               "market_home_line": float(line_home), "pick_side":"away","feats":feats}
                })

# ---- MONEYLINE ----
else:
    ml_obj=bundle.get("moneyline", {})
    ml_model=ml_obj.get("model"); ml_feats=ml_obj.get("features", [])
    if not ml_model or not ml_feats:
        st.error("Moneyline model not in bundle."); st.stop()

    for ev in events_ml:
        home,away = ev["home_team"], ev["away_team"]
        when=to_et(ev["commence_time"])

        # collect best prices + consensus
        best_h=None; best_a=None; ph_list=[]
        for b in ev.get("bookmakers",[]) or []:
            title=b.get("title",b.get("key","?"))
            if title not in BOOK_ALLOW: continue
            for m in b.get("markets",[]):
                if m.get("key")!="h2h": continue
                hp=ap=None
                for o in m.get("outcomes",[]):
                    nm,pr=o.get("name"),o.get("price")
                    if nm==home and isinstance(pr,int): hp=pr
                    if nm==away and isinstance(pr,int): ap=pr
                if hp is not None and ap is not None:
                    da, db = american_to_decimal(hp), american_to_decimal(ap)
                    s = 1/da + 1/db
                    ph_list.append((1/da)/s)
                    if (best_h is None) or (hp>best_h["price"]): best_h={"price":hp,"book":title}
                    if (best_a is None) or (ap>best_a["price"]): best_a={"price":ap,"book":title}
        if not best_h or not best_a: continue
        p_cons = float(np.mean(ph_list)) if ph_list else None

        X, feats = build_ml_feats(ml_feats, home, away, when, state)
        p_model = float(ml_model.predict_proba(X.reshape(1,-1))[0,1])

        # blend+clamp+shrink
        p_home = blend_prob_clamped(p_cons, p_model, stability_w, clamp_pp)
        p_home = shrink_toward_half(p_home, shrink)
        p_away = 1 - p_home

        # dev cap
        if p_cons is not None:
            if abs(p_home - p_cons) > max_dev_pp: p_home=None
            if abs(p_away - (1-p_cons)) > max_dev_pp: p_away=None

        # classify fav/dog using consensus (if present), else model
        home_is_fav = (p_cons is not None and p_cons>=0.5) or (p_cons is None and p_home is not None and p_home>=0.5)

        dec_h, dec_a = american_to_decimal(best_h["price"]), american_to_decimal(best_a["price"])

        if p_home is not None:
            ev_h = p_home*dec_h - 1
            _side = "fav" if home_is_fav else "dog"
            if (_side=="dog") and (p_home < min_dog_p): ev_h = -9.99
            rows.append({
                "Time (ET)": when.strftime("%Y-%m-%d %I:%M %p ET"), "Match": f"{away} @ {home}",
                "Pick": f"{home} ML @ {best_h['book']} {best_h['price']:+}",
                "P(used)": round(p_home,3), "EV %": round(ev_h*100,2), "Line":"-",
                "_side": _side,
                "_detail":{"type":"Moneyline","home":home,"away":away,"p_used":p_home,"ev_pct":ev_h*100,
                           "book":best_h["book"],"price":best_h["price"],"pick_team":home}
            })
        if p_away is not None:
            ev_a = p_away*dec_a - 1
            _side = "fav" if (not home_is_fav) else "dog"
            if (_side=="dog") and (p_away < min_dog_p): ev_a = -9.99
            rows.append({
                "Time (ET)": when.strftime("%Y-%m-%d %I:%M %p ET"), "Match": f"{away} @ {home}",
                "Pick": f"{away} ML @ {best_a['book']} {best_a['price']:+}",
                "P(used)": round(p_away,3), "EV %": round(ev_a*100,2), "Line":"-",
                "_side": _side,
                "_detail":{"type":"Moneyline","home":home,"away":away,"p_used":p_away,"ev_pct":ev_a*100,
                           "book":best_a["book"],"price":best_a["price"],"pick_team":away}
            })

# ---- RANK / FILTER / BALANCE ----
rows = [r for r in rows if np.isfinite(r["EV %"])]
rows = sorted(rows, key=lambda r: r["EV %"], reverse=True)
filtered = [r for r in rows if r["EV %"] >= (min_edge*100)]

# enforce min favorites & max underdogs after ranking
favs=[r for r in filtered if r.get("_side")=="fav"]
dogs=[r for r in filtered if r.get("_side")=="dog"]

# ensure at least min_favs
if len(favs) < min_favs:
    extras=[r for r in rows if (r.get("_side")=="fav" and r not in filtered)]
    need=min_favs-len(favs); filtered += extras[:need]

# cap dogs
filtered_f=[r for r in filtered if r.get("_side")=="fav"]
filtered_d=[r for r in filtered if r.get("_side")=="dog"][:max_dogs]
filtered = filtered_f + filtered_d

# interleave if asked
if interleave:
    favs=[r for r in filtered if r.get("_side")=="fav"]; dogs=[r for r in filtered if r.get("_side")=="dog"]
    mix=[]; n=max(len(favs),len(dogs))
    for i in range(n):
        if i<len(favs): mix.append(favs[i])
        if i<len(dogs): mix.append(dogs[i])
    filtered=mix

# fill to top_n
if len(filtered) < int(top_n):
    pool=[r for r in rows if r not in filtered]
    filtered += pool[:(int(top_n)-len(filtered))]
filtered = filtered[:int(top_n)]

# ---- OUTPUT ----
st.markdown(f"**{bet_type}** · {wlabel} · Showing **{len(filtered)}** of requested **{int(top_n)}** · Model built: `{(bundle.get('metadata') or {}).get('built_at','?')}`")
st.dataframe(pd.DataFrame([{k:v for k,v in r.items() if not k.startswith('_')} for r in filtered]), use_container_width=True)

# log picks
def append_picks_log(recs):
    rec_df = []
    for r in recs:
        d=r["_detail"]; home=d["home"]; away=d["away"]
        season = end.year
        row = {
            "picked_at": datetime.now(ET).isoformat(),
            "season": season,
            "home_team": home,
            "away_team": away,
            "market": d["type"],
            "p_used": float(d["p_used"]),
            "ev_pct": float(d["ev_pct"]),
            "book": d["book"],
            "price": int(d["price"]),
        }
        if d["type"]=="Spread":
            row.update({
                "pick_side": d["pick_side"],
                "pick_team": home if d["pick_side"]=="home" else away,
                "market_home_line": float(d["market_home_line"]),
                "line_for_pick": float(d["line_for_pick"]),
            })
        else:
            row.update({"pick_team": d["pick_team"]})
        rec_df.append(row)
    df=pd.DataFrame(rec_df)
    if os.path.exists(PICKS_LOG):
        old=pd.read_csv(PICKS_LOG)
        df=pd.concat([old,df], ignore_index=True)
    df.to_csv(PICKS_LOG, index=False)

append_picks_log(filtered)

# detail cards + last 5 H2H
def human_narrative_spread(home, away, feats, p_used, ev_pct, pick_text):
    bits=[]
    bits.append(f"**{pick_text}** — model thinks this covers **{p_used:.2f}** of the time (EV **{ev_pct:.1f}%**).")
    if abs(feats.get("elo_spread",0.0))>=0.5:
        side = "home" if feats["elo_spread"]<0 else "away"
        pts = abs(feats["elo_spread"])
        bits.append(f"Elo leans **{side}** by ~**{pts:.1f}** on neutral.")
    if abs(feats.get("rest_days_diff",0))>=2:
        rd=feats["rest_days_diff"]; bits.append(f"Rest edge: **{rd:+.0f} day(s)** (home−away).")
    if feats.get("divisional_game",0.0)==1.0:
        bits.append("It’s a **divisional** game (often tighter).")
    if feats.get("h2h_any_n",0)>0:
        bits.append(f"Recent head-to-head ATS rate (home side): **{feats['h2h_any_rate']:.2f}** over **{int(feats['h2h_any_n'])}** games.")
    return " ".join(bits)

def human_narrative_ml(home, away, feats, p_used, ev_pct, pick_text):
    bits=[f"**{pick_text}** — win chance **{p_used:.2f}** (EV **{ev_pct:.1f}%**)."]
    if abs(feats.get("elo_diff_home_minus_away",0))>=40:
        bits.append(f"Elo gap (home−away) **{feats['elo_diff_home_minus_away']:+.0f}**.")
    if abs(feats.get("rest_days_diff",0))>=2:
        bits.append(f"Rest edge **{feats['rest_days_diff']:+.0f} day(s)**.")
    return " ".join(bits)

for r in filtered:
    d=r["_detail"]; home=d["home"]; away=d["away"]
    if d["type"]=="Spread":
        feats=d["feats"]; paragraph = human_narrative_spread(home, away, feats, d["p_used"], d["ev_pct"], r["Pick"])
    else:
        paragraph = human_narrative_ml(home, away, d.get("feats",{}), d["p_used"], d["ev_pct"], r["Pick"])

    st.markdown(f"""
<div style="border:1px solid #e5e7eb;border-radius:12px;padding:12px 14px;margin:8px 0;">
  <div style="color:#6b7280;font-size:13px;">{r['Time (ET)']} · {d['type']}</div>
  <div><strong>{r['Match']}</strong> — {r['Pick']}</div>
  <div style="color:#6b7280;font-size:13px;">P(used): {d['p_used']:.3f} · EV: {d['ev_pct']:.2f}%</div>
  <hr style="border:none;border-top:1px solid #f2f2f2;"/>
  <div>{paragraph}</div>
</div>
""", unsafe_allow_html=True)

    # Last 5 head-to-head (REG season)
    try:
        if isinstance(df_hist_small, pd.DataFrame) and not df_hist_small.empty:
            H = TEAM_ABBR.get(home, home)
            A = TEAM_ABBR.get(away, away)
            tbl = last5_table(df_hist_small, H, A)
            if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                st.caption("Last 5 matchups (regular season):")
                st.dataframe(tbl, use_container_width=True)
    except Exception:
        pass
