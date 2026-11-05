"""
확장 팩터 연구 v2 — NYSE 시총 20% 유니버스 · 시가총액 가중 · 상폐 분류 정책
==========================================================================
[설정]
  - 유니버스: 매월 NYSE 시총 20% 분위수 미만 제외 + SPAC 제외 + 그 달 상장폐지 종목 제외
  - 가중: 형성월 시가총액 비례(VW). 분위 포트폴리오도 VW
  - 상폐: 일반·불명확 상폐 −50%, M&A·재분류는 CIZ 기록 수익률 (delisting.py)
  - 비용: 명목 편도 회전율 × 왕복 20bp, 보유 2003-02~2020-12 (215개월)
  - 표본 분할: 조합·국면·구성 선택은 IS(2003-02~2011-12) 성과만 사용, OOS(2012-01~)는 확인용

[단계]  py -3.13 factor_research_v2.py [stage ...]
  s0 상폐 정책 영향   s1 단일 신호 전수(Top-k·5분위·10분위)   s2 조합 탐색
  s3 시장 국면 전환형  s4 Top-k 구성 고도화(k·상한·버퍼·중첩)   s5 최종 유의성
"""
import itertools                             # 조합
import json                                  # 저장
import os                                    # 경로
import pickle                                # 캐시
import sys                                   # 인자
import time                                  # 시간
import warnings                              # 경고
import numpy as np                           # 수치
import pandas as pd                          # 데이터프레임
from scipy import stats                      # 분포
import selection_eval as se                  # 프레임워크
from signals import INFO                     # 기존 신호 41개
from signals_v2 import INFO_V2, market_state  # 새 신호 45개 + 국면

warnings.filterwarnings("ignore")                             # 경고 숨김
if hasattr(sys.stdout, "reconfigure"):                        # 콘솔 한글
    sys.stdout.reconfigure(encoding="utf-8")                  # UTF-8

PROFILE = os.environ.get("FR_PROFILE", "vw")                  # 가중 프로필: vw(순수 시총가중) / cap2(시총가중, 종목당 최대 2/n)
DATA_DIR = os.path.join(se.OUT_DIR, "v2")                    # 데이터 캐시 폴더(프로필 공통)
OUT = os.path.join(se.OUT_DIR, "v2" if PROFILE == "vw" else f"v2_{PROFILE}")  # 결과 폴더
os.makedirs(DATA_DIR, exist_ok=True)                          # 캐시 폴더 생성
os.makedirs(OUT, exist_ok=True)                               # 생성
CFG = se.Config(vol_cut=0.0, size_cut=0.20, size_basis="nyse", weighting="vw", exclude_delist_month=True,
                vw_cap_mult=2.0 if PROFILE == "cap2" else 0.0)  # 공통 설정
IS_END, OOS_START = "2011-12", "2012-01"                      # 표본 분할
KS = (20, 30, 50, 100)                                        # k 후보
FOCUS_K = 50                                                  # 조합 선택 기준 k
ALL_INFO = {**{k: v for k, v in INFO.items()}, **INFO_V2}     # 전체 신호 86개
TRIALS_FILE = os.path.join(OUT, "trials.json")                # 시행 횟수 기록


# ---------------------------------------------------------------------------
# 데이터·공통 도구
# ---------------------------------------------------------------------------
def load_data(policy="classified"):
    """MarketData 적재(캐시 사용)"""
    path = os.path.join(DATA_DIR, f"marketdata_{policy}.pkl")  # 캐시 경로
    if os.path.exists(path):                                  # 있으면
        return pickle.load(open(path, "rb"))                  # 읽기
    d = se.load_all(delist_policy=policy)                     # 적재
    pickle.dump(d, open(path, "wb"))                          # 저장
    return d                                                  # 반환


def add_trials(n, tag):
    """다중검정 보정용 누적 시행 횟수 기록"""
    t = json.load(open(TRIALS_FILE)) if os.path.exists(TRIALS_FILE) else {}  # 기존
    t[tag] = n                                                # 갱신
    json.dump(t, open(TRIALS_FILE, "w"), ensure_ascii=False, indent=1)  # 저장


def total_trials():
    """누적 시행 횟수"""
    return sum(json.load(open(TRIALS_FILE)).values()) if os.path.exists(TRIALS_FILE) else 1  # 합


_RANKS = {}                                                   # 유니버스 내 백분위 캐시


def ranks_of(name, d, cfg=CFG):
    """신호 → 유니버스 내 횡단면 백분위(클수록 좋음)"""
    if name not in _RANKS:                                    # 없으면
        _RANKS[name] = se.prepare_score(ALL_INFO[name][0](d), d, cfg).rank(axis=1, pct=True)  # 계산
    return _RANKS[name]                                       # 반환


def composite(d, names, cfg=CFG):
    """백분위 동일 평균(구성 지표 하나라도 없으면 제외)"""
    return sum(ranks_of(n, d, cfg) for n in names) / len(names)  # 평균


def run_weights(w, d, cfg=CFG):
    """비중 → 월 수익률 표(gross/net/turnover)"""
    return se.portfolio_returns(w, d, cfg)                    # 계산


def _select_loop(s, d, k, buffer, ind_cap):
    """월별 선택(버퍼·산업 상한). 기존 보유는 순위 k×buffer 안이면 유지, 산업(SIC2)별 최대 ceil(k×ind_cap)종목"""
    rank = s.rank(axis=1, ascending=False, method="first")    # 순위(1=최고)
    sic2 = np.floor(d.x["siccd"].reindex(index=s.index, columns=s.columns) / 100)  # 산업
    lim = int(np.ceil(k * ind_cap)) if ind_cap else k         # 산업별 최대 종목 수
    sel = np.zeros(s.shape, dtype=bool)                       # 선택
    cols = s.columns                                          # 종목
    prev = set()                                              # 직전 보유
    for i, t in enumerate(s.index):                           # 월마다
        r = rank.loc[t].dropna().sort_values()                # 순위순 종목
        g = sic2.loc[t]                                       # 산업
        cnt, cur = {}, []                                     # 산업별 개수, 이번 보유
        order = ([p for p in r.index if p in prev and r[p] <= k * buffer] if buffer else []) + list(r.index)  # 기존 보유 우선 → 순위순
        for p in order:                                       # 후보 순회
            if len(cur) >= k:                                 # 다 찼으면
                break                                         # 끝
            if p in cur:                                      # 이미 선택
                continue                                      # 건너뜀
            gi = g.get(p, np.nan)                             # 산업
            if ind_cap and cnt.get(gi, 0) >= lim:             # 산업 상한 도달
                continue                                      # 건너뜀
            cur.append(p)                                     # 선택
            cnt[gi] = cnt.get(gi, 0) + 1                      # 개수
        sel[i, cols.get_indexer(cur)] = True                  # 기록
        prev = set(cur)                                       # 갱신
    return pd.DataFrame(sel, index=s.index, columns=cols)     # 반환


def topk_weights(score, d, k, cfg=CFG, cap=None, buffer=None, overlap=1, ind_cap=None):
    """점수 → Top-k VW 비중 (cap: 종목당 고정 상한, buffer: 기존 보유 유지 순위 배수,
    overlap: 중첩 보유 개월 수, ind_cap: 한 산업(SIC2) 최대 비율). 종목당 2/k 상한은 cfg.vw_cap_mult로 적용"""
    s = se.prepare_score(score, d, cfg)                       # 유니버스 적용
    mcap = d.x["mcap"] if cfg.weighting == "vw" else None     # 가중 기준
    cap = cfg.vw_cap if cap is None else cap                  # 상한
    if not buffer and not ind_cap:                            # 단순 Top-k
        w = se.weights_topk(s, k, mcap, cap, cfg.vw_cap_mult)  # 기본 Top-k
    else:                                                     # 버퍼·산업 상한 규칙
        w = se.weigh(_select_loop(s, d, k, buffer, ind_cap), mcap, cap, cfg.vw_cap_mult)  # 비중
    if overlap > 1:                                           # 중첩 보유
        w = sum(w.shift(j).fillna(0.0) for j in range(overlap)) / overlap  # 최근 overlap개월 포트폴리오 평균
        w = w.div(w.sum(1).replace(0, np.nan), axis=0).fillna(0.0)  # 초기 구간 정규화
    return w                                                  # 반환


def metrics(r, d, turnover=None, bench=None):
    """성과 지표 한 줄"""
    rf = d.ff["RF"].reindex(r.index)                          # 무위험
    L = CFG.lags(len(r))                                      # NW 시차
    reg = se.ff3_regression(r, d.ff, L)                       # FF3
    bench = d.ff["MKT"].reindex(r.index) if bench is None else bench  # 시장
    ds = se.sharpe_diff_test(r, bench, rf, L)                 # ΔSR
    a, b = r.loc[:IS_END], r.loc[OOS_START:]                  # 분할
    out = {"연수익": se.ann_return(r), "변동성": se.ann_vol(r), "Sharpe": se.sharpe(r, rf),
           "MDD": se.max_drawdown(r), "ΔSR": ds["dSR"], "p(LW)": ds["p_LW"],
           "베타": reg["beta"], "α(연)": reg["alpha_ann"], "α t": reg["alpha_t"],
           "IS Sharpe": se.sharpe(a, rf), "OOS Sharpe": se.sharpe(b, rf),
           "IS MDD": se.max_drawdown(a), "OOS MDD": se.max_drawdown(b)}  # 지표
    if turnover is not None:                                  # 회전율
        out["회전율(월)"] = turnover.mean()                    # 평균
    return out                                                # 반환


def group_analysis(score, d, n):
    """n분위 VW(비용 전) → 분위별 지표 + 상위−하위 스프레드 검정"""
    s = se.prepare_score(score, d, CFG)                       # 유니버스 적용
    W = se.weights_groups(s, n, d.x["mcap"], CFG.vw_cap_mult)  # VW 분위(배수 상한 동일 적용)
    R = pd.DataFrame({f"G{g}": run_weights(w, d)["gross"] for g, w in W.items()})  # 분위 수익률
    rf = d.ff["RF"].reindex(R.index)                          # 무위험
    L = CFG.lags(len(R))                                      # 시차
    rows = {}                                                 # 분위별
    for c in R:                                               # 각 분위
        reg = se.ff3_regression(R[c], d.ff, L)                # FF3
        rows[c] = {"연수익": se.ann_return(R[c]), "Sharpe": se.sharpe(R[c], rf),
                   "변동성": se.ann_vol(R[c]), "α(연)": reg["alpha_ann"], "α t": reg["alpha_t"], "베타": reg["beta"]}  # 지표
    tab = pd.DataFrame(rows).T                                # 표
    sp = R.iloc[:, 0] - R.iloc[:, -1]                         # 상위 − 하위
    m, t, p = se.nw_mean_test(sp, L)                          # 원수익률 차이
    reg = se.ff3_regression(sp + rf, d.ff, L)                 # 위험조정 차이
    mi, ti, _ = se.nw_mean_test(sp.loc[:IS_END], L)           # IS 차이
    rfi = rf.loc[:IS_END]                                     # IS 무위험
    ai = se.ff3_regression(sp.loc[:IS_END] + rfi, d.ff, L)["alpha_t"]  # IS 위험조정 차이
    mo, to, _ = se.nw_mean_test(sp.loc[OOS_START:], L)        # OOS 차이
    rho = stats.spearmanr(np.arange(n, 0, -1), R.mean().values)[0]  # 단조성
    sh = tab["Sharpe"].values                                 # 분위별 Sharpe
    summ = {"top 연수익": tab["연수익"].iloc[0], "bottom 연수익": tab["연수익"].iloc[-1],
            "차이(월)": m, "차이 t": t, "차이 p": p, "차이 α t": reg["alpha_t"], "차이 α(연)": reg["alpha_ann"],
            "IS 차이 t": ti, "OOS 차이 t": to, "IS 차이 α t": ai,
            "top Sharpe": sh[0], "bottom Sharpe": sh[-1], "Sharpe 차이": sh[0] - sh[-1],
            "top α t": tab["α t"].iloc[0], "Spearman ρ": rho,
            "Sharpe ρ": stats.spearmanr(np.arange(n, 0, -1), sh)[0]}  # 요약
    return tab, summ, R                                       # 반환


def eval_topk(score, d, ks=KS, **kw):
    """Top-k(VW) k별 지표와 수익률"""
    rows, rets = {}, {}                                       # 결과
    for k in ks:                                              # k마다
        pr = run_weights(topk_weights(score, d, k, **kw), d)  # 수익률
        rows[k] = metrics(pr["net"], d, pr["turnover"])       # 지표
        rets[k] = pr["net"]                                   # 저장
    return pd.DataFrame(rows).T, pd.DataFrame(rets)           # 반환


def is_sharpe(score, d, k=FOCUS_K, **kw):
    """IS 구간 Top-k Sharpe (조합 선택용 빠른 평가)"""
    r = run_weights(topk_weights(score, d, k, **kw), d)["net"]  # 수익률
    return se.sharpe(r.loc[:IS_END], d.ff["RF"]), r           # IS Sharpe, 전체 수익률


# ---------------------------------------------------------------------------
# s0. 상폐 정책 영향 (기존 NYSE20 추천 조합 기준)
# ---------------------------------------------------------------------------
def stage0():
    """기존 추천(저베타FP+저MAX3m+저회전율)으로 상폐 정책·가중 방식별 비교"""
    names = ["low_beta_fp", "low_max_3m", "low_turnover"]     # 기존 추천
    rows = {}                                                 # 결과
    for pol in ["ciz", "classified"]:                         # 상폐 정책
        d = load_data(pol)                                    # 데이터
        _RANKS.clear()                                        # 캐시 비움(데이터가 다름)
        for wt in ["ew", "vw"]:                               # 가중
            cfg = se.Config(vol_cut=0.0, size_cut=0.20, size_basis="nyse", weighting=wt,
                            exclude_delist_month=(pol == "classified"))  # 설정
            sc = sum(se.prepare_score(INFO[n][0](d), d, cfg).rank(axis=1, pct=True) for n in names) / 3  # 조합 점수
            s = se.prepare_score(sc, d, cfg)                  # 유니버스
            w = se.weights_topk(s, 50, d.x["mcap"] if wt == "vw" else None)  # Top-50
            pr = se.portfolio_returns(w, d, cfg)              # 수익률
            rows[f"{pol} · {wt.upper()} · Top-50"] = metrics(pr["net"], d, pr["turnover"])  # 지표
        # 보유 종목 중 상폐 영향 건수
        if pol == "classified":                               # 분류 정책일 때
            dr = d.delist_ret.shift(-1).reindex(index=w.index, columns=w.columns)  # 다음달 상폐
            hit = (w > 0) & dr.notna()                        # 보유 중 상폐
            cls = d.x["delist_events"].set_index(["m", "PERMNO"])["class"]  # 분류
            hits = [(w.index[i] + 1, w.columns[j]) for i, j in zip(*np.where(hit.values))]  # 실제 라벨
            print("VW Top-50 보유 중 상폐 이벤트:", pd.Series([cls.get(h, "?") for h in hits]).value_counts().to_dict())  # 출력
    t = pd.DataFrame(rows).T                                  # 표
    t.to_csv(os.path.join(OUT, "s0_delist_policy.csv"), encoding="utf-8-sig")  # 저장
    print(t[["연수익", "변동성", "Sharpe", "MDD", "α t", "IS Sharpe", "OOS Sharpe", "회전율(월)"]].round(3))  # 출력
    ev = load_data("classified").x["delist_events"]           # 이벤트
    ev["class"].value_counts().to_csv(os.path.join(OUT, "s0_delist_counts.csv"), encoding="utf-8-sig")  # 분류 건수


# ---------------------------------------------------------------------------
# s1. 단일 신호 전수 검증
# ---------------------------------------------------------------------------
def stage1():
    """86개 신호: Top-k VW(k=20/30/50/100) + 5분위·10분위 VW"""
    d = load_data()                                           # 데이터
    topk_rows, q5_rows, d10_rows = [], [], []                 # 결과
    q5_tabs = {}                                              # 분위표 저장
    for i, (n, (f, desc, cat)) in enumerate(ALL_INFO.items()):  # 신호마다
        t0 = time.time()                                      # 시간
        sc = f(d)                                             # 신호
        tk, _ = eval_topk(sc, d)                              # Top-k
        for k, row in tk.iterrows():                          # k별
            topk_rows.append({"name": n, "설명": desc, "범주": cat, "k": k, **row})  # 저장
        tab5, s5, _ = group_analysis(sc, d, 5)                # 5분위
        tab10, s10, _ = group_analysis(sc, d, 10)             # 10분위
        q5_rows.append({"name": n, "설명": desc, "범주": cat, **s5})  # 저장
        d10_rows.append({"name": n, "설명": desc, "범주": cat, **s10})  # 저장
        q5_tabs[n] = {"q5": tab5, "d10": tab10}               # 분위표
        best = tk["Sharpe"].max()                             # 최고 Sharpe
        print(f"[{i+1}/{len(ALL_INFO)}] {n:18s} bestSR {best:.2f}  Q1−Q5 t {s5['차이 t']:+.2f} αt {s5['차이 α t']:+.2f}  ({time.time()-t0:.0f}s)", flush=True)  # 진행
    T = pd.DataFrame(topk_rows)                               # Top-k 표
    Q = pd.DataFrame(q5_rows).set_index("name")               # 5분위 요약
    D = pd.DataFrame(d10_rows).set_index("name")              # 10분위 요약
    T.to_csv(os.path.join(OUT, "s1_topk.csv"), index=False, encoding="utf-8-sig")  # 저장
    Q.to_csv(os.path.join(OUT, "s1_quintile.csv"), encoding="utf-8-sig")  # 저장
    D.to_csv(os.path.join(OUT, "s1_decile.csv"), encoding="utf-8-sig")  # 저장
    pickle.dump(q5_tabs, open(os.path.join(OUT, "s1_group_tables.pkl"), "wb"))  # 분위표
    add_trials(len(ALL_INFO) * len(KS), "s1_single_topk")     # 시행 기록


# ---------------------------------------------------------------------------
# s2. 조합 탐색 (IS만으로 선택)
# ---------------------------------------------------------------------------
def stage2(n_sr=12, n_sp=6):
    """후보 = IS Top-50 Sharpe 상위 n_sr + IS 5분위 원수익률 차이 t 상위 n_sp → 쌍·3중 순위평균"""
    d = load_data()                                           # 데이터
    T = pd.read_csv(os.path.join(OUT, "s1_topk.csv"))         # 단일 결과
    Q = pd.read_csv(os.path.join(OUT, "s1_quintile.csv"), index_col=0)  # 5분위
    t50 = T[T["k"] == FOCUS_K].set_index("name")              # Top-50
    cand_sr = t50["IS Sharpe"].sort_values(ascending=False).index[:n_sr].tolist()  # 위험조정 상위
    cand_sp = [n for n in Q["IS 차이 t"].sort_values(ascending=False).index if n not in cand_sr][:n_sp]  # 분위 격차 상위
    cands = cand_sr + cand_sp                                 # 후보
    print("후보(Sharpe):", cand_sr, "\n후보(분위격차):", cand_sp)  # 출력
    rows = []                                                 # 결과
    combos = [c for r in (2, 3) for c in itertools.combinations(cands, r)]  # 조합
    for i, c in enumerate(combos):                            # 조합마다
        sr, r = is_sharpe(composite(d, c), d)                 # IS Sharpe
        rf = d.ff["RF"]                                       # 무위험
        rows.append({"combo": "+".join(c), "n": len(c), "IS Sharpe": sr, "Sharpe": se.sharpe(r, rf),
                     "OOS Sharpe": se.sharpe(r.loc[OOS_START:], rf), "MDD": se.max_drawdown(r), "연수익": se.ann_return(r)})  # 저장
        if (i + 1) % 100 == 0:                                # 진행
            print(f"  {i+1}/{len(combos)}", flush=True)       # 출력
    C = pd.DataFrame(rows).sort_values("IS Sharpe", ascending=False)  # 정렬
    C.to_csv(os.path.join(OUT, "s2_combos.csv"), index=False, encoding="utf-8-sig")  # 저장
    json.dump({"cand_sr": cand_sr, "cand_sp": cand_sp}, open(os.path.join(OUT, "s2_cands.json"), "w"))  # 후보 저장
    add_trials(len(combos), "s2_combos")                      # 시행 기록
    print(C.head(20).round(3).to_string(index=False))         # 출력


# ---------------------------------------------------------------------------
# s3. 시장 국면 전환형 (공격 ↔ 방어)
# ---------------------------------------------------------------------------
AGG_SETS = {                                                  # 공격형 후보(문헌 기반 사전 정의)
    "AGG_MOM_BETA": ["mom_12_1", "high_beta_d12"],            # 모멘텀 + 고베타
    "AGG_TREND": ["high_52w", "mom_12_1", "trend_ma"],        # 추세 3종
    "AGG_MOM": ["mom_12_1"],                                  # 순수 모멘텀
}
REGIMES = ["bull_trend12", "bull_ma10", "bull_lowvol", "bull_breadth", "bull_calm_sent", "bull_score"]  # 국면 규칙


def stage3():
    """점수 = s_t·공격형 백분위 + (1−s_t)·방어형 백분위. 방어형 = IS 최고 조합(s2), s_t = 국면 지표(t월 말 기준)"""
    d = load_data()                                           # 데이터
    C = pd.read_csv(os.path.join(OUT, "s2_combos.csv"))       # 조합 결과
    defense = C.iloc[0]["combo"].split("+")                   # IS 최고 조합 = 방어형(저위험 계열일 것)
    ms = market_state(d).reindex(CFG.form_months())           # 국면(형성월)
    ms.to_csv(os.path.join(OUT, "s3_market_state.csv"), encoding="utf-8-sig")  # 저장
    D = composite(d, defense)                                 # 방어 점수
    rows = []                                                 # 결과
    rf = d.ff["RF"]                                           # 무위험
    for an, names in AGG_SETS.items():                        # 공격형
        A = composite(d, names)                               # 공격 점수
        for rg in REGIMES:                                    # 국면 규칙
            s = ms[rg].fillna(0.0)                            # 국면(결측이면 방어)
            sc = A.mul(s, axis=0) + D.mul(1 - s, axis=0)      # 혼합 점수
            sr, r = is_sharpe(sc, d)                          # 평가
            rows.append({"aggressive": an, "regime": rg, "IS Sharpe": sr, "Sharpe": se.sharpe(r, rf),
                         "OOS Sharpe": se.sharpe(r.loc[OOS_START:], rf), "MDD": se.max_drawdown(r),
                         "연수익": se.ann_return(r), "공격 국면 비율": s.mean()})  # 저장
    for an, names in AGG_SETS.items():                        # 비교: 공격형 단독
        sr, r = is_sharpe(composite(d, names), d)             # 평가
        rows.append({"aggressive": an, "regime": "항상 공격", "IS Sharpe": sr, "Sharpe": se.sharpe(r, rf),
                     "OOS Sharpe": se.sharpe(r.loc[OOS_START:], rf), "MDD": se.max_drawdown(r), "연수익": se.ann_return(r), "공격 국면 비율": 1.0})  # 저장
    sr, r = is_sharpe(D, d)                                   # 방어형 단독
    rows.append({"aggressive": "-", "regime": "항상 방어", "IS Sharpe": sr, "Sharpe": se.sharpe(r, rf),
                 "OOS Sharpe": se.sharpe(r.loc[OOS_START:], rf), "MDD": se.max_drawdown(r), "연수익": se.ann_return(r), "공격 국면 비율": 0.0})  # 저장
    R = pd.DataFrame(rows).sort_values("IS Sharpe", ascending=False)  # 정렬
    R.to_csv(os.path.join(OUT, "s3_regime.csv"), index=False, encoding="utf-8-sig")  # 저장
    json.dump({"defense": defense}, open(os.path.join(OUT, "s3_defense.json"), "w"))  # 저장
    add_trials(len(AGG_SETS) * len(REGIMES), "s3_regime")     # 시행 기록
    print("방어형:", defense)                                  # 출력
    print(R.round(3).to_string(index=False))                  # 출력


# ---------------------------------------------------------------------------
# s4. Top-k 구성 고도화
# ---------------------------------------------------------------------------
def _score_from_spec(d, spec):
    """전략 사양 → 점수 (조합 'a+b' 또는 국면형 'REGIME:agg:regime')"""
    if spec.startswith("REGIME:"):                            # 국면형
        _, an, rg = spec.split(":")                           # 분해
        defense = json.load(open(os.path.join(OUT, "s3_defense.json")))["defense"]  # 방어형
        s = market_state(d).reindex(CFG.form_months())[rg].fillna(0.0)  # 국면
        return composite(d, AGG_SETS[an]).mul(s, axis=0) + composite(d, defense).mul(1 - s, axis=0)  # 점수
    return composite(d, spec.split("+"))                      # 조합


def stage4(n_top=5):
    """상위 전략(조합 IS 상위 + 국면형 IS 상위)에 k·상한·버퍼·중첩 보유 격자 적용, IS로 선택"""
    d = load_data()                                           # 데이터
    C = pd.read_csv(os.path.join(OUT, "s2_combos.csv"))       # 조합
    R = pd.read_csv(os.path.join(OUT, "s3_regime.csv"))       # 국면
    specs = C["combo"].head(n_top).tolist()                   # 조합 상위
    Rr = R[R["regime"].isin(REGIMES)].head(2)                 # 국면형 상위 2
    specs += [f"REGIME:{a}:{g}" for a, g in zip(Rr["aggressive"], Rr["regime"])]  # 추가
    grid = [(k, cap, buf, ov) for k in KS for cap in (0.0, 0.05, 0.10) for buf in (None, 1.5, 2.0) for ov in (1, 3)]  # 격자
    rows = []                                                 # 결과
    rf = d.ff["RF"]                                           # 무위험
    for sp in specs:                                          # 전략마다
        sc = _score_from_spec(d, sp)                          # 점수
        for k, cap, buf, ov in grid:                          # 구성마다
            pr = run_weights(topk_weights(sc, d, k, cap=cap, buffer=buf, overlap=ov), d)  # 수익률
            r = pr["net"]                                     # 순수익
            rows.append({"spec": sp, "k": k, "cap": cap, "buffer": buf or 0, "overlap": ov,
                         "IS Sharpe": se.sharpe(r.loc[:IS_END], rf), "Sharpe": se.sharpe(r, rf),
                         "OOS Sharpe": se.sharpe(r.loc[OOS_START:], rf), "MDD": se.max_drawdown(r),
                         "연수익": se.ann_return(r), "회전율(월)": pr["turnover"].mean()})  # 저장
        print("done", sp, flush=True)                         # 진행
    G = pd.DataFrame(rows).sort_values("IS Sharpe", ascending=False)  # 정렬
    G.to_csv(os.path.join(OUT, "s4_construction.csv"), index=False, encoding="utf-8-sig")  # 저장
    add_trials(len(specs) * len(grid), "s4_construction")     # 시행 기록
    print(G.head(25).round(3).to_string(index=False))         # 출력


# ---------------------------------------------------------------------------
# s5. 포트폴리오 단위 국면 혼합 (방어 Top-k 와 모멘텀 Top-k 를 비중으로 섞음)
# ---------------------------------------------------------------------------
BLEND_DEF = {"DEF_IS": ["low_beta_fp", "lowturn_winner", "high_52w_low"],  # s2 IS 1위 조합
             "DEF_LOWRISK": ["low_beta_fp", "low_max", "low_dvol"]}    # s2 IS 3위 · 순수 저위험 조합
BLEND_RULES = {                                               # 모멘텀 비중 규칙 (국면 = t월 말 정보)
    "고정 50:50": lambda ms: pd.Series(0.5, index=ms.index),  # 항상 반반
    "MA10 상승 50% / 하락 0%": lambda ms: 0.5 * ms["bull_ma10"],  # 상승 국면만 모멘텀 절반
    "MA10 상승 100% / 하락 0%": lambda ms: 1.0 * ms["bull_ma10"],  # 상승 국면 전부 모멘텀
    "합성점수 × 50%": lambda ms: 0.5 * ms["bull_score"],       # 4개 국면 지표 평균 비례
}


def stage5(k=30, buffer=1.5):
    """w = (1−a_t)·방어 Top-k + a_t·모멘텀 Top-k, a_t = 국면 규칙. 가중은 각 Top-k 내부 VW"""
    d = load_data()                                           # 데이터
    ms = market_state(d).reindex(CFG.form_months()).fillna(0.0)  # 국면
    rf = d.ff["RF"]                                           # 무위험
    wM = topk_weights(composite(d, ["mom_12_1"]), d, k, buffer=buffer)  # 모멘텀 Top-k
    rows, rets = [], {}                                       # 결과
    for dn, names in BLEND_DEF.items():                       # 방어형
        wD = topk_weights(composite(d, names), d, k, buffer=buffer)  # 방어 Top-k
        for rn, rule in BLEND_RULES.items():                  # 규칙
            a = rule(ms)                                      # 모멘텀 비중
            w = wD.mul(1 - a, axis=0) + wM.mul(a, axis=0)     # 혼합 비중
            pr = run_weights(w, d)                            # 수익률
            key = f"{dn} | {rn}"                              # 이름
            rows.append({"전략": key, **metrics(pr["net"], d, pr["turnover"]), "모멘텀 평균비중": a.mean()})  # 지표
            rets[key] = pr["net"]                             # 저장
        rets[f"{dn} | 단독"] = run_weights(wD, d)["net"]       # 방어 단독
    rets["MOM | 단독"] = run_weights(wM, d)["net"]             # 모멘텀 단독
    R = pd.DataFrame(rows).sort_values("IS Sharpe", ascending=False)  # 정렬
    R.to_csv(os.path.join(OUT, "s5_blend.csv"), index=False, encoding="utf-8-sig")  # 저장
    pd.DataFrame(rets).to_csv(os.path.join(OUT, "s5_blend_returns.csv"), encoding="utf-8-sig")  # 수익률
    add_trials(len(BLEND_DEF) * len(BLEND_RULES), "s5_blend")  # 시행 기록
    print("방어·모멘텀 상관:", pd.DataFrame(rets).corr().round(2).loc["MOM | 단독"].to_dict())  # 상관
    print(R[["전략", "연수익", "변동성", "Sharpe", "MDD", "IS Sharpe", "OOS Sharpe", "α t", "회전율(월)"]].round(3).to_string(index=False))  # 출력


# ===========================================================================
# v3 절차 (s6~s9): 분위 격차 조건을 선택 기준에 포함 + 산업 상한 + 국면 혼합 구성 최적화
#   [사전 규칙] 조합 '격차 조건 충족' = IS D1−D10 원수익 차이 t > 0 이고 IS 차이 FF3 α t > 2
#              최종 선택 = IS Sharpe 최고(0.03 이내 동률이면 IS MDD가 얕은 쪽). 전체·OOS는 보고용
# ===========================================================================
def stage6(n_sr=12, n_sp=8, n_check=40):
    """조합 탐색 + IS 상위 n_check개의 10분위 격차 점검"""
    d = load_data()                                           # 데이터
    T = pd.read_csv(os.path.join(OUT, "s1_topk.csv"))         # 단일 결과
    Q = pd.read_csv(os.path.join(OUT, "s1_quintile.csv"), index_col=0)  # 5분위
    t50 = T[T["k"] == FOCUS_K].set_index("name")              # Top-50
    cand_sr = t50["IS Sharpe"].sort_values(ascending=False).index[:n_sr].tolist()  # 위험조정 상위
    cand_sp = [n for n in Q["IS 차이 t"].sort_values(ascending=False).index if n not in cand_sr][:n_sp]  # 격차 상위
    cands = cand_sr + cand_sp                                 # 후보
    print("후보(Sharpe):", cand_sr, "\n후보(분위격차):", cand_sp, flush=True)  # 출력
    rf = d.ff["RF"]                                           # 무위험
    rows = []                                                 # 결과
    combos = [c for r in (2, 3) for c in itertools.combinations(cands, r)]  # 조합
    for i, c in enumerate(combos):                            # 조합마다
        sr, r = is_sharpe(composite(d, c), d)                 # IS Sharpe
        rows.append({"combo": "+".join(c), "n": len(c), "IS Sharpe": sr, "IS MDD": se.max_drawdown(r.loc[:IS_END]),
                     "Sharpe": se.sharpe(r, rf), "OOS Sharpe": se.sharpe(r.loc[OOS_START:], rf),
                     "MDD": se.max_drawdown(r), "연수익": se.ann_return(r)})  # 저장
        if (i + 1) % 200 == 0:                                # 진행
            print(f"  {i+1}/{len(combos)}", flush=True)       # 출력
    C = pd.DataFrame(rows).sort_values("IS Sharpe", ascending=False).reset_index(drop=True)  # 정렬
    chk = []                                                  # 격차 점검
    for c in C["combo"].head(n_check):                        # IS 상위
        _, sm, _ = group_analysis(composite(d, c.split("+")), d, 10)  # 10분위
        chk.append({"combo": c, "IS D 차이 t": sm["IS 차이 t"], "IS D 차이 α t": sm["IS 차이 α t"],
                    "D 차이 t": sm["차이 t"], "D 차이 α t": sm["차이 α t"], "D1 연수익": sm["top 연수익"],
                    "D10 연수익": sm["bottom 연수익"], "D1 Sharpe": sm["top Sharpe"], "D10 Sharpe": sm["bottom Sharpe"]})  # 저장
    K_ = pd.DataFrame(chk)                                    # 표
    C = C.merge(K_, on="combo", how="left")                   # 결합
    C["격차조건"] = (C["IS D 차이 t"] > 0) & (C["IS D 차이 α t"] > 2)  # 사전 규칙
    C = C.sort_values(["격차조건", "IS Sharpe"], ascending=[False, False])  # 충족 우선
    C.to_csv(os.path.join(OUT, "s6_combos.csv"), index=False, encoding="utf-8-sig")  # 저장
    json.dump({"cand_sr": cand_sr, "cand_sp": cand_sp}, open(os.path.join(OUT, "s6_cands.json"), "w"))  # 후보
    add_trials(len(combos), "s6_combos")                      # 시행 기록
    print(C.head(15).round(3).to_string(index=False))         # 출력


def _defense(rank=0):
    """s6 격차조건 충족 조합 중 IS 순위 rank번째"""
    C = pd.read_csv(os.path.join(OUT, "s6_combos.csv"))       # 조합
    return C[C["격차조건"]].iloc[rank]["combo"].split("+")    # 이름 목록


BLEND_RULES_V3 = {                                            # 모멘텀 비중 a_t 규칙
    "합성점수×50%": lambda ms: 0.5 * ms["bull_score"],         # 4개 국면 지표 평균 비례
    "MA10 상승 50%": lambda ms: 0.5 * ms["bull_ma10"],        # 10개월선 위면 절반
    "합성점수×30%": lambda ms: 0.3 * ms["bull_score"],         # 보수적
    "추세+저변동 모두 50%": lambda ms: 0.5 * ms["bull_trend12"] * ms["bull_lowvol"],  # 두 조건 동시
}


def build_spec(d, spec, k, buffer=None, ind_cap=None):
    """전략 사양 → 비중. 'a+b' 조합 / 'BLEND:<규칙>' (방어=s6 1위, 공격=모멘텀) / 'REGIME:<공격형>:<국면>' 점수 혼합"""
    kw = dict(buffer=buffer, ind_cap=ind_cap)                 # 구성 옵션
    if spec.startswith("BLEND:"):                             # 포트폴리오 혼합
        ms = market_state(d).reindex(CFG.form_months()).fillna(0.0)  # 국면
        a = BLEND_RULES_V3[spec.split(":", 1)[1]](ms)         # 모멘텀 비중
        wD = topk_weights(composite(d, _defense()), d, k, **kw)  # 방어
        wM = topk_weights(composite(d, ["mom_12_1"]), d, k, **kw)  # 모멘텀
        return wD.mul(1 - a, axis=0) + wM.mul(a, axis=0)      # 혼합
    if spec.startswith("REGIME:"):                            # 점수 혼합
        _, an, rg = spec.split(":")                           # 분해
        s = market_state(d).reindex(CFG.form_months())[rg].fillna(0.0)  # 국면
        sc = composite(d, AGG_SETS[an]).mul(s, axis=0) + composite(d, _defense()).mul(1 - s, axis=0)  # 점수
        return topk_weights(sc, d, k, **kw)                   # 비중
    return topk_weights(composite(d, spec.split("+")), d, k, **kw)  # 조합


def spec_score(d, spec):
    """분위 분석용 점수 (BLEND는 a_t·모멘텀 + (1−a_t)·방어 점수로 근사)"""
    ms = market_state(d).reindex(CFG.form_months()).fillna(0.0)  # 국면
    if spec.startswith("BLEND:"):                             # 혼합
        a = BLEND_RULES_V3[spec.split(":", 1)[1]](ms)         # 비중
        return composite(d, ["mom_12_1"]).mul(a, axis=0) + composite(d, _defense()).mul(1 - a, axis=0)  # 점수
    if spec.startswith("REGIME:"):                            # 점수 혼합
        _, an, rg = spec.split(":")                           # 분해
        s = ms[rg]                                            # 국면
        return composite(d, AGG_SETS[an]).mul(s, axis=0) + composite(d, _defense()).mul(1 - s, axis=0)  # 점수
    return composite(d, spec.split("+"))                      # 조합


def stage7():
    """국면 전환형: 점수 혼합(REGIME) 18개 + 포트폴리오 혼합(BLEND) 4개, Top-50 기본 구성으로 IS 비교"""
    d = load_data()                                           # 데이터
    rf = d.ff["RF"]                                           # 무위험
    specs = [f"REGIME:{a}:{g}" for a in AGG_SETS for g in REGIMES] + [f"BLEND:{b}" for b in BLEND_RULES_V3] + ["+".join(_defense())]  # 사양
    rows = []                                                 # 결과
    for sp in specs:                                          # 사양마다
        r = run_weights(build_spec(d, sp, FOCUS_K), d)["net"]  # 수익률
        rows.append({"spec": sp, "IS Sharpe": se.sharpe(r.loc[:IS_END], rf), "IS MDD": se.max_drawdown(r.loc[:IS_END]),
                     "Sharpe": se.sharpe(r, rf), "OOS Sharpe": se.sharpe(r.loc[OOS_START:], rf),
                     "MDD": se.max_drawdown(r), "연수익": se.ann_return(r)})  # 저장
    R = pd.DataFrame(rows).sort_values("IS Sharpe", ascending=False)  # 정렬
    R.to_csv(os.path.join(OUT, "s7_regime.csv"), index=False, encoding="utf-8-sig")  # 저장
    add_trials(len(specs) - 1, "s7_regime")                   # 시행 기록
    print("방어형:", _defense())                               # 출력
    print(R.round(3).to_string(index=False))                  # 출력


def stage8(n_combo=4):
    """구성 격자: (격차조건 충족 상위 조합 + IS 상위 국면형 2개 + BLEND 4개) × k × 버퍼 × 산업 상한"""
    d = load_data()                                           # 데이터
    rf = d.ff["RF"]                                           # 무위험
    C = pd.read_csv(os.path.join(OUT, "s6_combos.csv"))       # 조합
    R = pd.read_csv(os.path.join(OUT, "s7_regime.csv"))       # 국면
    specs = C[C["격차조건"]]["combo"].head(n_combo).tolist()   # 조합
    specs += R[R["spec"].str.startswith("REGIME")]["spec"].head(2).tolist()  # 점수 혼합 상위
    specs += [f"BLEND:{b}" for b in BLEND_RULES_V3]           # 포트폴리오 혼합
    grid = [(k, b, ic) for k in KS for b in (None, 1.5, 2.0) for ic in (None, 0.3)]  # 격자
    rows = []                                                 # 결과
    for sp in specs:                                          # 사양마다
        for k, b, ic in grid:                                 # 구성마다
            pr = run_weights(build_spec(d, sp, k, b, ic), d)  # 수익률
            r = pr["net"]                                     # 순수익
            rows.append({"spec": sp, "k": k, "buffer": b or 0, "ind_cap": ic or 0,
                         "IS Sharpe": se.sharpe(r.loc[:IS_END], rf), "IS MDD": se.max_drawdown(r.loc[:IS_END]),
                         "Sharpe": se.sharpe(r, rf), "OOS Sharpe": se.sharpe(r.loc[OOS_START:], rf),
                         "MDD": se.max_drawdown(r), "OOS MDD": se.max_drawdown(r.loc[OOS_START:]),
                         "연수익": se.ann_return(r), "회전율(월)": pr["turnover"].mean()})  # 저장
        print("done", sp, flush=True)                         # 진행
    G = pd.DataFrame(rows).sort_values("IS Sharpe", ascending=False)  # 정렬
    G.to_csv(os.path.join(OUT, "s8_construction.csv"), index=False, encoding="utf-8-sig")  # 저장
    add_trials(len(specs) * len(grid), "s8_construction")     # 시행 기록
    print(G.head(25).round(3).to_string(index=False))         # 출력


def pick_final(G, tie=0.03):
    """사전 규칙: IS Sharpe 최고, 0.03 이내 동률이면 IS MDD가 얕은 쪽"""
    top = G["IS Sharpe"].max()                                # 최고
    near = G[G["IS Sharpe"] >= top - tie]                     # 동률권
    return near.sort_values("IS MDD", ascending=False).iloc[0]  # MDD가 얕은(큰) 쪽


def stage9():
    """최종 선택 + 보유 종목 상장폐지 점검 (분류별 건수·기여, 불명확·일반 상폐 목록)"""
    d = load_data()                                           # 데이터
    G = pd.read_csv(os.path.join(OUT, "s8_construction.csv"))  # 격자
    f = pick_final(G)                                         # 최종
    fin = {"spec": f["spec"], "k": int(f["k"]), "buffer": float(f["buffer"]) or None, "ind_cap": float(f["ind_cap"]) or None,
           "defense": _defense()}                             # 사양
    json.dump(fin, open(os.path.join(OUT, "s9_final.json"), "w"), ensure_ascii=False)  # 저장
    print("최종:", fin)                                        # 출력
    w = build_spec(d, fin["spec"], fin["k"], fin["buffer"], fin["ind_cap"])  # 비중
    ev = d.x["delist_events"].set_index(["m", "PERMNO"])      # 이벤트
    rows = []                                                 # 상폐 보유
    held = (w > 0).values                                     # 보유
    dr = d.delist_ret.shift(-1).reindex(index=w.index, columns=w.columns).values  # 다음달 상폐
    for i, j in zip(*np.where(held & ~np.isnan(dr))):         # 보유 중 상폐
        t, p = w.index[i] + 1, w.columns[j]                   # 월, 종목
        e = ev.loc[(t, p)]                                    # 이벤트
        rows.append({"월": str(t), "PERMNO": p, "회사": d.info["COMNAM"].get(p), "분류": e["class"],
                     "비중": w.iloc[i, j], "직전 주가": abs(e["MthPrc"]), "12개월 로그수익": e["r12"],
                     "변동성 축소비": e["volratio"], "CIZ 상폐월 수익": e["dret"], "적용 수익": e["applied_ret"],
                     "포트폴리오 기여": w.iloc[i, j] * e["applied_ret"],
                     "−50% 대신 CIZ 적용 시 차이": w.iloc[i, j] * (e["dret"] - e["applied_ret"])})  # 저장
    gone = held & np.isnan(dr) & se._after_last(w.index, d.last_month.reindex(w.columns)).values  # 흔적 없이 퇴출
    fwd = d.ret.shift(-1).reindex(index=w.index, columns=w.columns)  # 다음달 수익
    gone &= fwd.isna().values                                 # 수익 없음
    if d.fill_ret is not None:                                # CIZ 보완
        gone &= d.fill_ret.shift(-1).reindex(index=w.index, columns=w.columns).isna().values  # 없음
    for i, j in zip(*np.where(gone)):                         # 퇴출
        p = w.columns[j]                                      # 종목
        rows.append({"월": str(w.index[i] + 1), "PERMNO": p, "회사": d.info["COMNAM"].get(p), "분류": "흔적없음(−50%)",
                     "비중": w.iloc[i, j], "적용 수익": -0.5, "포트폴리오 기여": -0.5 * w.iloc[i, j]})  # 저장
    X = pd.DataFrame(rows)                                    # 표
    X.to_csv(os.path.join(OUT, "s9_delist_holdings.csv"), index=False, encoding="utf-8-sig")  # 저장
    if len(X):                                                # 요약
        print(X.groupby("분류").agg(건수=("PERMNO", "size"), 기여합=("포트폴리오 기여", "sum")).round(4))  # 출력
        print(X[X["분류"].isin(["perf", "unclear", "흔적없음(−50%)"])].round(3).to_string(index=False))  # 점검 대상


# ===========================================================================
# s10. 기술적 모멘텀·추세·돌파 신호 (signals_tech.py) + 절대 모멘텀 필터·현금 대피
# ===========================================================================
B_SPEC = {"names": ["low_beta_fp", "low_vol_12m", "coskew"], "k": 30, "buffer": 1.5, "ind_cap": 0.3}  # 14장 B안


def _with_cash(w, d):
    """비중 합이 1 미만이면 나머지는 현금(무위험수익) → 순수익률·회전율"""
    pr = run_weights(w, d)                                    # 주식 부분
    cash = (1 - w.sum(1)).clip(lower=0)                       # 현금 비중(형성월)
    cash.index = cash.index + 1                               # 보유월로
    pr["net"] = pr["net"] + cash * d.ff["RF"].reindex(pr.index)  # 현금 수익 가산
    return pr                                                 # 반환


def stage10():
    """기술 신호 단일 검증 + 필터·듀얼 모멘텀 변형 + B안 오버레이"""
    from signals_tech import INFO_TECH, above_sma, abs_mom    # 기술 신호
    d = load_data()                                           # 데이터
    rf = d.ff["RF"]                                           # 무위험
    tk_rows, dec_rows, dec_tabs = [], [], {}                  # 결과
    for n, (f, ds, cat) in INFO_TECH.items():                 # 단일 신호
        sc = f(d)                                             # 점수
        tk, _ = eval_topk(sc, d)                              # Top-k
        for k, row in tk.iterrows():                          # k별
            tk_rows.append({"name": n, "설명": ds, "범주": cat, "k": k, **row})  # 저장
        if n != "tech_composite_f":                           # 필터형은 대상이 적어 분위 생략
            t10, s10, _ = group_analysis(sc, d, 10)           # 10분위
            t5, s5, _ = group_analysis(sc, d, 5)              # 5분위
            dec_rows.append({"name": n, "설명": ds, **{f"D {a}": b for a, b in s10.items()}, **{f"Q {a}": b for a, b in s5.items()}})  # 저장
            dec_tabs[n] = t10                                 # 분위표
        print(f"{n:18s} bestSR {tk['Sharpe'].max():.2f}", flush=True)  # 진행
    pd.DataFrame(tk_rows).to_csv(os.path.join(OUT, "s10_tech_topk.csv"), index=False, encoding="utf-8-sig")  # 저장
    pd.DataFrame(dec_rows).to_csv(os.path.join(OUT, "s10_tech_groups.csv"), index=False, encoding="utf-8-sig")  # 저장
    pickle.dump(dec_tabs, open(os.path.join(OUT, "s10_tech_decile_tables.pkl"), "wb"))  # 분위표
    # --- 필터·듀얼 모멘텀 변형 ---
    sma10, am = above_sma(d, 10), abs_mom(d)                  # 종목 추세·절대 모멘텀 필터
    mkt12 = np.expm1(np.log1p(d.ff["MKT"]).rolling(12, 12).sum())  # 시장 12개월
    rf12 = np.expm1(np.log1p(d.ff["RF"]).rolling(12, 12).sum())  # 무위험 12개월
    mkt_on = (mkt12 > rf12).reindex(CFG.form_months()).fillna(True).astype(float)  # 시장 절대 모멘텀(형성월)
    bases = {"mom_12_1": composite(d, ["mom_12_1"]), "roc_3_6": INFO_TECH["roc_3_6"][0](d), "clenow": INFO_TECH["clenow"][0](d)}  # 기반
    rows = []                                                 # 결과

    def record(name, w, cash=False):                          # 한 변형 기록
        pr = _with_cash(w, d) if cash else run_weights(w, d)  # 수익률
        m = metrics(pr["net"], d, pr["turnover"])             # 지표
        inv = w.sum(1).mean()                                 # 평균 주식 비중
        rows.append({"전략": name, **m, "평균 주식비중": inv})  # 저장

    for bn, sc in bases.items():                              # 기반마다
        for k in (30, 50):                                    # k
            w0 = topk_weights(sc, d, k)                       # 기본
            record(f"{bn} | Top-{k}", w0)                     # 기본
            record(f"{bn} + 10개월선 필터 | Top-{k}", topk_weights(sc.where(sma10), d, k))  # 추세 필터
            record(f"{bn} + 절대모멘텀 필터 | Top-{k}", topk_weights(sc.where(am), d, k))  # 듀얼(현금 없음)
            wf = topk_weights(sc.where(am & sma10), d, k)     # 두 필터 통과 종목
            n_ok = (wf > 0).sum(1)                            # 통과 수
            record(f"{bn} + 듀얼·현금대피(종목) | Top-{k}", wf.mul(n_ok.clip(upper=k) / k, axis=0), cash=True)  # 빈 자리는 현금
            record(f"{bn} + 듀얼·현금대피(시장) | Top-{k}", w0.mul(mkt_on, axis=0), cash=True)  # 시장 하락 추세면 전액 현금
    # --- B안 오버레이 ---
    bsc = composite(d, B_SPEC["names"])                       # B 점수
    kw = dict(buffer=B_SPEC["buffer"], ind_cap=B_SPEC["ind_cap"])  # B 구성
    K = B_SPEC["k"]                                           # k
    wB = topk_weights(bsc, d, K, **kw)                        # B 기본
    record("B안 (14장)", wB)                                  # 기준
    record("B안 + 10개월선 필터", topk_weights(bsc.where(sma10), d, K, **kw))  # 추세 필터
    tc = se.prepare_score(INFO_TECH["tech_composite"][0](d), d, CFG).rank(axis=1, pct=True)  # 기술 종합 백분위
    record("B안 + 기술종합점수(4번째 구성요소)", topk_weights((bsc * 3 + tc) / 4, d, K, **kw))  # 합성
    record("B안 + 듀얼·현금대피(시장)", wB.mul(mkt_on, axis=0), cash=True)  # 시장 필터
    record("B안 + 절반 현금대피(시장 하락 시 50%)", wB.mul(0.5 + 0.5 * mkt_on, axis=0), cash=True)  # 절반 대피
    R = pd.DataFrame(rows)                                    # 표
    R.to_csv(os.path.join(OUT, "s10_filters.csv"), index=False, encoding="utf-8-sig")  # 저장
    add_trials(len(INFO_TECH) * len(KS) + len(R), "s10_tech")  # 시행 기록
    print(R[["전략", "연수익", "변동성", "Sharpe", "MDD", "IS Sharpe", "OOS Sharpe", "α t", "평균 주식비중"]].round(3).to_string(index=False))  # 출력


if __name__ == "__main__":                                    # 실행
    stages = sys.argv[1:] or ["s0", "s1", "s2", "s3", "s4"]   # 단계
    for st in stages:                                         # 순서대로
        t0 = time.time()                                      # 시간
        print(f"\n===== {st} =====", flush=True)              # 머리말
        globals()[f"stage{st[1:]}"]()                         # 실행
        print(f"[{st}] {time.time()-t0:.0f}s", flush=True)    # 소요
