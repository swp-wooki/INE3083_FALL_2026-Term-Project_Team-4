"""
확장 신호(Signal) 정의 v2 — 보유 데이터만으로 만든 새 팩터 후보
==================================================================
[규칙] signals.py와 동일: f(d) -> DataFrame(행=월, 열=PERMNO), 클수록 좋은 종목, t월 말 정보만 사용

[범주]
  A. FF5 롤링 적재량 — 재무제표 없이 가치·수익성·투자 성향을 수익률 공변동으로 근사
  B. 일간 집계 기반 — 변동성·유동성의 '변화'와 '안정성', 복권성
  C. 거래량 × 가격 상호작용
  D. 가격·모멘텀 변형
  E. 이벤트·배당
  T. 기술적 지표 — 월간 가격·거래량으로 계산한 RSI, MACD, 볼린저밴드, 스토캐스틱,
     거래량 프로파일(VPVR 근사: VWAP·POC·상단 매물대), 미실현 손익(CGO, Grinblatt–Han 2005)
     ※ 일간 가격 원자료가 없어 전통적 일간 지표 대신 월간 버전을 사용 (한계로 명시)
  R. 시장 국면 지표(market_state) — 시장 추세·변동성·폭(breadth)·심리(거래 과열, 신규상장)
"""
import numpy as np                           # 수치 계산
import pandas as pd                          # 데이터프레임
from signals import cum_return, rmean, roll_beta, by_industry, turnover, mom_12_1, mom_6_1, mom_12_7  # 공통 유틸 재사용


def _pct(x: pd.DataFrame) -> pd.DataFrame:
    """월별 횡단면 백분위(0~1)"""
    return x.rank(axis=1, pct=True)                           # 백분위


# ---------------------------------------------------------------------------
# A. FF5 롤링 회귀 (36개월, 창 안에 결측 없는 종목만)
# ---------------------------------------------------------------------------
_FF5_CACHE = {}                                               # 결과 캐시


def _ff5_rolling(d, window: int = 36) -> dict:
    """종목별 롤링 FF5 회귀 → 알파, 고유변동성, R², 각 요인 베타, 당월 잔차"""
    key = (id(d), window)                                     # 캐시 키
    if key in _FF5_CACHE:                                     # 이미 계산했으면
        return _FF5_CACHE[key]                                # 재사용
    fac = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]              # 요인
    ff = d.ff.reindex(d.ret.index)                            # 기간 맞춤
    Y = d.ret.sub(ff["RF"], axis=0).values                    # 초과수익
    Xall = np.column_stack([np.ones(len(ff)), ff[fac].values])  # 상수 + 5요인
    names = ["alpha", "ivol", "r2", "resid"] + fac            # 결과 이름
    out = {n: np.full(Y.shape, np.nan) for n in names}        # 결과 배열
    for t in range(window - 1, len(Y)):                       # 창이 채워진 시점부터
        X = Xall[t - window + 1:t + 1]                        # 창 내 요인
        y = Y[t - window + 1:t + 1]                           # 창 내 수익률
        ok = ~np.isnan(y).any(0)                              # 결측 없는 종목
        if ok.sum() == 0 or np.isnan(X).any():                # 대상 없으면
            continue                                          # 건너뜀
        B, *_ = np.linalg.lstsq(X, y[:, ok], rcond=None)      # 전 종목 동시 OLS
        e = y[:, ok] - X @ B                                  # 잔차
        yc = y[:, ok] - y[:, ok].mean(0)                      # 중심화 수익률
        out["alpha"][t, ok] = B[0]                            # 월 알파
        out["ivol"][t, ok] = e.std(0, ddof=6)                 # 잔차 표준편차
        out["r2"][t, ok] = 1 - (e ** 2).sum(0) / (yc ** 2).sum(0)  # 결정계수
        out["resid"][t, ok] = e[-1]                           # 당월 잔차
        for j, f in enumerate(fac):                           # 요인 베타
            out[f][t, ok] = B[j + 1]                          # 저장
    res = {n: pd.DataFrame(a, index=d.ret.index, columns=d.ret.columns) for n, a in out.items()}  # 데이터프레임화
    _FF5_CACHE[key] = res                                     # 캐시
    return res                                                # 반환


def rmw_beta(d): return _ff5_rolling(d)["RMW"]                # 수익성 요인 적재량 ↑ (퀄리티 대리)
def hml_beta(d): return _ff5_rolling(d)["HML"]                # 가치 요인 적재량 ↑ (B/M 대리)
def cma_beta(d): return _ff5_rolling(d)["CMA"]                # 보수적 투자 요인 적재량 ↑
def ff5_alpha_36(d): return _ff5_rolling(d)["alpha"]          # FF5 알파(36개월)
def low_ivol_ff5_36(d): return -_ff5_rolling(d)["ivol"]       # FF5 고유변동성 ↓
def low_r2_ff5(d): return -_ff5_rolling(d)["r2"]              # 가격 동조성 ↓ (Morck–Yeung–Yu 2000)
def resid_reversal(d): return -_ff5_rolling(d)["resid"]       # 당월 FF5 잔차 반전


def _ex_mkt(d):
    """종목 초과수익과 시장 초과수익"""
    rf = d.ff["RF"].reindex(d.ret.index)                      # 무위험
    return d.ret.sub(rf, axis=0), d.ff["Mkt-RF"].reindex(d.ret.index)  # (종목, 시장)


def low_downside_beta(d):
    """하락장 베타(시장 초과수익<0인 달만, 60개월·최소 12개 하락월): 낮을수록 좋음 (Ang–Chen–Xing 2006의 방어적 해석)"""
    ex, m = _ex_mkt(d)                                        # 초과수익
    down = (m < 0).values[:, None]                            # 하락월
    y = ex.where(np.broadcast_to(down, ex.shape))             # 하락월 수익만
    return -roll_beta(y, m, 60, 12)                           # 부호 반전


def coskew(d):
    """공왜도(60개월, 최소 36): E[(r−r̄)(m−m̄)²]/(σ_r σ_m²). 음의 공왜도일수록 기대수익↑ → −coskew (Harvey–Siddique 2000)"""
    ex, m = _ex_mkt(d)                                        # 초과수익
    mask = ex.notna()                                         # 관측 달
    M = pd.DataFrame(np.where(mask, m.values[:, None], np.nan), index=ex.index, columns=ex.columns)  # 종목별 시장
    n, mp = 60, 36                                            # 창·최소
    E = lambda z: z.rolling(n, mp).mean()                     # 롤링 평균
    ey, em, em2 = E(ex), E(M), E(M * M)                       # 1·2차 적률
    num = E(ex * M * M) - 2 * em * E(ex * M) - ey * em2 + 2 * ey * em ** 2  # 공왜도 분자
    vy = E(ex * ex) - ey ** 2                                 # 종목 분산
    vm = em2 - em ** 2                                        # 시장 분산
    return -(num / (np.sqrt(vy) * vm))                        # 부호 반전


# ---------------------------------------------------------------------------
# B. 일간 집계 기반
# ---------------------------------------------------------------------------
def low_volofvol(d):
    """변동성의 변동성: 월 실현변동성 12개월 변동계수 ↓ (Baltussen–van Bekkum–van der Grient 2018)"""
    rv = d.x["rvol_d"]                                        # 월 실현변동성
    return -(rv.rolling(12, 9).std() / rv.rolling(12, 9).mean())  # 부호 반전


def low_vol_shock(d):
    """변동성 충격: 이번 달 변동성 ÷ 직전 12개월 평균 (급등할수록 나쁨)"""
    rv = d.x["rvol_d"]                                        # 실현변동성
    return -np.log(rv / rmean(rv, 12, 9).shift(1))            # 로그비율 반전


def liq_shock(d):
    """유동성 충격: −log(Amihud 이번 달 ÷ 직전 12개월 평균). 유동성이 나빠진 종목 회피 (Bali et al. 2014)"""
    il = d.x["illiq_d"].where(d.x["illiq_d"] > 0)             # 양수만
    return -np.log(il / rmean(il, 12, 9).shift(1))            # 부호 반전


def low_idio_share(d):
    """고유위험 비중 ivol²/rvol² (3개월 평균) ↓"""
    return -rmean((d.x["ivol_d"] / d.x["rvol_d"]) ** 2, 3)    # 부호 반전


def low_range(d):
    """월중 범위(최대 일수익 − 최저 일수익) 3개월 평균 ↓"""
    return -rmean(d.x["max_d"] - d.x["min_d"], 3)             # 부호 반전


def low_max_scaled(d):
    """변동성으로 나눈 MAX ↓ — 변동성 효과를 뺀 순수 복권성 (Asness et al. 2020)"""
    return -(d.x["max_d"] / d.x["rvol_d"])                    # 부호 반전


def low_lottery(d):
    """복권성 지수(고MAX·고왜도·저가) ↓ (Kumar 2009)"""
    lot = (_pct(d.x["max_d"]) + _pct(d.x["skew_d"]) + _pct(-np.log(d.x["prc"].abs()))) / 3  # 복권성 백분위 평균
    return -lot                                               # 부호 반전


def low_dvol_cv(d):
    """거래대금 12개월 변동계수 ↓ (유동성 안정성)"""
    dv = d.x["dvol_m"]                                        # 월 거래대금
    return -(dv.rolling(12, 9).std() / dv.rolling(12, 9).mean())  # 부호 반전


def high_beta_d12(d):
    """고베타(일간 베타 12개월 평균) — 공격형 구성요소"""
    return rmean(d.x["beta_d"].clip(-3, 5), 12, 9)            # 절단 후 평균


# ---------------------------------------------------------------------------
# C. 거래량 × 가격
# ---------------------------------------------------------------------------
def lowturn_winner(d):
    """저회전율 승자: 모멘텀 백분위 + 저회전율 백분위 (Lee–Swaminathan 2000)"""
    return (_pct(mom_12_1(d)) + _pct(-rmean(turnover(d), 12, 9))) / 2  # 평균


def suv(d):
    """표준화 이상거래량: 월별 횡단면에서 log(회전율/직전 12개월 평균)을 |수익률|·부호로 회귀한 잔차 (Garfinkel–Sokobin 2006)"""
    tv = turnover(d)                                          # 회전율
    ab = np.log(tv / rmean(tv, 12, 9).shift(1)).replace([np.inf, -np.inf], np.nan)  # 이상거래량
    out = pd.DataFrame(np.nan, index=ab.index, columns=ab.columns)  # 결과
    for t in ab.index:                                        # 월마다
        y, r = ab.loc[t], d.ret.loc[t]                        # 이상거래량, 수익률
        ok = y.notna() & r.notna()                            # 둘 다 있는 종목
        if ok.sum() < 50:                                     # 표본 부족
            continue                                          # 건너뜀
        X = np.column_stack([np.ones(ok.sum()), r[ok].abs(), (r[ok] > 0).astype(float)])  # 상수·|r|·부호
        b, *_ = np.linalg.lstsq(X, y[ok].values, rcond=None)  # OLS
        out.loc[t, ok[ok].index] = y[ok].values - X @ b       # 잔차
    return out                                                # 반환


def dvol_trend(d):
    """거래대금 추세: log(최근 3개월 평균 ÷ 12개월 평균)"""
    dv = d.x["dvol_m"]                                        # 거래대금
    return np.log(rmean(dv, 3) / rmean(dv, 12, 9))            # 로그비율


def quiet_loser(d):
    """조용한 하락 반전: 직전월 하락 백분위 + 이상거래량 낮음 백분위 (거래량 동반 없는 하락은 되돌림)"""
    tv = turnover(d)                                          # 회전율
    abn = tv / rmean(tv, 12, 9).shift(1)                      # 이상거래량
    return (_pct(-d.ret) + _pct(-abn)) / 2                    # 평균


# ---------------------------------------------------------------------------
# D. 가격·모멘텀 변형
# ---------------------------------------------------------------------------
def mom_sharpe(d):
    """위험조정 모멘텀: 12-1 누적수익 ÷ 같은 구간 월수익 표준편차"""
    sd = d.ret.rolling(11, 11).std().shift(1)                 # t−11~t−1 변동성
    return mom_12_1(d) / sd                                   # 비율


def _info_discreteness(d):
    """정보 이산성 ID = sign(PRET) × (%음수달 − %양수달), t−11~t−1 (Da–Gurun–Warachka 2014, 월간 근사)"""
    r = d.ret.shift(1)                                        # 최근월 제외
    pos = (r > 0).astype(float).where(r.notna()).rolling(11, 11).mean()  # 양수 비율
    neg = (r < 0).astype(float).where(r.notna()).rolling(11, 11).mean()  # 음수 비율
    return np.sign(mom_12_1(d)) * (neg - pos)                 # ID


def frog_in_pan(d): return -_info_discreteness(d)             # 연속적 정보(ID 낮음)일수록 좋음


def mom_fip(d):
    """FIP 모멘텀: 모멘텀 백분위 + 연속 정보 백분위"""
    return (_pct(mom_12_1(d)) + _pct(frog_in_pan(d))) / 2     # 평균


def mom_accel(d): return mom_6_1(d) - mom_12_7(d)             # 모멘텀 가속


def high_52w_low(d):
    """52주 최저가 대비 상승률: 가격 ÷ 12개월 최저 월말가"""
    return d.price / d.price.rolling(12, 12).min()            # 비율


def low_mdd_12m(d):
    """최근 12개월 경로상 최대낙폭이 작을수록 좋음"""
    P = d.price.values.astype(np.float32)                     # 가격 배열
    T, N = P.shape                                            # 크기
    out = np.full((T, N), np.nan, dtype=np.float32)           # 결과
    for t in range(11, T):                                    # 창 끝 시점
        w = P[t - 11:t + 1]                                   # 12개월 창
        run = np.fmax.accumulate(w, axis=0)                   # 누적 고점
        dd = np.nanmin(w / run - 1, axis=0)                   # 최대 낙폭
        dd[np.isnan(w).any(0)] = np.nan                       # 결측 있으면 제외
        out[t] = dd                                           # 저장
    return pd.DataFrame(out, index=d.price.index, columns=d.price.columns).astype(float)  # 낙폭이 0에 가까울수록 큼


def ind_lead_lag(d):
    """산업 선도-지연: 같은 SIC2 산업 대형주(산업 내 시총 상위 30%)의 당월 시총가중 수익 (Hou 2007)"""
    sic2 = np.floor(d.x["siccd"] / 100)                       # 산업
    mc = d.x["mcap"]                                          # 시총
    out = pd.DataFrame(np.nan, index=d.ret.index, columns=d.ret.columns)  # 결과
    for t in d.ret.index:                                     # 월마다
        df = pd.DataFrame({"g": sic2.loc[t], "m": mc.loc[t], "r": d.ret.loc[t]}).dropna()  # 표본
        if df.empty:                                          # 없으면
            continue                                          # 건너뜀
        big = df["m"] >= df.groupby("g")["m"].transform(lambda s: s.quantile(0.7))  # 산업 내 대형주
        b = df[big]                                           # 대형주
        vw = (b["r"] * b["m"]).groupby(b["g"]).sum() / b["m"].groupby(b["g"]).sum()  # 산업별 대형주 VW 수익
        g_all = sic2.loc[t].dropna()                          # 모든 종목의 산업
        out.loc[t, g_all.index] = g_all.map(vw).values        # 배치
    return out                                                # 반환


def intra_ind_mom(d):
    """산업 내 상대 모멘텀: 12-1 모멘텀 − 산업 평균"""
    m = mom_12_1(d)                                           # 모멘텀
    return m - by_industry(m, d.x["siccd"])                   # 산업 조정


# ---------------------------------------------------------------------------
# E. 이벤트·배당
# ---------------------------------------------------------------------------
def _div_yield_m(d):
    """월 배당수익률 = 총수익 − 가격수익"""
    return (d.x["ret_tot"] - d.x["retx"]).clip(lower=0)       # 음수 잡음 제거


def div_month(d):
    """배당 예정월 효과: t+1월이 과거 배당 주기(12개월·3개월 전)와 겹치면 가점 (Hartzmark–Solomon 2013). 동점은 배당수익률로"""
    y = _div_yield_m(d)                                       # 월 배당수익률
    pred = (y.shift(11) > 0).astype(float) + (y.shift(2) > 0).astype(float)  # t+1과 같은 주기의 과거 배당 여부
    return pred + y.shift(11).fillna(0) + y.shift(2).fillna(0)  # 여부 + 배당수익률(동점 해소)


def div_growth(d):
    """배당 성장: 최근 12개월 주당배당(수정가 기준) ÷ 그 이전 12개월 − 1 (배당 없던 종목은 0)"""
    dps = _div_yield_m(d) * d.price.shift(1)                  # 월 주당배당(수정 기준 근사)
    s12 = dps.rolling(12, 12).sum()                           # 최근 12개월
    prev = s12.shift(12)                                      # 이전 12개월
    g = (s12 / prev - 1).where(prev > 0)                      # 성장률
    return g.clip(-1, 3).fillna(0).where(s12.notna())         # 절단·결측 처리


def split_drift(d):
    """주식분할 후 표류: 최근 12개월 내 분할(원주가/수정주가 비율이 25% 이상 하락)일수록, 최근일수록 높음 (Ikenberry et al. 1996)"""
    f = d.x["prc"].abs() / d.price                            # 원주가/수정주가 비율
    sp = (f / f.shift(1) < 0.75).astype(float)                # 분할 월
    score = pd.DataFrame(0.0, index=f.index, columns=f.columns)  # 기본 0
    for k in range(12):                                       # 최근 12개월
        score = np.maximum(score, sp.shift(k).fillna(0) * (1 - k / 12))  # 최근 분할일수록 큼
    return score.where(f.notna())                             # 가격 없는 종목 제외


def exch_upgrade(d):
    """거래소 상향 이전(NASDAQ·AMEX → NYSE) 후 12개월, 최근일수록 높음"""
    up = ((d.exch == 1) & d.exch.shift(1).isin([2, 3])).astype(float)  # 상향 이전 월
    score = pd.DataFrame(0.0, index=up.index, columns=up.columns)  # 기본 0
    for k in range(12):                                       # 최근 12개월
        score = np.maximum(score, up.shift(k).fillna(0) * (1 - k / 12))  # 최근일수록 큼
    return score.where(d.price.notna())                       # 가격 있는 종목만


# ---------------------------------------------------------------------------
# T. 기술적 지표 (월간)
# ---------------------------------------------------------------------------
def _rsi(d, n):
    """RSI(n개월) = 상승폭 합 ÷ (상승폭 합 + 하락폭 합)"""
    up = d.ret.clip(lower=0).rolling(n, n).sum()              # 상승폭
    dn = (-d.ret).clip(lower=0).rolling(n, n).sum()           # 하락폭
    return up / (up + dn)                                     # 0~1


def rsi_12(d): return _rsi(d, 12)                             # RSI 12개월 (높을수록 추세)
def rsi_3_oversold(d): return -_rsi(d, 3)                     # RSI 3개월 과매도(낮을수록 반등 기대)


def _macd(d):
    """MACD(12,26,9개월) → (정규화 MACD선, 히스토그램)"""
    P = d.price                                               # 수정주가
    e12 = P.ewm(span=12, adjust=False, min_periods=12).mean()  # 12개월 EMA
    e26 = P.ewm(span=26, adjust=False, min_periods=26).mean()  # 26개월 EMA
    line = (e12 - e26) / P                                    # 가격으로 정규화한 MACD선
    sig = line.ewm(span=9, adjust=False, min_periods=9).mean()  # 시그널선
    return line, line - sig                                   # 선, 히스토그램


def macd_line(d): return _macd(d)[0]                          # MACD선 ↑ (중기 추세)
def macd_hist(d): return _macd(d)[1]                          # MACD 히스토그램 ↑ (추세 가속)


def boll_pctb(d):
    """볼린저 %B(20개월, ±2σ): 밴드 내 위치 ↑ (추세 돌파 해석)"""
    ma, sd = d.price.rolling(20, 20).mean(), d.price.rolling(20, 20).std()  # 중심선, 표준편차
    return (d.price - (ma - 2 * sd)) / (4 * sd)               # %B


def low_boll_width(d):
    """볼린저 밴드폭 ↓ (4σ/중심선, 수렴 = 안정)"""
    ma, sd = d.price.rolling(20, 20).mean(), d.price.rolling(20, 20).std()  # 중심선, 표준편차
    return -(4 * sd / ma)                                     # 부호 반전


def stoch_12(d):
    """스토캐스틱 %K(12개월 월말가 범위 내 위치)"""
    lo, hi = d.price.rolling(12, 12).min(), d.price.rolling(12, 12).max()  # 최저·최고
    return (d.price - lo) / (hi - lo)                         # 0~1


def _profile(d, n=12):
    """거래량 프로파일(VPVR 근사): 월 회전율을 가중치로 한 과거 n개월 가격 분포"""
    tv = turnover(d).clip(upper=1)                            # 회전율(분할 영향 없음)
    P = d.price                                               # 수정주가
    lagsP = [P.shift(k) for k in range(n)]                    # 과거 가격
    lagsW = [tv.shift(k) for k in range(n)]                   # 과거 거래 가중치
    return lagsP, lagsW                                       # 반환


def vwap_dev(d):
    """12개월 거래량가중평균가(VWAP) 대비 현재가 괴리 ↑ (매물대 위 = 지지)"""
    Ps, Ws = _profile(d)                                      # 프로파일
    num = sum(p * w for p, w in zip(Ps, Ws))                  # Σ P·w
    den = sum(Ws)                                             # Σ w
    return d.price / (num / den) - 1                          # 괴리율


def poc_dev(d):
    """POC(12개월 중 거래가 가장 많았던 달의 가격) 대비 현재가 괴리 ↑"""
    Ps, Ws = _profile(d)                                      # 프로파일
    W = np.stack([w.values for w in Ws])                      # (12,T,N)
    P = np.stack([p.values for p in Ps])                      # (12,T,N)
    bad = np.isnan(W).any(0) | np.isnan(P).any(0)             # 결측 창
    idx = np.nanargmax(np.where(np.isnan(W), -1, W), axis=0)  # 최대 거래 달
    poc = np.take_along_axis(P, idx[None], 0)[0]              # 그 달 가격
    poc[bad] = np.nan                                         # 결측 처리
    return d.price / pd.DataFrame(poc, index=d.price.index, columns=d.price.columns) - 1  # 괴리율


def low_overhead(d):
    """상단 매물 비중 ↓: 과거 12개월 거래 중 현재가보다 높은 가격에서 거래된 비율 (VPVR 저항대)"""
    Ps, Ws = _profile(d)                                      # 프로파일
    above = sum(w * (p > d.price).astype(float) for p, w in zip(Ps, Ws))  # 현재가 위 거래량
    return -(above / sum(Ws))                                 # 부호 반전


def cgo(d):
    """미실현 자본이득(CGO, 36개월 회전율 가중 기준가격 대비) ↑ (Grinblatt–Han 2005)"""
    tv = turnover(d).clip(upper=1)                            # 회전율
    P = d.price                                               # 수정주가
    ref, wsum = 0, 0                                          # 기준가격 분자·분모
    surv = pd.DataFrame(1.0, index=P.index, columns=P.columns)  # 이후 달들에 팔리지 않고 남은 비율
    for n in range(36):                                       # 0~35개월 전
        v = tv.shift(n)                                       # n개월 전 회전율
        w = v * surv                                          # 그때 산 물량 중 지금까지 보유분
        ref = ref + w * P.shift(n)                            # 가중 가격
        wsum = wsum + w                                       # 가중치 합
        surv = surv * (1 - v)                                 # 생존 비율 갱신
    R = ref / wsum                                            # 기준가격
    return (P - R) / P                                        # CGO


# ---------------------------------------------------------------------------
# R. 시장 국면 지표 (모두 t월 말까지 정보, 임계값은 확장창(expanding) 기준)
# ---------------------------------------------------------------------------
def _expanding_z(s: pd.Series, minp: int = 24) -> pd.Series:
    """확장창 z-점수 (미래 정보 없이 과거 평균·표준편차로 표준화)"""
    return (s - s.expanding(minp).mean()) / s.expanding(minp).std()  # z


def market_state(d) -> pd.DataFrame:
    """월별 시장 국면 지표. bull_* 는 1 = 공격적 국면, 0 = 방어적 국면"""
    idx = d.ret.index                                         # 월
    mkt = d.ff["MKT"].reindex(idx)                            # 시장 총수익
    lvl = (1 + mkt.fillna(0)).cumprod()                       # 시장 지수
    mc = d.x["mcap"]                                          # 시총
    out = pd.DataFrame(index=idx)                             # 결과
    out["mkt_ret12"] = np.expm1(np.log1p(mkt).rolling(12, 12).sum())  # 12개월 시장 수익
    out["bull_trend12"] = (out["mkt_ret12"] > 0).astype(float)  # 12개월 상승 추세 (Cooper–Gutierrez–Hameed 2004)
    out["bull_ma10"] = (lvl > lvl.rolling(10, 10).mean()).astype(float)  # 10개월 이동평균 위 (Faber 2007)
    rv = (d.x["rvol_d"] * mc).sum(1) / mc.where(d.x["rvol_d"].notna()).sum(1)  # 시총가중 평균 일간변동성(VIX 대리)
    out["mkt_vol"] = rv.rolling(3, 1).mean()                  # 3개월 평균
    out["bull_lowvol"] = (out["mkt_vol"] < out["mkt_vol"].expanding(24).median()).astype(float)  # 과거 중앙값보다 낮은 변동성
    above = (d.price > d.price.rolling(12, 12).mean())        # 종목별 12개월 평균 위
    out["breadth"] = above.sum(1) / d.price.rolling(12, 12).mean().notna().sum(1)  # 상승 종목 비율(시장 폭)
    out["bull_breadth"] = (out["breadth"] > 0.5).astype(float)  # 절반 이상 상승 추세
    tv = turnover(d)                                          # 회전율
    agg_turn = (tv * mc).sum(1) / mc.where(tv.notna()).sum(1)  # 시장 회전율(시총가중)
    first = d.price.notna().idxmax()                          # 종목별 첫 등장 월
    first = first[first > idx[0]]                             # 데이터 시작월 등장은 신규상장 아님
    nipo = first.value_counts().reindex(idx, fill_value=0).rolling(12, 12).sum()  # 12개월 신규상장 수
    out["sentiment"] = (_expanding_z(np.log(agg_turn / agg_turn.rolling(36, 12).mean()))  # 거래 과열
                        + _expanding_z(np.log1p(nipo))) / 2  # + 신규상장 붐 (Baker–Wurgler 2006 구성요소)
    out["bull_calm_sent"] = (out["sentiment"] < 0.5).astype(float).where(out["sentiment"].notna())  # 심리 과열 아니면 공격
    out["bull_score"] = out[["bull_trend12", "bull_ma10", "bull_lowvol", "bull_breadth"]].mean(1)  # 합성 국면점수(0~1)
    return out                                                # 반환


# 이름 → (함수, 한글 설명, 범주)
INFO_V2 = {
    "rmw_beta": (rmw_beta, "수익성(RMW) 적재량 높음", "A.요인적재량"),
    "hml_beta": (hml_beta, "가치(HML) 적재량 높음", "A.요인적재량"),
    "cma_beta": (cma_beta, "보수투자(CMA) 적재량 높음", "A.요인적재량"),
    "ff5_alpha_36": (ff5_alpha_36, "FF5 알파(36개월)", "A.요인적재량"),
    "low_ivol_ff5_36": (low_ivol_ff5_36, "FF5 고유변동성 낮음", "A.요인적재량"),
    "low_r2_ff5": (low_r2_ff5, "가격 동조성(FF5 R²) 낮음", "A.요인적재량"),
    "low_downside_beta": (low_downside_beta, "하락장 베타 낮음", "A.요인적재량"),
    "coskew": (coskew, "음의 공왜도(−coskew)", "A.요인적재량"),
    "low_volofvol": (low_volofvol, "변동성의 변동성 낮음", "B.일간집계"),
    "low_vol_shock": (low_vol_shock, "변동성 충격 없음", "B.일간집계"),
    "liq_shock": (liq_shock, "유동성 악화 없음", "B.일간집계"),
    "low_idio_share": (low_idio_share, "고유위험 비중 낮음", "B.일간집계"),
    "low_range": (low_range, "월중 수익 범위 좁음", "B.일간집계"),
    "low_max_scaled": (low_max_scaled, "변동성 대비 MAX 낮음", "B.일간집계"),
    "low_lottery": (low_lottery, "복권성 낮음", "B.일간집계"),
    "low_dvol_cv": (low_dvol_cv, "거래대금 안정성", "B.일간집계"),
    "high_beta_d12": (high_beta_d12, "고베타(일간 12개월)", "B.일간집계"),
    "lowturn_winner": (lowturn_winner, "저회전율 승자", "C.거래량×가격"),
    "suv": (suv, "표준화 이상거래량(SUV)", "C.거래량×가격"),
    "dvol_trend": (dvol_trend, "거래대금 증가 추세", "C.거래량×가격"),
    "quiet_loser": (quiet_loser, "조용한 하락 반전", "C.거래량×가격"),
    "resid_reversal": (resid_reversal, "FF5 잔차 단기반전", "C.거래량×가격"),
    "mom_sharpe": (mom_sharpe, "위험조정 모멘텀", "D.모멘텀변형"),
    "frog_in_pan": (frog_in_pan, "정보 연속성(Frog-in-the-pan)", "D.모멘텀변형"),
    "mom_fip": (mom_fip, "모멘텀 + 정보 연속성", "D.모멘텀변형"),
    "mom_accel": (mom_accel, "모멘텀 가속", "D.모멘텀변형"),
    "high_52w_low": (high_52w_low, "52주 최저가 대비 상승", "D.모멘텀변형"),
    "low_mdd_12m": (low_mdd_12m, "12개월 최대낙폭 작음", "D.모멘텀변형"),
    "ind_lead_lag": (ind_lead_lag, "산업 대형주 선행수익", "D.모멘텀변형"),
    "intra_ind_mom": (intra_ind_mom, "산업 내 상대 모멘텀", "D.모멘텀변형"),
    "div_month": (div_month, "배당 예정월", "E.이벤트·배당"),
    "div_growth": (div_growth, "배당 성장", "E.이벤트·배당"),
    "split_drift": (split_drift, "주식분할 후 표류", "E.이벤트·배당"),
    "exch_upgrade": (exch_upgrade, "NYSE 상향 이전", "E.이벤트·배당"),
    "rsi_12": (rsi_12, "RSI 12개월(추세)", "T.기술적지표"),
    "rsi_3_oversold": (rsi_3_oversold, "RSI 3개월 과매도(반등)", "T.기술적지표"),
    "macd_line": (macd_line, "MACD선(12,26개월)", "T.기술적지표"),
    "macd_hist": (macd_hist, "MACD 히스토그램", "T.기술적지표"),
    "boll_pctb": (boll_pctb, "볼린저 %B(20개월)", "T.기술적지표"),
    "low_boll_width": (low_boll_width, "볼린저 밴드폭 좁음", "T.기술적지표"),
    "stoch_12": (stoch_12, "스토캐스틱 %K(12개월)", "T.기술적지표"),
    "vwap_dev": (vwap_dev, "12개월 VWAP 대비 위치(VPVR)", "T.기술적지표"),
    "poc_dev": (poc_dev, "POC 대비 위치(VPVR)", "T.기술적지표"),
    "low_overhead": (low_overhead, "상단 매물 비중 낮음(VPVR)", "T.기술적지표"),
    "cgo": (cgo, "미실현 자본이득(CGO)", "T.기술적지표"),
}
SIGNALS_V2 = {k: v[0] for k, v in INFO_V2.items()}            # 이름 → 함수
