"""
종목 선택 신호(Signal) 정의
============================
[규칙]
  - 모든 신호 함수는 f(d: MarketData) -> DataFrame (행=월 Period, 열=PERMNO) 을 반환
  - 값이 클수록 "좋은 종목" (Top-k는 큰 값부터 선택, D1 = 최상위 10%) → 문헌상 방향으로 부호를 맞춤
  - t행의 값은 반드시 t월 말까지의 정보만으로 계산 → t+1월에 보유 (미래정보 누설 금지)
  - 새 신호는 함수를 만들고 SIGNALS(함수)와 INFO(설명)에 등록하면 자동 인식

[데이터 출처]
  - d.ret / d.price / d.vol : 과제 파일(ADJ_PRC, VOL)
  - d.x["mcap","ret_tot","retx","mthvol","shrout","siccd","prc"] : CRSP 월별(CIZ) 보조 파일
  - d.x["max_d","min_d","ivol_d","beta_d","rvol_d","skew_d","illiq_d","dvol_m"] : 일간 자료 기반 월별 특성치
  - d.ff : Fama-French 팩터

[재무 데이터 필요 → 제외] asset turnover, EBITDA/EV, mispricing(performance), 가치(B/M), 수익성, 투자 등
"""
import numpy as np                           # 수치 계산
import pandas as pd                          # 데이터프레임
from selection_eval import MarketData        # 데이터 묶음 타입


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def cum_return(ret: pd.DataFrame, start_lag: int, end_lag: int) -> pd.DataFrame:
    """t−end_lag ~ t−start_lag 구간 누적수익률 (구간 내 수익률이 모두 있어야 계산)"""
    n = end_lag - start_lag + 1                               # 구간 길이
    lr = np.log1p(ret)                                        # 로그수익률
    s = lr.rolling(n, min_periods=n).sum().shift(start_lag)   # 구간 합(결측 있으면 NaN) 후 시차 적용
    return np.expm1(s)                                        # 단순 누적수익률로 환산


def rmean(x: pd.DataFrame, n: int, min_periods: int = None) -> pd.DataFrame:
    """최근 n개월 이동평균 (t 포함)"""
    return x.rolling(n, min_periods=min_periods or n).mean()  # 이동평균


def roll_beta(y: pd.DataFrame, m: pd.Series, n: int, minp: int) -> pd.DataFrame:
    """종목별 롤링 베타 cov(y,m)/var(m) — 종목마다 관측 가능한 달만 사용"""
    mask = y.notna()                                          # 관측 여부
    M = pd.DataFrame(np.where(mask, m.reindex(y.index).values[:, None], np.nan),
                     index=y.index, columns=y.columns)        # 종목별로 관측 달만 남긴 시장수익률
    my, mm = y.rolling(n, minp).mean(), M.rolling(n, minp).mean()  # 평균
    cov = (y * M).rolling(n, minp).mean() - my * mm           # 공분산
    var = (M * M).rolling(n, minp).mean() - mm ** 2           # 분산
    return cov / var                                          # 베타


def by_industry(x: pd.DataFrame, sic: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """월별로 SIC 2자리 산업 평균값을 각 종목에 배치"""
    sic2 = np.floor(sic / 100)                                # SIC 2자리
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)  # 결과
    for t in x.index:                                         # 월마다
        v, g = x.loc[t], sic2.loc[t]                          # 값, 산업
        ok = v.notna() & g.notna()                            # 둘 다 있는 종목
        if ok.sum() == 0:                                     # 없으면
            continue                                          # 건너뜀
        out.loc[t, ok[ok].index] = v[ok].groupby(g[ok]).transform(how).values  # 산업 평균 배치
    return out                                                # 반환


def turnover(d: MarketData) -> pd.DataFrame:
    """월 주식회전율 = 거래량 / 발행주식수 (같은 달 원단위끼리 비율 → 분할 영향 없음)"""
    return d.x["mthvol"] / d.x["shrout"]                      # 회전율


# ---------------------------------------------------------------------------
# A. 모멘텀/반전 (과제 파일 수익률)
# ---------------------------------------------------------------------------
def mom_12_1(d): return cum_return(d.ret, 1, 11)              # 모멘텀 12-1
def mom_6_1(d): return cum_return(d.ret, 1, 5)                # 모멘텀 6-1
def mom_12_7(d): return cum_return(d.ret, 6, 11)              # 중기 모멘텀(t−11~t−6, Novy-Marx 2012)
def st_reversal(d): return -d.ret                             # 단기 반전(직전월 수익 낮을수록 좋음)
def lt_reversal(d): return -cum_return(d.ret, 12, 35)         # 장기 반전(t−35~t−12 수익 낮을수록 좋음)
def high_52w(d): return d.price / d.price.rolling(12, min_periods=12).max()  # 52주 신고가 근접도


def mom_consistency(d):
    """모멘텀 일관성: t−11~t−1 중 양(+)수익 달의 비율 (Grinblatt–Moskowitz 2004)"""
    pos = (d.ret > 0).astype(float).where(d.ret.notna())      # 양수익 여부(결측 유지)
    return pos.rolling(11, min_periods=11).mean().shift(1)    # 11개월 비율, 최근월 제외


def trend_ma(d):
    """추세: 현재가 / 12개월 이동평균 가격 (Han–Zhou–Zhu 2016 단순형)"""
    return d.price / rmean(d.price, 12)                       # 이동평균 대비 위치


def seasonality(d):
    """계절성: 과거 같은 달(12·24·36·48·60개월 전) 평균 수익률 (Heston–Sadka 2008)"""
    stack = np.stack([d.ret.shift(k).values for k in (11, 23, 35, 47, 59)])  # t+1과 같은 달 수익률들
    cnt = np.sum(~np.isnan(stack), axis=0)                    # 사용 가능 개수
    avg = np.nanmean(np.where(cnt >= 2, 1, np.nan) * stack, axis=0)  # 2개 이상일 때 평균
    return pd.DataFrame(avg, index=d.ret.index, columns=d.ret.columns)  # 복원


def ind_mom(d):
    """산업 모멘텀: SIC 2자리 산업의 6-1 모멘텀 평균 (Moskowitz–Grinblatt 1999)"""
    return by_industry(mom_6_1(d), d.x["siccd"])              # 산업 평균


def ind_rel_reversal(d):
    """산업 상대 단기반전: −(직전월 수익 − 산업 평균) (Da–Liu–Schaumburg 2014)"""
    return -(d.ret - by_industry(d.ret, d.x["siccd"]))        # 산업 대비 초과 하락일수록 좋음


# ---------------------------------------------------------------------------
# B. 위험/변동성 (월간)
# ---------------------------------------------------------------------------
def low_vol_12m(d): return -d.ret.rolling(12, min_periods=12).std()   # 월간 변동성 12개월
def low_vol_36m(d): return -d.ret.rolling(36, min_periods=24).std()   # 월간 변동성 36개월


def low_beta_60m(d):
    """CAPM 베타(월간 60개월, 최소 24개월): 낮을수록 좋음 (Black–Jensen–Scholes 1972)"""
    ex = d.ret.sub(d.ff["RF"].reindex(d.ret.index), axis=0)   # 초과수익
    return -roll_beta(ex, d.ff["Mkt-RF"], 60, 24)             # 부호 반전


_FF3_CACHE = {}                                               # FF3 롤링 결과 캐시


def _ff3_rolling(d: MarketData, window: int = 36):
    """종목별 롤링 FF3 회귀(창 36개월, 결측 없는 종목만) → 알파, 잔차변동성, 잔차모멘텀"""
    key = (id(d), window)                                     # 캐시 키
    if key in _FF3_CACHE:                                     # 이미 계산했으면
        return _FF3_CACHE[key]                                # 재사용
    ff = d.ff.reindex(d.ret.index)                            # 팩터 기간 맞춤
    Y = d.ret.sub(ff["RF"], axis=0).values                    # 종목 초과수익
    Xall = np.column_stack([np.ones(len(ff)), ff[["Mkt-RF", "SMB", "HML"]].values])  # 상수+FF3
    alpha, ivol, rmom = (np.full(Y.shape, np.nan) for _ in range(3))  # 결과 배열 3개
    for t in range(window - 1, len(Y)):                       # 창이 채워지는 시점부터
        X = Xall[t - window + 1:t + 1]                        # 창 내 팩터
        y = Y[t - window + 1:t + 1]                           # 창 내 수익률
        ok = ~np.isnan(y).any(0)                              # 결측 없는 종목
        if ok.sum() == 0 or np.isnan(X).any():                # 대상 없음/팩터 결측
            continue                                          # 건너뜀
        B, *_ = np.linalg.lstsq(X, y[:, ok], rcond=None)      # 전 종목 동시 OLS
        e = y[:, ok] - X @ B                                  # 잔차
        alpha[t, ok] = B[0]                                   # 월 알파
        ivol[t, ok] = e.std(0, ddof=4)                        # 잔차 표준편차
        e12 = e[-12:-1]                                       # t−11 ~ t−1 잔차
        rmom[t, ok] = e12.sum(0) / e12.std(0, ddof=1)         # 표준화 잔차 누적 (Blitz–Huij–Martens 2011)
    to_df = lambda a: pd.DataFrame(a, index=d.ret.index, columns=d.ret.columns)  # 배열→데이터프레임
    _FF3_CACHE[key] = (to_df(alpha), to_df(ivol), to_df(rmom))  # 캐시
    return _FF3_CACHE[key]                                    # 반환


def resid_mom(d): return _ff3_rolling(d)[2]                   # 잔차 모멘텀
def ff3_alpha_36(d): return _ff3_rolling(d)[0]                # FF3 알파(36개월)
def low_ivol_ff3_36(d): return -_ff3_rolling(d)[1]            # FF3 고유변동성(36개월)


# ---------------------------------------------------------------------------
# C. 규모/유동성/주식수 (CRSP 월별 보조 파일)
# ---------------------------------------------------------------------------
def size_small(d): return -np.log(d.x["mcap"])                # 소형주
def size_big(d): return np.log(d.x["mcap"])                   # 대형주
def high_price(d): return np.log(d.x["prc"].abs())            # 명목주가(저가주 회피)
def low_turnover(d): return -rmean(turnover(d), 12, 9)        # 저회전율(12개월)
def low_turnover_vol(d): return -turnover(d).rolling(12, min_periods=9).std()  # 회전율 변동성 낮을수록(Chordia et al. 2001)
def abn_volume(d): return turnover(d) / rmean(turnover(d), 12, 9).shift(1)  # 이상거래량(고거래량 프리미엄, Gervais et al. 2001)
def low_dvol(d): return -np.log(rmean(d.x["dvol_m"], 12, 9))  # 거래대금 낮을수록(Brennan et al. 1998)


def net_issuance(d):
    """순주식발행(12개월): Δlog(시총) − Σlog(1+배당제외수익). 낮을수록 좋음 (Pontiff–Woodgate 2008)"""
    return -(np.log(d.x["mcap"]).diff(12) - np.log1p(d.x["retx"]).rolling(12, min_periods=12).sum())  # 부호 반전


def comp_issuance(d):
    """복합 발행(36개월): Δlog(시총) − Σlog(1+총수익). 낮을수록 좋음 (Daniel–Titman 2006)"""
    return -(np.log(d.x["mcap"]).diff(36) - np.log1p(d.x["ret_tot"]).rolling(36, min_periods=36).sum())  # 부호 반전


def div_yield(d):
    """배당수익률(12개월 합): 총수익 − 가격수익. 높을수록 좋음"""
    return (d.x["ret_tot"] - d.x["retx"]).rolling(12, min_periods=12).sum()  # 월 배당수익 합


# ---------------------------------------------------------------------------
# D. 일간 자료 기반 (daily_signals.csv)
# ---------------------------------------------------------------------------
def low_max(d): return -d.x["max_d"]                          # MAX(직전월 일간 최고수익) (Bali et al. 2011)
def low_max_3m(d): return -rmean(d.x["max_d"], 3)             # MAX 3개월 평균
def high_min(d): return d.x["min_d"]                          # MIN(직전월 일간 최저수익이 덜 나쁠수록)
def low_rvol_1m(d): return -d.x["rvol_d"]                     # 일간 변동성(1개월)
def low_rvol_d(d): return -rmean(d.x["rvol_d"], 3)            # 일간 변동성(3개월)
def low_rvol_12m(d): return -rmean(d.x["rvol_d"], 12, 9)      # 일간 변동성(12개월)
def low_ivol_d(d): return -d.x["ivol_d"]                      # 고유변동성(CAPM 21일) (Ang et al. 2006)
def low_ivol_3m(d): return -rmean(d.x["ivol_d"], 3)           # 고유변동성 3개월 평균
def low_beta_1m(d): return -d.x["beta_d"]                     # 일간 베타(1개월)
def low_skew(d): return -d.x["skew_d"]                        # 일간 왜도(1개월)
def low_skew_3m(d): return -rmean(d.x["skew_d"], 3)           # 일간 왜도(3개월)
def illiq(d): return rmean(d.x["illiq_d"], 12, 9)             # Amihud 비유동성(높을수록 프리미엄)


def low_beta_fp(d):
    """Frazzini–Pedersen 형 베타(일간 베타 12개월 평균, 1 방향 축소): 낮을수록 좋음 (BAB)"""
    b = rmean(d.x["beta_d"].clip(-3, 5), 12, 9)               # 월별 일간베타 평균(극단값 절단)
    return -(0.6 * b + 0.4)                                   # Vasicek형 축소


def lowvol_liq(d):
    """LOWVOL_LIQ: 저변동성 + 고유동성 (일간변동성 3개월·거래대금 3개월의 횡단면 백분위 평균)"""
    v = (-rmean(d.x["rvol_d"], 3)).rank(axis=1, pct=True)    # 저변동성 백분위
    q = rmean(d.x["dvol_m"], 3).rank(axis=1, pct=True)        # 거래대금 백분위
    return (v + q) / 2                                        # 평균


# 이름 → (함수, 한글 설명, 범주)
INFO = {
    "mom_12_1": (mom_12_1, "모멘텀 12-1개월", "모멘텀/반전"),
    "mom_6_1": (mom_6_1, "모멘텀 6-1개월", "모멘텀/반전"),
    "mom_12_7": (mom_12_7, "중기 모멘텀 12-7개월", "모멘텀/반전"),
    "mom_consistency": (mom_consistency, "모멘텀 일관성(양수익 달 비율)", "모멘텀/반전"),
    "resid_mom": (resid_mom, "잔차 모멘텀(FF3)", "모멘텀/반전"),
    "trend_ma": (trend_ma, "추세(가격/12개월 이동평균)", "모멘텀/반전"),
    "high_52w": (high_52w, "52주 신고가 근접도", "모멘텀/반전"),
    "ind_mom": (ind_mom, "산업 모멘텀", "모멘텀/반전"),
    "seasonality": (seasonality, "계절성(같은 달 과거수익)", "모멘텀/반전"),
    "st_reversal": (st_reversal, "단기 반전(1개월)", "모멘텀/반전"),
    "ind_rel_reversal": (ind_rel_reversal, "산업 상대 단기반전", "모멘텀/반전"),
    "lt_reversal": (lt_reversal, "장기 반전(36-13개월)", "모멘텀/반전"),
    "low_vol_12m": (low_vol_12m, "저변동성(월간 12개월)", "위험/변동성"),
    "low_vol_36m": (low_vol_36m, "저변동성(월간 36개월)", "위험/변동성"),
    "low_rvol_1m": (low_rvol_1m, "저변동성(일간 1개월)", "위험/변동성"),
    "low_rvol_d": (low_rvol_d, "저변동성(일간 3개월)", "위험/변동성"),
    "low_rvol_12m": (low_rvol_12m, "저변동성(일간 12개월)", "위험/변동성"),
    "low_ivol_d": (low_ivol_d, "저고유변동성(CAPM 21일)", "위험/변동성"),
    "low_ivol_3m": (low_ivol_3m, "저고유변동성(CAPM, 3개월 평균)", "위험/변동성"),
    "low_ivol_ff3_36": (low_ivol_ff3_36, "저고유변동성(FF3 월간 36개월)", "위험/변동성"),
    "low_beta_60m": (low_beta_60m, "저베타(CAPM 월간 60개월)", "위험/변동성"),
    "low_beta_fp": (low_beta_fp, "저베타(Frazzini–Pedersen형)", "위험/변동성"),
    "low_beta_1m": (low_beta_1m, "저베타(일간 1개월)", "위험/변동성"),
    "low_max": (low_max, "저MAX(일간 최고수익 1개월)", "위험/변동성"),
    "low_max_3m": (low_max_3m, "저MAX(3개월 평균)", "위험/변동성"),
    "high_min": (high_min, "MIN(일간 최저수익이 덜 나쁨)", "위험/변동성"),
    "low_skew": (low_skew, "저왜도(일간 1개월)", "위험/변동성"),
    "low_skew_3m": (low_skew_3m, "저왜도(일간 3개월)", "위험/변동성"),
    "ff3_alpha_36": (ff3_alpha_36, "FF3 알파(36개월)", "위험/변동성"),
    "lowvol_liq": (lowvol_liq, "LOWVOL_LIQ(저변동성+유동성)", "위험/변동성"),
    "size_small": (size_small, "소형주(시가총액 작음)", "규모/유동성"),
    "size_big": (size_big, "대형주(시가총액 큼)", "규모/유동성"),
    "high_price": (high_price, "고명목주가", "규모/유동성"),
    "low_turnover": (low_turnover, "저회전율(12개월)", "규모/유동성"),
    "low_turnover_vol": (low_turnover_vol, "회전율 변동성 낮음", "규모/유동성"),
    "abn_volume": (abn_volume, "이상거래량(고거래량 프리미엄)", "규모/유동성"),
    "low_dvol": (low_dvol, "거래대금 낮음", "규모/유동성"),
    "illiq": (illiq, "Amihud 비유동성", "규모/유동성"),
    "net_issuance": (net_issuance, "순주식발행 적음(12개월)", "발행/배당"),
    "comp_issuance": (comp_issuance, "복합발행 적음(36개월)", "발행/배당"),
    "div_yield": (div_yield, "배당수익률", "발행/배당"),
}
SIGNALS = {k: v[0] for k, v in INFO.items()}                  # 이름 → 함수 (기존 인터페이스 유지)
