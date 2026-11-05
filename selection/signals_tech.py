"""
기술적 모멘텀·추세 신호 (피드백 목록 중 보유 데이터로 가능한 것만, 월말가 기준)
=============================================================================
  1. 상대 모멘텀: ROC(1·3·6·12개월), 3·6개월 가중평균, 추세의 질(로그가격 회귀 R², Clenow 기울기×R²), 위험조정 모멘텀(6개월)
  2. 절대 모멘텀·추세 필터: 10개월(≈200일)·5개월(≈100일) 이동평균 위, 12개월 수익 > 무위험 누적(듀얼 모멘텀)
  3. 돌파: 3·6개월(≈60·120일) 돈키언 신고가 돌파 강도, 볼린저(20개월, 2σ) 상단 돌파 강도
  4. 종합 점수: Rank(ROC 3M) + Rank(RSI 14M) + Rank(MACD 히스토그램), RSI 50~70 구간 우대, MACD>0·히스토그램>0
  ※ 일간 가격 원자료가 없어 20일 돈키언·일봉 볼린저·RSI(14일)는 월간 버전으로 대체
"""
import numpy as np                           # 수치
import pandas as pd                          # 데이터프레임
from signals import cum_return, rmean        # 공통 유틸
from signals_v2 import _macd, _rsi           # MACD·RSI 재사용


def _pct(x): return x.rank(axis=1, pct=True)                  # 횡단면 백분위


def roc(d, n): return cum_return(d.ret, 0, n - 1)             # 최근 n개월 누적수익(당월 포함)
def roc_1(d): return roc(d, 1)                                # 1개월
def roc_3(d): return roc(d, 3)                                # 3개월
def roc_6(d): return roc(d, 6)                                # 6개월
def roc_12(d): return roc(d, 12)                              # 12개월(최근월 포함)
def roc_3_6(d): return 0.5 * roc(d, 3) + 0.5 * roc(d, 6)      # 3·6개월 가중평균(각 50%)


def _log_trend(d, n=12):
    """최근 n개월 로그 수정주가를 시간에 회귀 → (기울기, R²)"""
    y = np.log(d.price)                                       # 로그가격
    x = np.arange(n) - (n - 1) / 2                            # 중심화 시간
    ys = [y.shift(n - 1 - i) for i in range(n)]               # 창 내 가격(오래된 것부터)
    ybar = sum(ys) / n                                        # 평균
    sxx = (x ** 2).sum()                                      # Σx²
    sxy = sum(xi * yi for xi, yi in zip(x, ys))               # Σx·y (중심화 x)
    b = sxy / sxx                                             # 기울기(월)
    sst = sum((yi - ybar) ** 2 for yi in ys)                  # 총변동
    r2 = (b ** 2 * sxx) / sst                                 # 결정계수
    return b, r2                                              # 반환


def trend_r2(d):
    """추세의 질: sign(기울기) × R² (12개월 로그가격 회귀)"""
    b, r2 = _log_trend(d)                                     # 회귀
    return np.sign(b) * r2                                    # 부호 × R²


def clenow(d):
    """Clenow 모멘텀: 연율화 기울기 × R²"""
    b, r2 = _log_trend(d)                                     # 회귀
    return (np.exp(12 * b) - 1) * r2                          # 연율 기울기 × R²


def mom_sharpe_6(d):
    """위험조정 모멘텀(6개월): 6개월 수익 ÷ 6개월 월수익 표준편차"""
    return roc(d, 6) / d.ret.rolling(6, 6).std()              # 비율


def above_sma(d, n):
    """추세 필터: 월말가 > n개월 이동평균 (True/False)"""
    return d.price > d.price.rolling(n, n).mean()             # 필터


def abs_mom(d):
    """절대 모멘텀 필터: 12개월 수익 > 같은 기간 무위험 누적수익"""
    rf12 = np.expm1(np.log1p(d.ff["RF"].reindex(d.ret.index)).rolling(12, 12).sum())  # 무위험 누적
    return roc(d, 12).gt(rf12, axis=0)                        # 필터


def donchian(d, n):
    """돈키언 돌파 강도: 월말가 ÷ 직전 n개월 최고 월말가 (>1 이면 신고가 돌파)"""
    return d.price / d.price.shift(1).rolling(n, n).max()     # 비율


def donchian_3(d): return donchian(d, 3)                      # ≈60일
def donchian_6(d): return donchian(d, 6)                      # ≈120일


def boll_break(d):
    """볼린저 상단 돌파 강도: (가격 − 상단) ÷ 2σ, 20개월·2σ (>0 이면 상단 위 마감)"""
    ma, sd = d.price.rolling(20, 20).mean(), d.price.rolling(20, 20).std()  # 중심·표준편차
    return (d.price - (ma + 2 * sd)) / (2 * sd)               # 돌파 강도


def rsi_band(d):
    """RSI(14개월) 50~70 우대: 구간 안은 1 − |RSI−0.6|, 밖은 0 − 거리"""
    r = _rsi(d, 14)                                           # RSI(0~1)
    inside = (r >= 0.5) & (r <= 0.7)                          # 건강한 모멘텀 구간
    return (1 - (r - 0.6).abs()).where(inside, -(r - 0.6).abs()).where(r.notna())  # 점수


def tech_composite(d):
    """종합 기술 점수 = Rank(ROC 3M) + Rank(RSI 14M) + Rank(MACD 히스토그램) (동일 가중)"""
    return (_pct(roc(d, 3)) + _pct(_rsi(d, 14)) + _pct(_macd(d)[1])) / 3  # 평균


def tech_composite_f(d):
    """필터형 종합 점수: 종합 점수 중 RSI 50~70 · MACD선>0 · 히스토그램>0 · 10개월선 위 종목만"""
    line, hist = _macd(d)                                     # MACD
    r = _rsi(d, 14)                                           # RSI
    ok = (r >= 0.5) & (r <= 0.7) & (line > 0) & (hist > 0) & above_sma(d, 10)  # 조건
    return tech_composite(d).where(ok)                        # 조건 밖 제외


INFO_TECH = {
    "roc_1": (roc_1, "ROC 1개월", "1.상대모멘텀"),
    "roc_3": (roc_3, "ROC 3개월", "1.상대모멘텀"),
    "roc_6": (roc_6, "ROC 6개월", "1.상대모멘텀"),
    "roc_12": (roc_12, "ROC 12개월", "1.상대모멘텀"),
    "roc_3_6": (roc_3_6, "ROC 3·6개월 가중평균", "1.상대모멘텀"),
    "trend_r2": (trend_r2, "추세의 질(부호×R²)", "1.상대모멘텀"),
    "clenow": (clenow, "Clenow(연율 기울기×R²)", "1.상대모멘텀"),
    "mom_sharpe_6": (mom_sharpe_6, "위험조정 모멘텀 6개월", "1.상대모멘텀"),
    "donchian_3": (donchian_3, "돈키언 3개월 돌파 강도", "3.돌파"),
    "donchian_6": (donchian_6, "돈키언 6개월 돌파 강도", "3.돌파"),
    "boll_break": (boll_break, "볼린저 상단 돌파 강도", "3.돌파"),
    "rsi_band": (rsi_band, "RSI 50~70 우대", "4.종합점수"),
    "tech_composite": (tech_composite, "종합(ROC3+RSI+MACD히스토)", "4.종합점수"),
    "tech_composite_f": (tech_composite_f, "종합 + RSI밴드·MACD>0·10개월선 필터", "4.종합점수"),
}
