"""
종목 선택(Top-k selection) 기준 검증 프레임워크
====================================================
[사용 데이터] data 폴더의 CSV(과제 파일 + WRDS 보조 2종, data/selection_inputs) + Fama-French 5 Factor 파일만 사용 (결합 근거: data_audit.py)

[분석 1] Top-k 포트폴리오 vs 시장 포트폴리오
  - 신호 상위 k(20/30/50/100)개 종목, 동일가중, 월 1회 리밸런싱
  - 거래량(VOL) 하위 20% 종목 배제 (매 형성월 횡단면 기준)
  - 거래비용 = 명목 회전율 × 왕복 20bp
  - 보유기간: 2003-02 ~ 2020-12 (215개월, 길이 검증 후 진행)
  - 지표: 연수익, 변동성, Sharpe, MDD, ΔSR(p값), 베타(FF3), 회전율 (+FF3 알파 등)

[분석 2] 10분위(Decile) 분석
  - 신호 기준 1등(D1)~꼴등(D10) 10% 단위 10그룹, 동일가중, 월 1회 리밸런싱
  - 그룹별 평균수익률/유의성, 그룹 간 차이 유의성(쌍별 t, 결합 Wald), 단조성 검정, D1−D10 스프레드

[지표 정의 통일]
  - 연수익   : (1 + r.mean())**12 − 1
  - 변동성   : r.std(ddof=1) * sqrt(12)
  - Sharpe   : mean(r − rf) / std(r − rf, ddof=1) * sqrt(12)   (분자·분모 모두 초과수익)
  - 베타     : FF3 회귀 (r − rf) = α + β·MKT + s·SMB + h·HML 의 β
  - 회전율   : 명목(목표비중 기준) 편도 회전율 = ½ Σ|w_t − w_{t−1}|  (가격변동에 따른 비중 drift 미반영)
  - 거래비용 : 회전율 × 20bp(왕복)  → 매수 10bp + 매도 10bp 와 동일
  - ΔSR p값 : Ledoit–Wolf(2008) HAC 델타법(주), Jobson–Korkie/Memmel(2003)(참고)

[시점 규칙 — 미래정보 누설 방지]
  - 형성월 t 말의 정보(가격·수익률·거래량 ≤ t)로 신호/유니버스/비중 결정 → t+1월 수익률로 성과 측정
  - 수익률 인덱스는 "보유월(t+1)" 기준으로 표기
"""
from dataclasses import dataclass, field    # 설정/데이터 묶음을 위한 dataclass
from typing import Callable, Dict, Optional  # 타입 힌트
import os                                    # 경로 처리
import warnings                              # 경고 출력
import numpy as np                           # 수치 계산
import pandas as pd                          # 데이터프레임 처리
from scipy import stats                      # 정규/카이제곱 분포, 순위상관

BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # 이 파일이 있는 폴더(selection)
ROOT_DIR = os.path.dirname(BASE_DIR)                            # 저장소 루트
INPUT_DIR = os.path.join(ROOT_DIR, "data", "selection_inputs")  # 선정 전용 입력 폴더
OUT_DIR = os.environ.get("SELECTION_OUTPUT_DIR", os.path.join(ROOT_DIR, "results", "reproduced", "selection"))  # 결과·캐시 폴더
FF_PATH = os.path.join(INPUT_DIR, "ff5_monthly.csv")            # FF5 월별(선정 당시 Kenneth French 사이트 수집본, 소수 단위)


# ---------------------------------------------------------------------------
# 0. 설정
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """백테스트 공통 설정값 (모든 신호에 동일하게 적용해야 비교가 공정함)"""
    start: str = "2003-02"                   # 첫 보유월 (형성월은 2003-01)
    end: str = "2020-12"                     # 마지막 보유월
    n_months: int = 215                      # 기대 수익률 시리즈 길이 (검증용)
    top_ks: tuple = (20, 30, 50, 100)        # 분석 1의 k 값들
    vol_cut: float = 0.20                    # 거래량 하위 20% 배제
    exclude_shell: bool = True               # SPAC(기업인수목적회사, 현금성 껍데기 회사) 제외
    size_cut: float = 0.0                    # 시가총액 하위 제외 비율(0이면 미적용)
    size_basis: str = "nyse"                 # 시총 분위수 기준: "nyse"(NYSE 브레이크포인트) 또는 "all"(전체 종목)
    dvol_cut: float = 0.0                    # 월 거래대금(dvol_m, 달러) 하위 제외 비율(0이면 미적용)
    price_min: float = 0.0                   # 그 달 실제 종가(CIZ MthPrc) 하한(달러, 0이면 미적용)
    min_days: int = 0                        # 그 달 최소 거래일 수(일간 자료 n, 0이면 미적용)
    cost_rt: float = 0.0020                  # 왕복 거래비용 20bp
    missing_ret_fill: float = 0.0            # 보유 중 다음달 수익률이 없는 종목(상장폐지 등)의 수익률 가정값
    weighting: str = "ew"                    # 포트폴리오 가중: "ew"(동일가중) 또는 "vw"(형성월 시가총액 비례)
    vw_cap: float = 0.0                      # 시총가중 시 종목당 비중 상한(0이면 미적용, 초과분은 나머지에 비례 재배분)
    vw_cap_mult: float = 0.0                 # 시총가중 시 종목당 상한 = 동일가중 비중의 몇 배(2 → 최대 2/n, 0이면 미적용)
    exclude_delist_month: bool = False       # 형성월에 상장폐지된(CIZ 상태 D) 종목은 매수 불가로 제외
    n_groups: int = 10                       # 분석 2의 분위 수
    nw_lags: Optional[int] = None            # Newey-West 시차 (None이면 floor(4(T/100)^(2/9)) 자동)
    n_boot: int = 2000                       # 단조성(MR) 검정 부트스트랩 횟수
    seed: int = 42                           # 난수 시드(재현성)

    def form_months(self) -> pd.PeriodIndex:
        """형성월 목록: 보유월보다 1개월 앞선 구간"""
        s = pd.Period(self.start, "M") - 1   # 첫 형성월
        e = pd.Period(self.end, "M") - 1     # 마지막 형성월
        return pd.period_range(s, e, freq="M")  # 월 단위 기간 인덱스

    def lags(self, T: int) -> int:
        """Newey-West 시차 결정"""
        if self.nw_lags is not None:         # 사용자가 지정했으면
            return self.nw_lags              # 그대로 사용
        return int(np.floor(4 * (T / 100) ** (2 / 9)))  # 자동 규칙(T=215 → 4)


# ---------------------------------------------------------------------------
# 1. 데이터 적재
# ---------------------------------------------------------------------------
@dataclass
class MarketData:
    """넓은 형태(행=월, 열=PERMNO)로 정리한 데이터 묶음"""
    price: pd.DataFrame                      # 수정주가(ADJ_PRC, 절댓값)
    ret: pd.DataFrame                        # 월 수익률 (ADJ_PRC 기반, 연속월일 때만 계산)
    vol: pd.DataFrame                        # 월 거래량(VOL)
    exch: pd.DataFrame                       # 거래소 코드
    ff: pd.DataFrame                         # FF5 팩터 + RF (월, 소수 단위)
    info: pd.DataFrame = field(default=None)  # PERMNO별 최신 TICKER/COMNAM (참고용)
    fill_ret: pd.DataFrame = field(default=None)  # ADJ_PRC 수익률 결측 시 대체 수익률(V2 MthRet: 상장폐지·공백월)
    delist_ret: pd.DataFrame = field(default=None)  # 상폐월 수익률 덮어쓰기(분류 정책: 일반·불명확 −50%, M&A·재분류 CIZ 값)
    last_month: pd.Series = field(default=None)  # PERMNO별 마지막 관측월(그 뒤로 수익률 흔적이 없으면 퇴출로 간주)
    missing_exit_ret: Optional[float] = None     # 흔적 없이 사라진 종목의 가정 수익률(None이면 Config.missing_ret_fill)
    x: dict = field(default_factory=dict)    # 보조 특성치(시가총액·일간신호 등) 넓은 형태 모음


def _parse_dates(s: pd.Series) -> pd.PeriodIndex:
    """date 열을 월(Period) 단위로 변환 (YYYYMMDD 정수/문자열, YYYY-MM-DD 모두 처리)"""
    if pd.api.types.is_numeric_dtype(s):                      # 숫자형(예: 20000131)이면
        dt = pd.to_datetime(s.astype("int64").astype(str), format="%Y%m%d")  # 문자열로 바꿔 파싱
    else:                                                     # 문자열이면
        dt = pd.to_datetime(s)                                # 형식 자동 추론
    return dt.dt.to_period("M")                               # 월 단위로 변환


def load_ff(path: str = FF_PATH) -> pd.DataFrame:
    """저장된 FF5 월별 파일을 읽는다 (소수 단위, 인덱스=월 Period)"""
    ff = pd.read_csv(path, index_col=0)                       # CSV 읽기
    ff.index = pd.PeriodIndex(ff.index.astype(str), freq="M")  # 인덱스를 월 Period로
    ff["MKT"] = ff["Mkt-RF"] + ff["RF"]                       # 시장 포트폴리오 총수익률 = 초과수익 + 무위험
    return ff                                                 # 반환


SPAC_NAME = r"ACQUISI?TIONS?|\bACQ\b|ACQUICOR|ACQUIRCO|BLANK CHECK"  # SPAC식 회사명 패턴(ACQUISITION·복수형·오타·약칭 ACQ 포함)


def load_crsp(csv_path: str, ff_path: str = FF_PATH) -> MarketData:
    """과제 CSV를 읽어 정제 후 넓은 형태로 변환한다."""
    df = pd.read_csv(csv_path, low_memory=False)              # 원본 CSV 읽기
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]  # 열 이름 통일(ADJ PRC → ADJ_PRC)
    df["PERMNO"] = pd.to_numeric(df["PERMNO"], errors="coerce")  # 종목ID 숫자화
    df["EXCHCD"] = pd.to_numeric(df["EXCHCD"], errors="coerce")  # 거래소 코드 숫자화
    df["ADJ_PRC"] = pd.to_numeric(df["ADJ_PRC"], errors="coerce").abs()  # 가격 숫자화, 음수(호가중간값 표시) 절댓값
    df["VOL"] = pd.to_numeric(df["VOL"], errors="coerce")     # 거래량 숫자화
    df["month"] = _parse_dates(df["DATE"])                    # 날짜 → 월
    n0 = len(df)                                              # 정제 전 행 수 기록
    df = df[df["EXCHCD"].isin([1, 2, 3])]                     # NYSE/AMEX/NASDAQ만 사용
    df = df[df["PERMNO"].notna()]                             # 종목ID 없는 행 제거
    df.loc[df["ADJ_PRC"] <= 0, "ADJ_PRC"] = np.nan            # 0 이하 가격은 결측 처리
    df = df.sort_values(["PERMNO", "month"])                  # 종목·월 순 정렬
    ndup = df.duplicated(["PERMNO", "month"], keep="last").sum()  # 종목-월 중복 개수
    df = df.drop_duplicates(["PERMNO", "month"], keep="last")  # 중복 시 마지막 행 유지
    df["PERMNO"] = df["PERMNO"].astype("int64")               # 종목ID 정수화
    print(f"[load_crsp] 원본 {n0:,}행 → 정제 후 {len(df):,}행 (거래소 필터·중복 {ndup}건 제거)")  # 정제 결과 출력
    months = pd.period_range(df["month"].min(), df["month"].max(), freq="M")  # 빈 달 없이 전체 월 인덱스
    price = df.pivot(index="month", columns="PERMNO", values="ADJ_PRC").reindex(months)  # 가격 넓은 형태
    vol = df.pivot(index="month", columns="PERMNO", values="VOL").reindex(months)        # 거래량 넓은 형태
    exch = df.pivot(index="month", columns="PERMNO", values="EXCHCD").reindex(months)    # 거래소 넓은 형태
    df["SHELL"] = df["COMNAM"].astype(str).str.contains(SPAC_NAME, regex=True).astype(float)  # 그 달 회사명이 SPAC식 이름인지
    shell = df.pivot(index="month", columns="PERMNO", values="SHELL").reindex(months).fillna(0) > 0  # SPAC 이름 여부(넓은 형태)
    ret = price / price.shift(1) - 1                          # 월 수익률: 직전월 가격이 있을 때만 계산(공백 월은 NaN)
    info = df.groupby("PERMNO")[["TICKER", "COMNAM"]].last() if "TICKER" in df else None  # 참고용 종목 정보
    ff = load_ff(ff_path)                                     # FF 팩터 읽기
    return MarketData(price=price, ret=ret, vol=vol, exch=exch, ff=ff, info=info, x={"shell": shell})  # 묶어서 반환


V2_PATH = os.path.join(INPUT_DIR, "crsp_monthly_ciz.csv.gz")     # WRDS CRSP 월별(CIZ) 파일(원본명 llfzpekib12cqzql.csv): 시가총액/발행주식/상폐수익률
DAILY_PATH = os.path.join(INPUT_DIR, "daily_signals.csv.gz")     # 일간 데이터로 만든 월별 특성치 파일(일간 베타 beta_d 등)
IFE_PATH = os.path.join(ROOT_DIR, "data", "raw", "security_prices.csv.gz")  # 과제 기본 파일(원본명 IFE_term_stock_data.csv)


def load_v2(path: str, permnos, months) -> Dict[str, pd.DataFrame]:
    """CRSP 월별(CIZ) 파일 → 과제 종목·기간에 맞춘 넓은 형태 특성치"""
    cols = ["PERMNO", "MthCalDt", "TradingStatusFlg", "SICCD", "MthPrc", "MthCap",
            "MthRet", "MthRetx", "MthVol", "ShrOut"]           # 필요한 열만
    v = pd.read_csv(path, usecols=cols, low_memory=False)     # 읽기
    v["month"] = pd.to_datetime(v["MthCalDt"]).dt.to_period("M")  # 날짜 → 월
    v = v[v["PERMNO"].isin(permnos)]                          # 과제 파일 종목만
    for c in cols[3:]:                                        # 숫자 열
        v[c] = pd.to_numeric(v[c], errors="coerce")           # 숫자화
    v["_pri"] = (v["TradingStatusFlg"] != "A").astype(int)    # 중복 시 정상거래(A) 행 우선
    v = v.sort_values(["PERMNO", "month", "_pri"]).drop_duplicates(["PERMNO", "month"], keep="first")  # 종목-월 중복 제거
    piv = lambda c: v.pivot(index="month", columns="PERMNO", values=c).reindex(index=months, columns=permnos)  # 넓은 형태 변환 함수
    out = {"mcap": piv("MthCap"), "ret_tot": piv("MthRet"), "retx": piv("MthRetx"),
           "mthvol": piv("MthVol"), "shrout": piv("ShrOut"), "siccd": piv("SICCD"), "prc": piv("MthPrc"),
           "status": v.pivot(index="month", columns="PERMNO", values="TradingStatusFlg").reindex(index=months, columns=permnos)}  # 특성치 묶음
    return out                                                # 반환


def load_daily(path: str, permnos, months) -> Dict[str, pd.DataFrame]:
    """일간 신호 파일(월별 집계) → 넓은 형태. t월 값은 t월 일간자료만으로 계산됨(t월 말 사용 가능)"""
    s = pd.read_csv(path)                                     # 읽기
    s["month"] = pd.PeriodIndex(s["ym"], freq="M")            # YYYY-MM → 월
    s = s[s["permno"].isin(permnos)]                          # 과제 종목만
    out = {}                                                  # 결과
    for c in ["n", "max_d", "min_d", "rvol_d", "skew_d", "ivol_d", "beta_d", "illiq_d", "dvol_m"]:  # 각 특성치
        out[c] = s.pivot(index="month", columns="permno", values=c).reindex(index=months, columns=permnos)  # 넓은 형태
    return out                                                # 반환


def load_all(ife_path: str = IFE_PATH, v2_path: Optional[str] = V2_PATH,
             daily_path: Optional[str] = DAILY_PATH, use_delist_ret: bool = True,
             delist_policy: str = "ciz") -> MarketData:
    """과제 파일(기준) + V2(시총·상폐수익률) + 일간신호를 PERMNO·월로 결합.
    - 종목 유니버스·수익률은 과제 파일(ADJ_PRC)이 기준 (보조 파일은 특성치/결측 보완용)
    - use_delist_ret=True: 보유 중 ADJ_PRC 수익률이 없으면 V2 MthRet(상장폐지 월 수익률 등)로 대체
    - delist_policy="classified": 상폐 사유 추정(delisting.py) → 일반·불명확 상폐는 상폐월 −50%,
      M&A·재분류는 CIZ 기록 수익률. 어느 파일에도 다음 수익률 흔적 없이 사라진 종목도 −50%"""
    d = load_crsp(ife_path)                                   # 과제 파일 적재(기준)
    months, permnos = d.price.index, d.price.columns          # 기준 월·종목
    if v2_path and os.path.exists(v2_path):                   # V2 파일이 있으면
        v2 = load_v2(v2_path, permnos, months)                # 적재
        d.x.update(v2)                                        # 특성치에 추가
        d.x["shell"] = d.x["shell"] | (v2["siccd"] == 6770)   # SPAC = SPAC식 이름 또는 그 달 SIC 6770(blank check)
        if use_delist_ret:                                    # 상폐수익률 사용 시
            d.fill_ret = v2["ret_tot"]                        # 대체 수익률 지정
        print("[load_all] V2 결합 완료 (시가총액·발행주식·상폐수익률)")  # 로그
    if daily_path and os.path.exists(daily_path):             # 일간신호 파일이 있으면
        d.x.update(load_daily(daily_path, permnos, months))   # 특성치에 추가
        print("[load_all] 일간신호 결합 완료")                   # 로그
    if delist_policy == "classified" and v2_path and daily_path:  # 상폐 분류 정책이면
        from delisting import classify_delistings, DELIST_RET  # 분류 모듈
        ev = classify_delistings(v2_path, daily_path)         # 이벤트 분류
        ev = ev[ev["PERMNO"].isin(permnos) & ev["m"].isin(months)]  # 과제 종목·기간
        d.delist_ret = ev.pivot_table(index="m", columns="PERMNO", values="applied_ret", aggfunc="first"
                                      ).reindex(index=months, columns=permnos)  # 상폐월 수익률 표
        seen = d.price.notna() | d.delist_ret.notna()         # 관측 여부(과제 가격 또는 상폐월)
        if d.fill_ret is not None:                            # CIZ 수익률도 관측으로 인정
            seen |= d.fill_ret.notna()                        # 결합
        d.last_month = seen[::-1].idxmax()                    # 종목별 마지막 관측월
        d.missing_exit_ret = DELIST_RET                       # 흔적 없이 사라진 종목의 가정 수익률
        d.x["delist_events"] = ev                             # 참고용 저장
        print(f"[load_all] 상폐 분류 적용: {ev['class'].value_counts().to_dict()}")  # 로그
    return d                                                 # 반환


def data_report(d: MarketData, cfg: Config = Config()) -> pd.DataFrame:
    """데이터 품질 점검용 요약 (기간, 종목수, 극단수익률, 유니버스 크기)"""
    fm = cfg.form_months()                                    # 형성월
    uni = investable_universe(d, cfg).reindex(fm)             # 형성월 유니버스
    r = d.ret.stack()                                         # 전체 수익률 1열로
    rep = {                                                   # 요약 항목 딕셔너리
        "기간": f"{d.price.index.min()} ~ {d.price.index.max()}",  # 데이터 기간
        "고유 PERMNO 수": d.price.shape[1],                    # 종목 수
        "월별 종목수(평균)": round(d.price.notna().sum(1).mean(), 1),  # 월평균 종목 수
        "월 수익률 관측치": len(r),                             # 수익률 개수
        "|r|>100% 건수": int((r.abs() > 1).sum()),              # 극단 수익률 점검
        "r<-90% 건수": int((r < -0.9).sum()),                  # 극단 하락 점검
        "유니버스 종목수(최소)": int(uni.sum(1).min()),          # 필터 후 최소 종목 수
        "유니버스 종목수(평균)": round(uni.sum(1).mean(), 1),    # 필터 후 평균 종목 수
        "FF 기간": f"{d.ff.index.min()} ~ {d.ff.index.max()}",  # FF 커버리지
    }
    return pd.Series(rep, name="값").to_frame()               # 표 형태로 반환


# ---------------------------------------------------------------------------
# 2. 유니버스 / 포트폴리오 구성
# ---------------------------------------------------------------------------
_UNI_CACHE = {}                                               # 유니버스 캐시((데이터, 유니버스 관련 설정) → 결과)


def investable_universe(d: MarketData, cfg: Config) -> pd.DataFrame:
    """형성월 t의 투자가능 종목 (같은 데이터·설정이면 캐시 재사용)"""
    key = (id(d), cfg.vol_cut, cfg.exclude_shell, cfg.size_cut, cfg.size_basis, cfg.price_min,
           cfg.min_days, cfg.dvol_cut, cfg.exclude_delist_month)  # 유니버스를 결정하는 설정만
    if key not in _UNI_CACHE:                                 # 처음이면
        _UNI_CACHE[key] = _investable_universe(d, cfg)        # 계산
    return _UNI_CACHE[key]                                    # 반환


def _investable_universe(d: MarketData, cfg: Config) -> pd.DataFrame:
    """형성월 t의 투자가능 종목: 가격·거래량 존재 + 거래량 하위 vol_cut 배제 (신호와 무관하게 공통)"""
    ok = d.price.notna() & d.vol.notna()                      # t월 가격·거래량이 있는 종목
    if cfg.exclude_shell and "shell" in d.x:                  # SPAC 제외 설정이면
        ok &= ~d.x["shell"]                                   # 그 달 SPAC으로 식별된 종목 제외(시점 정보만 사용)
    if cfg.size_cut > 0 and "mcap" in d.x:                    # 시총 필터 설정이면
        mc = d.x["mcap"].where(ok)                            # 후보 종목 시가총액
        ref = mc.where(d.exch == 1) if cfg.size_basis == "nyse" else mc  # 분위수 기준 집단(NYSE 또는 전체)
        ok &= mc.ge(ref.quantile(cfg.size_cut, axis=1), axis=0)  # 분위수 이상만 통과
    if cfg.price_min > 0 and "prc" in d.x:                    # 주가 하한 설정이면
        ok &= d.x["prc"].abs() >= cfg.price_min               # 그 달 실제 종가(조정 전) 기준으로 저가주 제외
    if cfg.exclude_delist_month and "status" in d.x:          # 상폐월 종목 제외 설정이면
        ok &= d.x["status"].ne("D")                           # 그 달 상장폐지된 종목은 매수 불가
        if d.delist_ret is not None:                          # 분류 정책이면 이벤트 기준으로도 제외
            ok &= d.delist_ret.isna()                         # (같은 달 A·D 행이 공존해 상태가 A로 남은 경우 포함)
    if cfg.min_days > 0 and "n" in d.x:                       # 최소 거래일 설정이면
        ok &= d.x["n"] >= cfg.min_days                        # 그 달 거래일 수가 부족한 종목 제외
    if cfg.dvol_cut > 0 and "dvol_m" in d.x:                  # 거래대금 필터 설정이면
        dv = d.x["dvol_m"].where(ok)                          # 후보 종목의 그 달 거래대금(달러)
        ok &= dv.ge(dv.quantile(cfg.dvol_cut, axis=1), axis=0)  # 월별 하위 비율 제외(거래대금 결측도 제외)
    v = d.vol.where(ok)                                       # 해당 종목들의 거래량만 남김
    thr = v.quantile(cfg.vol_cut, axis=1)                     # 월별 횡단면 20% 분위수
    return ok & v.ge(thr, axis=0)                             # 분위수 이상만 통과(하위 20% 배제)


def prepare_score(score: pd.DataFrame, d: MarketData, cfg: Config) -> pd.DataFrame:
    """신호를 형성월·유니버스에 맞추고 유니버스 밖 종목은 NaN 처리"""
    s = score.copy()                                          # 원본 보호
    if not isinstance(s.index, pd.PeriodIndex):               # 인덱스가 Period가 아니면
        s.index = pd.PeriodIndex(pd.to_datetime(s.index), freq="M")  # 월 Period로 변환
    fm = cfg.form_months()                                    # 형성월
    uni = investable_universe(d, cfg).reindex(index=fm, columns=d.price.columns, fill_value=False)  # 유니버스
    s = s.reindex(index=fm, columns=d.price.columns)          # 형성월·전체종목으로 정렬
    s = s.replace([np.inf, -np.inf], np.nan)                  # 무한대 제거
    return s.where(uni)                                       # 유니버스 밖은 NaN


def weigh(sel: pd.DataFrame, mcap: Optional[pd.DataFrame] = None, cap: float = 0.0,
          cap_mult: float = 0.0) -> pd.DataFrame:
    """선택 행렬(True/False) → 비중. mcap이 없으면 동일가중, 있으면 형성월 시총 비례.
    cap: 종목당 고정 상한, cap_mult: 동일가중 비중(1/n)의 배수 상한(예: 2 → 한 종목 최대 2/n). 둘 다 주면 작은 쪽"""
    if mcap is None:                                          # 동일가중
        return sel.astype(float).div(sel.sum(1).replace(0, np.nan), axis=0).fillna(0.0)  # 1/n
    m = mcap.reindex(index=sel.index, columns=sel.columns).abs()  # 형성월 시가총액
    raw = m.where(sel).fillna(0.0)                            # 선택 종목 시총(결측은 0 → 제외)
    w = raw.div(raw.sum(1).replace(0, np.nan), axis=0).fillna(0.0)  # 시총 비례
    n = (raw > 0).sum(1).replace(0, np.nan)                   # 월별 보유 종목 수
    caps = pd.Series(np.inf, index=w.index)                   # 월별 상한
    if cap > 0:                                               # 고정 상한
        caps = caps.clip(upper=cap)                           # 적용
    if cap_mult > 0:                                          # 동일가중 배수 상한
        caps = np.minimum(caps, (cap_mult / n).fillna(np.inf))  # 적용
    if np.isfinite(caps).any():                               # 상한이 있으면
        for _ in range(200):                                  # 절단·재배분 반복
            excess = w.sub(caps, axis=0).clip(lower=0).sum(1)  # 상한 초과분
            if excess.max() < 1e-10:                          # 초과가 없으면
                break                                         # 종료
            w = w.clip(upper=caps, axis=0)                    # 절단
            free = w.where((w > 0) & w.lt(caps - 1e-12, axis=0), 0.0)  # 재배분 받을 종목(상한 미만)
            w = w + free.div(free.sum(1).replace(0, np.nan), axis=0).fillna(0.0).mul(excess, axis=0)  # 비례 재배분
    return w                                                  # 반환


def _mcap_for(d: "MarketData", cfg: Config) -> Optional[pd.DataFrame]:
    """설정이 시총가중이면 형성월 시가총액, 아니면 None"""
    return d.x["mcap"] if cfg.weighting == "vw" else None     # 선택


def weights_topk(s: pd.DataFrame, k: int, mcap: Optional[pd.DataFrame] = None, cap: float = 0.0,
                 cap_mult: float = 0.0) -> pd.DataFrame:
    """점수 상위 k 종목 비중 (동점은 PERMNO 순서로 결정 → 재현 가능). mcap을 주면 시총가중"""
    rank = s.rank(axis=1, ascending=False, method="first")    # 높은 점수 = 1등
    sel = rank <= k                                           # 상위 k 선택
    n = sel.sum(axis=1)                                       # 월별 실제 선택 수
    if (n < k).any():                                         # 유효 종목이 k보다 적은 달이 있으면
        warnings.warn(f"Top-{k}: {int((n < k).sum())}개월은 유효 종목이 {k}개 미만")  # 경고
    return weigh(sel, mcap, cap, cap_mult)                    # 비중


def weights_groups(s: pd.DataFrame, n_groups: int, mcap: Optional[pd.DataFrame] = None,
                   cap_mult: float = 0.0) -> Dict[int, pd.DataFrame]:
    """점수 기준 n분위(1=최상위) 비중 (mcap을 주면 시총가중)"""
    pct = s.rank(axis=1, ascending=False, method="first", pct=True)  # 백분위 순위(상위일수록 작음)
    grp = np.ceil(pct * n_groups)                             # 1..n 그룹 번호
    out = {}                                                  # 결과 딕셔너리
    for g in range(1, n_groups + 1):                          # 각 그룹마다
        out[g] = weigh(grp == g, mcap, 0.0, cap_mult)         # 동일/시총가중(배수 상한 가능)
    return out                                                # 반환


def weights_ew_universe(d: MarketData, cfg: Config) -> pd.DataFrame:
    """벤치마크: 유니버스 전체 동일가중 (다음달 수익률 존재 여부와 무관)"""
    fm = cfg.form_months()                                    # 형성월
    uni = investable_universe(d, cfg).reindex(index=fm, fill_value=False).astype(float)  # 유니버스 0/1
    return uni.div(uni.sum(1), axis=0).fillna(0.0)            # 1/N 비중


def portfolio_returns(w: pd.DataFrame, d: MarketData, cfg: Config) -> pd.DataFrame:
    """비중(형성월 기준) → 보유월 수익률/회전율/비용 시리즈. 길이 215 검증 포함."""
    fwd = d.ret.shift(-1).reindex(index=w.index, columns=w.columns)  # t행에 t+1 수익률 배치
    held = w > 0                                              # 보유 여부
    miss = (held & fwd.isna()).sum(1)                         # 보유 중 다음달 ADJ_PRC 수익률 결측(상폐 등) 개수
    if d.fill_ret is not None:                                # 대체 수익률(V2 상장폐지 수익률 등)이 있으면
        alt = d.fill_ret.shift(-1).reindex(index=w.index, columns=w.columns)  # t+1 대체 수익률
        fwd = fwd.fillna(alt)                                 # ADJ_PRC 수익률이 없을 때만 대체
    if d.delist_ret is not None:                              # 상폐 분류 정책이면
        dr = d.delist_ret.shift(-1).reindex(index=w.index, columns=w.columns)  # t+1이 상폐월인 종목의 정책 수익률
        fwd = fwd.mask(dr.notna(), dr)                        # 상폐월 수익률은 정책 값으로 덮어씀(벡터화)
        gone = held & fwd.isna() & _after_last(w.index, d.last_month.reindex(w.columns))  # t+1 이후 흔적 없이 사라짐
        fwd = fwd.mask(gone, d.missing_exit_ret)              # 퇴출 가정 수익률(−50%)
    gross = (w * fwd.fillna(cfg.missing_ret_fill)).sum(1)     # 총수익률(결측은 가정값 적용)
    turnover = 0.5 * w.diff().abs().sum(1)                    # 명목 편도 회전율 ½Σ|Δw|
    turnover.iloc[0] = 0.5 * w.iloc[0].abs().sum()            # 첫 달: 현금에서 최초 매수(=0.5)
    cost = turnover * cfg.cost_rt                             # 거래비용 = 회전율 × 왕복 20bp
    out = pd.DataFrame({"gross": gross, "turnover": turnover, "cost": cost,
                        "net": gross - cost, "n_hold": held.sum(1), "n_missing": miss})  # 결과 표
    out.index = out.index + 1                                 # 형성월 t → 보유월 t+1 로 라벨 이동
    _check_length(out, cfg)                                   # 215개월 검증
    return out                                                # 반환


def _after_last(idx: pd.PeriodIndex, lastm: pd.Series) -> pd.DataFrame:
    """(형성월 t, 종목)마다 t+1이 그 종목의 마지막 관측월보다 뒤인지"""
    t1 = np.array([(p + 1).ordinal for p in idx])             # t+1 서수
    lm = np.array([p.ordinal if isinstance(p, pd.Period) else np.iinfo(np.int64).max for p in lastm.values])  # 마지막 관측월 서수
    return pd.DataFrame(t1[:, None] > lm[None, :], index=idx, columns=lastm.index)  # 비교


def _check_length(x, cfg: Config):
    """최종 수익률 시리즈가 2003-02~2020-12, 215개인지 확인 (아니면 중단)"""
    ok = (len(x) == cfg.n_months and str(x.index[0]) == cfg.start
          and str(x.index[-1]) == cfg.end)                    # 길이·시작·끝 모두 확인
    if not ok:                                                # 불일치하면
        raise ValueError(f"수익률 시리즈 검증 실패: 길이 {len(x)} ({x.index[0]}~{x.index[-1]}), "
                         f"기대 {cfg.n_months} ({cfg.start}~{cfg.end})")  # 에러로 중단


# ---------------------------------------------------------------------------
# 3. 성과 지표
# ---------------------------------------------------------------------------
def ann_return(r: pd.Series) -> float:
    """연수익 = (1 + 월평균)^12 − 1"""
    return (1 + r.mean()) ** 12 - 1                           # 산술평균 복리 환산


def ann_vol(r: pd.Series) -> float:
    """연변동성 = 월 표준편차(ddof=1) × √12"""
    return r.std(ddof=1) * np.sqrt(12)                        # 연율화


def sharpe(r: pd.Series, rf: pd.Series) -> float:
    """Sharpe = 초과수익 평균 / 초과수익 표준편차(ddof=1) × √12"""
    ex = r - rf.reindex(r.index)                              # 월 초과수익
    return ex.mean() / ex.std(ddof=1) * np.sqrt(12)           # 연율화 Sharpe


def max_drawdown(r: pd.Series) -> float:
    """최대낙폭: 초기자산 1 포함 누적가치의 고점 대비 최대 하락률"""
    wealth = (1 + r).cumprod()                                # 누적 가치
    peak = np.maximum.accumulate(np.r_[1.0, wealth.values])[1:]  # 시작점(1) 포함 누적 고점
    return (wealth.values / peak - 1).min()                   # 최저 낙폭(음수)


def nw_mean_test(x: pd.Series, lags: int):
    """평균이 0인지 Newey-West t검정 → (평균, t, p)"""
    import statsmodels.api as sm

    x = x.dropna()                                            # 결측 제거
    res = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})  # 상수항 회귀
    return res.params[0], res.tvalues[0], res.pvalues[0]      # 평균, t값, p값


def ff3_regression(r: pd.Series, ff: pd.DataFrame, lags: int) -> dict:
    """FF3 회귀: (r − rf) = α + β·MKT + s·SMB + h·HML  (HAC 표준오차)"""
    import statsmodels.api as sm

    f = ff.reindex(r.index)                                   # 기간 맞추기
    ex = r - f["RF"]                                          # 초과수익
    X = sm.add_constant(f[["Mkt-RF", "SMB", "HML"]])          # 설명변수 + 상수
    res = sm.OLS(ex, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lags})  # 회귀
    return {"beta": res.params["Mkt-RF"], "smb": res.params["SMB"], "hml": res.params["HML"],
            "alpha_ann": res.params["const"] * 12, "alpha_t": res.tvalues["const"],
            "r2": res.rsquared}                               # 베타/알파(연율, 단순×12)/t/R²


def sharpe_diff_test(ra: pd.Series, rb: pd.Series, rf: pd.Series, lags: int) -> dict:
    """ΔSR = SR(a) − SR(b) 유의성: Ledoit–Wolf(2008) HAC 델타법 + Memmel(2003)"""
    idx = ra.index.intersection(rb.index)                     # 공통 기간
    u = (ra - rf).reindex(idx).values                         # a 초과수익
    v = (rb - rf).reindex(idx).values                         # b 초과수익
    T = len(u)                                                # 표본 수
    mu_a, mu_b = u.mean(), v.mean()                           # 평균
    g_a, g_b = (u ** 2).mean(), (v ** 2).mean()               # 2차 적률
    va, vb = g_a - mu_a ** 2, g_b - mu_b ** 2                 # 분산(ddof=0, 델타법 정의)
    d_hat = mu_a / np.sqrt(va) - mu_b / np.sqrt(vb)           # 월 ΔSR (델타법용)
    grad = np.array([g_a / va ** 1.5, -g_b / vb ** 1.5,
                     -0.5 * mu_a / va ** 1.5, 0.5 * mu_b / vb ** 1.5])  # ΔSR의 (μa,μb,γa,γb) 기울기
    Y = np.column_stack([u - mu_a, v - mu_b, u ** 2 - g_a, v ** 2 - g_b])  # 적률 조건 편차
    psi = Y.T @ Y / T                                         # 0차 자기공분산
    for j in range(1, lags + 1):                              # Bartlett 가중 HAC
        G = Y[j:].T @ Y[:-j] / T                              # j차 자기공분산
        psi += (1 - j / (lags + 1)) * (G + G.T)               # 가중 합산
    se_lw = np.sqrt(grad @ psi @ grad / T)                    # ΔSR 표준오차
    p_lw = 2 * (1 - stats.norm.cdf(abs(d_hat / se_lw)))       # 양측 p값
    sa, sb = u.mean() / u.std(ddof=1), v.mean() / v.std(ddof=1)  # 월 Sharpe(ddof=1)
    rho = np.corrcoef(u, v)[0, 1]                             # 상관계수
    var_m = (2 - 2 * rho + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / T  # Memmel 분산
    p_m = 2 * (1 - stats.norm.cdf(abs((sa - sb) / np.sqrt(var_m))))  # Memmel p값
    return {"dSR": (sa - sb) * np.sqrt(12), "p_LW": p_lw, "p_Memmel": p_m}  # 연율 ΔSR, p값들


def perf_row(r: pd.Series, ff: pd.DataFrame, cfg: Config, bench: Dict[str, pd.Series],
             turnover: Optional[pd.Series] = None, n_hold: Optional[pd.Series] = None) -> dict:
    """한 포트폴리오의 지표 한 줄 (bench: {이름: 수익률} 대비 ΔSR 계산)"""
    L = cfg.lags(len(r))                                      # NW 시차
    rf = ff["RF"].reindex(r.index)                            # 무위험수익률
    reg = ff3_regression(r, ff, L)                            # FF3 회귀
    row = {"연수익": ann_return(r), "변동성": ann_vol(r), "Sharpe": sharpe(r, rf),
           "MDD": max_drawdown(r)}                            # 기본 지표
    for name, rb in bench.items():                            # 벤치마크별
        if r.equals(rb):                                      # 자기 자신과의 비교는
            t = {"dSR": np.nan, "p_LW": np.nan, "p_Memmel": np.nan}  # 의미 없으므로 공란
        else:                                                 # 다른 포트폴리오면
            t = sharpe_diff_test(r, rb, rf, L)                # ΔSR 검정
        row[f"ΔSR vs {name}"] = t["dSR"]                      # ΔSR
        row[f"p(LW) vs {name}"] = t["p_LW"]                   # LW p값
        row[f"p(Memmel) vs {name}"] = t["p_Memmel"]           # Memmel p값
    row["베타(FF3)"] = reg["beta"]                            # 시장 베타
    row["회전율(월)"] = turnover.mean() if turnover is not None else np.nan  # 평균 월 편도 회전율
    row["α(FF3,연)"] = reg["alpha_ann"]                       # 연율 알파
    row["α t"] = reg["alpha_t"]                               # 알파 t값
    row["SMB"] = reg["smb"]                                   # 사이즈 노출
    row["HML"] = reg["hml"]                                   # 가치 노출
    row["누적수익"] = (1 + r).prod() - 1                       # 전체기간 누적수익
    row["평균종목수"] = n_hold.mean() if n_hold is not None else np.nan  # 평균 보유 종목 수
    return row                                                # 반환


# ---------------------------------------------------------------------------
# 4. 분석 1 : Top-k vs 시장
# ---------------------------------------------------------------------------
def benchmarks(d: MarketData, cfg: Config) -> Dict[str, pd.DataFrame]:
    """벤치마크 수익률: MKT(FF Mkt-RF+RF, CRSP 가치가중 시장), EW(유니버스 동일가중, 비용 차감)"""
    months = pd.period_range(cfg.start, cfg.end, freq="M")   # 보유월
    mkt = d.ff["MKT"].reindex(months)                         # 시장 수익률
    if mkt.isna().any():                                      # FF 기간이 부족하면
        raise ValueError("FF 데이터가 보유기간을 모두 포함하지 않음")  # 중단
    _check_length(mkt, cfg)                                   # 215개월 검증
    ew = portfolio_returns(weights_ew_universe(d, cfg), d, cfg)  # 동일가중 유니버스
    return {"MKT": pd.DataFrame({"net": mkt}), "EW": ew}      # 반환


def evaluate_topk(score: pd.DataFrame, d: MarketData, cfg: Config = Config(),
                  use_net: bool = True):
    """분석 1: Top-k 성과표와 월별 수익률 반환 (use_net=True면 비용 차감 후)"""
    s = prepare_score(score, d, cfg)                          # 신호 정렬/유니버스 적용
    bm = benchmarks(d, cfg)                                   # 벤치마크
    col = "net" if use_net else "gross"                       # 사용할 수익률 종류
    bench_r = {"MKT": bm["MKT"]["net"], "EW": bm["EW"][col]}  # ΔSR 비교 대상
    rows, rets, details = {}, {}, {}                          # 결과 저장용
    for k in cfg.top_ks:                                      # k마다
        pr = portfolio_returns(weights_topk(s, k, _mcap_for(d, cfg), cfg.vw_cap, cfg.vw_cap_mult), d, cfg)  # 포트폴리오 수익률
        rows[f"Top-{k}"] = perf_row(pr[col], d.ff, cfg, bench_r, pr["turnover"], pr["n_hold"])  # 지표
        rets[f"Top-{k}"] = pr[col]                            # 수익률 저장
        details[f"Top-{k}"] = pr                              # 상세(회전율/비용 등) 저장
    rows["EW 유니버스"] = perf_row(bm["EW"][col], d.ff, cfg, bench_r, bm["EW"]["turnover"], bm["EW"]["n_hold"])  # EW 벤치
    rows["MKT (시장)"] = perf_row(bm["MKT"]["net"], d.ff, cfg, bench_r)  # 시장 벤치
    rets["EW 유니버스"] = bm["EW"][col]                        # EW 수익률
    rets["MKT (시장)"] = bm["MKT"]["net"]                      # 시장 수익률
    table = pd.DataFrame(rows).T                              # 표로 변환
    return table, pd.DataFrame(rets), details                 # 성과표, 월수익률, 상세


# ---------------------------------------------------------------------------
# 5. 분석 2 : 10분위
# ---------------------------------------------------------------------------
def _stationary_bootstrap_idx(T: int, block: float, rng) -> np.ndarray:
    """Politis–Romano 정상 부트스트랩 인덱스 (평균 블록 길이 block)"""
    idx = np.empty(T, dtype=int)                              # 결과 배열
    idx[0] = rng.integers(T)                                  # 첫 위치 무작위
    for t in range(1, T):                                     # 이후 위치
        idx[t] = rng.integers(T) if rng.random() < 1 / block else (idx[t - 1] + 1) % T  # 새 블록 or 이어가기
    return idx                                                # 반환


def monotonic_test(R: pd.DataFrame, cfg: Config, decreasing: bool = True) -> dict:
    """Patton–Timmermann(2010) MR 검정. H1: D1 > D2 > … > Dn (decreasing=True)"""
    X = R.values if decreasing else R.values[:, ::-1]         # 방향에 맞게 열 순서
    diffs = X[:, :-1] - X[:, 1:]                              # 인접 그룹 차이(월별)
    dbar = diffs.mean(0)                                      # 평균 차이
    J = dbar.min()                                            # 검정통계량: 최소 인접 차이
    rng = np.random.default_rng(cfg.seed)                     # 난수 생성기
    T = len(diffs)                                            # 표본 길이
    Jb = np.empty(cfg.n_boot)                                 # 부트스트랩 통계량
    for b in range(cfg.n_boot):                               # 반복
        ib = _stationary_bootstrap_idx(T, 6.0, rng)           # 재표본 인덱스(평균 블록 6개월)
        Jb[b] = (diffs[ib].mean(0) - dbar).min()              # 귀무가설 하 중심화된 최소값
    return {"J(최소 인접차)": J, "p": (Jb >= J).mean()}         # 통계량과 p값


def evaluate_deciles(score: pd.DataFrame, d: MarketData, cfg: Config = Config(),
                     use_net: bool = False) -> dict:
    """분석 2: 10분위 분석 (기본은 비용 전 gross; use_net=True면 비용 차감)"""
    s = prepare_score(score, d, cfg)                          # 신호 정렬
    W = weights_groups(s, cfg.n_groups, _mcap_for(d, cfg), cfg.vw_cap_mult)  # 분위별 비중
    col = "net" if use_net else "gross"                       # 수익률 종류
    ff = d.ff                                                 # 팩터
    pr = {g: portfolio_returns(w, d, cfg) for g, w in W.items()}  # 분위별 수익률 계산
    R = pd.DataFrame({f"D{g}": pr[g][col] for g in pr})       # 분위 수익률 행렬(T×10)
    L = cfg.lags(len(R))                                      # NW 시차
    rf = ff["RF"].reindex(R.index)                            # 무위험
    rows = {}                                                 # 분위별 지표
    for g in pr:                                              # 각 분위
        r = R[f"D{g}"]                                        # 수익률
        m, t, p = nw_mean_test(r - rf, L)                     # 초과수익 평균 유의성
        reg = ff3_regression(r, ff, L)                        # FF3
        rows[f"D{g}"] = {"월평균": r.mean(), "초과 t(NW)": t, "연수익": ann_return(r),
                         "변동성": ann_vol(r), "Sharpe": sharpe(r, rf), "MDD": max_drawdown(r),
                         "베타(FF3)": reg["beta"], "α(FF3,연)": reg["alpha_ann"], "α t": reg["alpha_t"],
                         "회전율(월)": pr[g]["turnover"].mean(), "평균종목수": pr[g]["n_hold"].mean()}  # 지표
    table = pd.DataFrame(rows).T                              # 분위 성과표
    top, bot = R.iloc[:, 0], R.iloc[:, -1]                    # D1, D10
    spread = top - bot                                        # 롱숏 스프레드
    m, t, p = nw_mean_test(spread, L)                         # 스프레드 유의성
    reg = ff3_regression(spread + rf, ff, L)                  # 스프레드 FF3(무위험 더해 초과수익=스프레드로 만듦)
    spread_stats = pd.Series({"월평균": m, "t(NW)": t, "p": p, "연율(×12)": m * 12,
                              "Sharpe(스프레드)": spread.mean() / spread.std(ddof=1) * np.sqrt(12),
                              "α(FF3,연)": reg["alpha_ann"], "α t": reg["alpha_t"],
                              "베타(FF3)": reg["beta"], "승률(>0)": (spread > 0).mean(),
                              "MDD": max_drawdown(spread)}, name="D1−D10")  # 스프레드 요약
    n = R.shape[1]                                            # 그룹 수
    diff_mean = pd.DataFrame(np.nan, index=R.columns, columns=R.columns)  # 쌍별 평균차 행렬
    diff_t = diff_mean.copy()                                 # 쌍별 t값 행렬
    for i in range(n):                                        # 행 그룹
        for j in range(n):                                    # 열 그룹
            if i != j:                                        # 자기 자신 제외
                mm, tt, _ = nw_mean_test(R.iloc[:, i] - R.iloc[:, j], L)  # 차이 검정
                diff_mean.iloc[i, j], diff_t.iloc[i, j] = mm, tt  # 저장
    adj = pd.DataFrame([dict(zip(["월평균차", "t(NW)", "p"],
                                 nw_mean_test(R.iloc[:, i] - R.iloc[:, i + 1], L)))
                        for i in range(n - 1)],
                       index=[f"D{i+1}−D{i+2}" for i in range(n - 1)])  # 인접 그룹 차이 검정
    D = (R.iloc[:, :-1].values - R.iloc[:, [-1]].values)      # Di − D10 (T×9)
    dbar = D.mean(0)                                          # 평균 차이 벡터
    Dc = D - dbar                                             # 중심화
    T = len(D)                                                # 표본 수
    S = Dc.T @ Dc / T                                         # HAC 공분산 0차
    for j in range(1, L + 1):                                 # Bartlett 가중
        G = Dc[j:].T @ Dc[:-j] / T                            # j차
        S += (1 - j / (L + 1)) * (G + G.T)                    # 합산
    W_stat = T * dbar @ np.linalg.solve(S, dbar)              # Wald 통계량
    joint = {"Wald χ²": W_stat, "df": n - 1, "p": 1 - stats.chi2.cdf(W_stat, n - 1)}  # 전 분위 평균 동일성 결합검정
    rho, rho_p = stats.spearmanr(np.arange(n, 0, -1), R.mean().values)  # 신호순위(D1=최고)-평균수익 순위상관(양수=신호 유효)
    mono = {"Spearman ρ(분위, 평균)": rho, "ρ p": rho_p,
            "MR 검정 p (D1>…>D10)": monotonic_test(R, cfg, True)["p"],
            "MR 검정 p (D1<…<D10)": monotonic_test(R, cfg, False)["p"]}  # 단조성 결과
    return {"table": table, "returns": R, "spread": spread, "spread_stats": spread_stats,
            "pair_mean": diff_mean, "pair_t": diff_t, "adjacent": adj,
            "joint": pd.Series(joint), "monotonic": pd.Series(mono)}  # 결과 묶음


# ---------------------------------------------------------------------------
# 6. 출력/그림
# ---------------------------------------------------------------------------
MAIN_COLS = ["연수익", "변동성", "Sharpe", "MDD", "ΔSR vs MKT", "p(LW) vs MKT", "베타(FF3)", "회전율(월)"]  # 요청 지표
EXTRA_COLS = ["ΔSR vs EW", "p(LW) vs EW", "α(FF3,연)", "α t", "SMB", "HML", "평균종목수"]  # 추가 지표
PCT_COLS = {"연수익", "변동성", "MDD", "회전율(월)", "α(FF3,연)", "누적수익", "월평균", "연율(×12)", "승률(>0)"}  # % 표시 열


def fmt_table(t: pd.DataFrame) -> pd.DataFrame:
    """표를 보기 좋게 문자열로 변환 (%/소수 자리 통일)"""
    out = pd.DataFrame(index=t.index)                         # 결과 표
    for c in t.columns:                                       # 열마다
        if c in PCT_COLS:                                     # 백분율 열
            out[c] = t[c].map(lambda x: "" if pd.isna(x) else f"{x*100:.2f}%")  # % 2자리
        elif c.startswith("p"):                               # p값 열
            out[c] = t[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")  # 소수 3자리
        else:                                                 # 나머지
            out[c] = t[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")  # 소수 3자리
    return out                                                # 반환


def summary_topk(table: pd.DataFrame) -> pd.DataFrame:
    """요청 형식 표: ΔSR과 p값을 'ΔSR (p)'로 합침"""
    t = table.copy()                                          # 복사
    t["ΔSR (p)"] = [("" if pd.isna(a) or abs(a) < 1e-12 else f"{a:+.3f} ({p:.3f})")
                    for a, p in zip(t["ΔSR vs MKT"], t["p(LW) vs MKT"])]  # 시장 대비 ΔSR(p)
    t["ΔSR vs EW (p)"] = [("" if pd.isna(a) or abs(a) < 1e-12 else f"{a:+.3f} ({p:.3f})")
                          for a, p in zip(t["ΔSR vs EW"], t["p(LW) vs EW"])]  # EW 대비
    f = fmt_table(table)                                      # 숫자 포맷
    cols = ["연수익", "변동성", "Sharpe", "MDD"]                # 앞 열
    return pd.concat([f[cols], t[["ΔSR (p)"]], f[["베타(FF3)", "회전율(월)"]],
                      t[["ΔSR vs EW (p)"]], f[["α(FF3,연)", "α t", "평균종목수"]]], axis=1)  # 최종 표


def plot_topk(rets: pd.DataFrame, title: str, path: str):
    """Top-k 및 벤치마크 누적가치(로그축) 그림 저장"""
    import matplotlib.pyplot as plt                           # 그림 라이브러리
    plt.rcParams["font.family"] = "Malgun Gothic"             # 한글 폰트(윈도우)
    plt.rcParams["axes.unicode_minus"] = False                # 마이너스 기호 깨짐 방지
    fig, ax = plt.subplots(figsize=(10, 5.5))                 # 캔버스
    wealth = (1 + rets).cumprod()                             # 누적가치
    wealth.loc[wealth.index[0] - 1] = 1.0                     # 시작점(2003-01 = 1) 추가
    wealth = wealth.sort_index()                              # 시간순 정렬
    x = wealth.index.to_timestamp()                           # x축 날짜
    for c in wealth.columns:                                  # 각 시리즈
        bench = c.startswith(("MKT", "EW"))                   # 벤치마크 여부
        ax.plot(x, wealth[c], lw=2.2 if bench else 1.4,
                ls="--" if bench else "-", color="black" if c.startswith("MKT") else None,
                label=f"{c}  ({wealth[c].iloc[-1]:.2f}x)")    # 선 그리기(범례에 최종배수)
    ax.set_yscale("log")                                      # 로그축
    ax.set_title(title)                                       # 제목
    ax.set_ylabel("포트폴리오 가치 (시작=1, 로그)")              # y축 라벨
    ax.grid(alpha=0.3)                                        # 격자
    ax.legend(frameon=False, fontsize=9)                      # 범례
    fig.tight_layout()                                        # 여백 정리
    fig.savefig(path, dpi=150)                                # 저장
    plt.close(fig)                                            # 메모리 해제


def plot_deciles(res: dict, title: str, path: str):
    """분위별 연수익 막대 + D1/D10/스프레드 누적가치 그림 저장"""
    import matplotlib.pyplot as plt                           # 그림 라이브러리
    plt.rcParams["font.family"] = "Malgun Gothic"             # 한글 폰트
    plt.rcParams["axes.unicode_minus"] = False                # 마이너스 기호
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))     # 1×2 캔버스
    t = res["table"]                                          # 분위 표
    a1.bar(t.index, t["연수익"] * 100, color="#4C72B0")        # 연수익 막대
    for i, v in enumerate(t["초과 t(NW)"]):                    # 각 막대 위에
        a1.text(i, t["연수익"].iloc[i] * 100, f"t={v:.1f}", ha="center",
                va="bottom" if t["연수익"].iloc[i] >= 0 else "top", fontsize=8)  # 초과수익 t값 표기
    a1.axhline(0, color="gray", lw=0.8)                       # 0선
    a1.set_ylabel("연수익 (%)")                                # y축
    a1.set_title("분위별 연수익 (D1=최상위)")                    # 제목
    R = res["returns"]                                        # 분위 수익률
    lines = {"D1": R.iloc[:, 0], f"D{R.shape[1]}": R.iloc[:, -1], "D1-D10": res["spread"]}  # 그릴 시리즈(그림은 ASCII 하이픈)
    for name, r in lines.items():                             # 각 시리즈
        w = (1 + r).cumprod()                                 # 누적가치
        a2.plot(w.index.to_timestamp(), w, label=f"{name} ({w.iloc[-1]:.2f}x)")  # 선
    a2.set_yscale("log")                                      # 로그축
    a2.set_title("누적가치 (로그)")                              # 제목
    a2.grid(alpha=0.3)                                        # 격자
    a2.legend(frameon=False, fontsize=9)                      # 범례
    fig.suptitle(title)                                       # 전체 제목
    fig.tight_layout()                                        # 여백
    fig.savefig(path, dpi=150)                                # 저장
    plt.close(fig)                                            # 해제


def run_signal(name: str, score: pd.DataFrame, d: MarketData, cfg: Config = Config(),
               out_dir: Optional[str] = None, verbose: bool = True) -> dict:
    """한 신호에 대해 분석 1·2를 모두 실행하고 결과(표·그림)를 저장"""
    table, rets, det = evaluate_topk(score, d, cfg)           # 분석 1
    dec = evaluate_deciles(score, d, cfg)                     # 분석 2
    if out_dir:                                               # 저장 경로가 있으면
        od = os.path.join(out_dir, name)                      # 신호별 폴더
        os.makedirs(od, exist_ok=True)                        # 폴더 생성
        table.to_csv(os.path.join(od, "topk_metrics.csv"), encoding="utf-8-sig")  # 성과표(원자료)
        summary_topk(table).to_csv(os.path.join(od, "topk_summary.csv"), encoding="utf-8-sig")  # 요약표
        rets.to_csv(os.path.join(od, "topk_returns.csv"), encoding="utf-8-sig")  # 월수익률
        dec["table"].to_csv(os.path.join(od, "decile_metrics.csv"), encoding="utf-8-sig")  # 분위표
        dec["returns"].to_csv(os.path.join(od, "decile_returns.csv"), encoding="utf-8-sig")  # 분위 수익률
        dec["pair_t"].to_csv(os.path.join(od, "decile_pair_t.csv"), encoding="utf-8-sig")  # 쌍별 t
        pd.concat([dec["spread_stats"], dec["joint"], dec["monotonic"]]).to_csv(
            os.path.join(od, "decile_tests.csv"), encoding="utf-8-sig")  # 검정 결과
        plot_topk(rets, f"[{name}] Top-k vs 시장 (비용 차감 후)", os.path.join(od, "topk_wealth.png"))  # 그림1
        plot_deciles(dec, f"[{name}] 10분위 분석 (비용 전)", os.path.join(od, "deciles.png"))  # 그림2
    if verbose:                                               # 화면 출력
        with pd.option_context("display.width", 250, "display.max_columns", 30):  # 넓게 출력
            print(f"\n{'=' * 100}\n신호: {name}\n{'=' * 100}")  # 머리말
            print("[분석 1] Top-k vs 시장 (비용 차감 후, ΔSR p값 = Ledoit–Wolf HAC)")  # 제목
            print(summary_topk(table))                        # 요약표
            print("\n[분석 2] 10분위 (비용 전)")                  # 제목
            print(fmt_table(dec["table"][["월평균", "초과 t(NW)", "연수익", "변동성", "Sharpe",
                                          "MDD", "베타(FF3)", "α(FF3,연)", "α t", "회전율(월)"]]))  # 분위표
            print("\nD1−D10 스프레드:")                        # 스프레드
            print(fmt_table(dec["spread_stats"].to_frame().T).T)  # 스프레드 표
            print("\n인접 분위 차이:")                          # 인접 차이
            print(fmt_table(dec["adjacent"]))                 # 표
            print("\n결합검정(모든 분위 평균 동일):", dec["joint"].round(4).to_dict())  # Wald
            print("단조성:", dec["monotonic"].round(4).to_dict())  # 단조성
    return {"topk": table, "topk_returns": rets, "topk_details": det, "deciles": dec}  # 결과 반환


def leaderboard(results: Dict[str, dict], k_focus=(20, 50)) -> pd.DataFrame:
    """여러 신호 결과를 한 표로 비교 (종목선택 기준 결정용)"""
    rows = {}                                                 # 행 모음
    for name, res in results.items():                         # 신호별
        t, dec = res["topk"], res["deciles"]                  # 결과 꺼내기
        row = {}                                              # 한 행
        for k in k_focus:                                     # 관심 k마다
            row[f"Top{k} 연수익"] = t.loc[f"Top-{k}", "연수익"]   # 연수익(절대수익 기준 시장 초과 여부)
            row[f"Top{k} Sharpe"] = t.loc[f"Top-{k}", "Sharpe"]  # Sharpe
            row[f"Top{k} ΔSR"] = t.loc[f"Top-{k}", "ΔSR vs MKT"]  # 시장 대비 ΔSR
            row[f"Top{k} p"] = t.loc[f"Top-{k}", "p(LW) vs MKT"]  # p값
            row[f"Top{k} α t"] = t.loc[f"Top-{k}", "α t"]       # FF3 알파 t
            row[f"Top{k} 회전율"] = t.loc[f"Top-{k}", "회전율(월)"]  # 회전율
        row["D1−D10 월평균"] = dec["spread_stats"]["월평균"]      # 스프레드 크기
        row["D1−D10 t"] = dec["spread_stats"]["t(NW)"]          # 스프레드 t
        row["D1−D10 α t"] = dec["spread_stats"]["α t"]          # 스프레드 FF3 알파 t (위험조정 격차)
        row["D1 α t"] = dec["table"].loc["D1", "α t"]            # 최상위 분위 알파 t (롱온리 관점 핵심)
        row["Spearman ρ"] = dec["monotonic"]["Spearman ρ(분위, 평균)"]  # 단조성
        row["MR p(D1>…)"] = dec["monotonic"]["MR 검정 p (D1>…>D10)"]  # MR p
        rows[name] = row                                      # 저장
    return pd.DataFrame(rows).T                               # 표로 반환
