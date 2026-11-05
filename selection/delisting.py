"""
상장폐지 사유 분류 (CRSP CIZ 파일에 상폐 코드가 없어 직전 정보로 추정)
=====================================================================
[분류 규칙] (상폐월 = CIZ TradingStatusFlg 'D' 행의 월, 직전월 = 마지막 정상거래(A) 월)
  0) 현금 인수 우선: 변동성 축소비 < 0.5 이고 상폐월 수익 ≥ −10%, 12개월 로그수익 ≥ −50%면 주가와 무관하게 인수합병
  1) 일반 상폐(perf)  : 직전월 주가 < $5, 또는 직전 12개월 로그수익 < −50%, 또는 상폐월 수익 < −30%
  2) 증권 재분류(reclass): 같은 PERMCO(회사)가 상폐월 이후에도 다른 PERMNO로 거래됨 (주식 종류 전환 등)
  3) 인수합병(merger) : (a) 현금 인수형 — 직전월 일간 변동성이 평소(3~14개월 전 평균)의 절반 미만(인수가에 가격 고정),
                        또는 최근 6개월 내 +20% 이상 급등(인수 발표 프리미엄) 후 변동성이 평소의 75% 미만
                        (b) 주식교환 합병·자진상폐형 — 직전 주가 ≥ $10, 12개월 로그수익 ≥ −25%, 상폐월 수익 ≥ −10%
                            (부실 징후가 없는 종목; 1차 규칙에서 '불명확'으로 남은 대형주가 Warner-Lambert, BellSouth,
                             Dow, DuPont, Aetna 등 실제 합병 사례였음을 확인하고 추가)
  4) 불명확(unclear)  : 위 어디에도 해당하지 않음
[수익률 처리] 일반 상폐·불명확 → 상폐월 수익률 −50%, 인수합병·재분류 → CIZ 기록 수익률 그대로
"""
import numpy as np                           # 수치
import pandas as pd                          # 데이터프레임

DELIST_RET = -0.50                           # 일반 상폐·불명확 상폐의 상폐월 수익률 가정


def classify_delistings(v2_path: str, daily_path: str) -> pd.DataFrame:
    """상폐 이벤트별 특성과 분류 결과 (PERMNO, 상폐월, class, 적용 수익률)"""
    v = pd.read_csv(v2_path, usecols=["PERMNO", "PERMCO", "TradingStatusFlg", "MthCalDt", "MthPrc", "MthRet"])  # 필요한 열
    v["m"] = pd.to_datetime(v["MthCalDt"]).dt.to_period("M")  # 월
    v = v.drop_duplicates(["PERMNO", "m", "TradingStatusFlg"])  # 완전 중복 제거
    D = (v[v["TradingStatusFlg"] == "D"].drop_duplicates(["PERMNO", "m"])
         [["PERMNO", "PERMCO", "m", "MthRet"]].rename(columns={"MthRet": "dret"}))  # 상폐월 행
    A = v[v["TradingStatusFlg"] == "A"].sort_values(["PERMNO", "m"]).drop_duplicates(["PERMNO", "m"])  # 정상거래 행
    A["lr"] = np.log1p(A["MthRet"])                          # 로그수익
    g = A.groupby("PERMNO")                                   # 종목별
    A["r12"] = g["lr"].transform(lambda s: s.rolling(12, 6).sum())  # 직전 12개월 누적 로그수익
    A["jump6"] = g["MthRet"].transform(lambda s: s.rolling(6, 1).max())  # 최근 6개월 최대 월수익
    last = A[["PERMNO", "m", "MthPrc", "r12", "jump6"]].rename(columns={"m": "lm"})  # 직전월 후보
    e = D.merge(last, on="PERMNO")                            # 상폐 이벤트 × 정상 월
    e = e[e["lm"] < e["m"]].sort_values("lm").groupby(["PERMNO", "m"]).tail(1)  # 상폐 전 마지막 정상 월
    dl = pd.read_csv(daily_path, usecols=["permno", "ym", "rvol_d"])  # 일간 변동성
    dl["lm"] = pd.PeriodIndex(dl["ym"], freq="M")             # 월
    dl = dl.sort_values(["permno", "lm"])                     # 정렬
    dl["rv_norm"] = dl.groupby("permno")["rvol_d"].transform(lambda s: s.shift(3).rolling(12, 6).mean())  # 평소 변동성
    e = e.merge(dl[["permno", "lm", "rvol_d", "rv_norm"]].rename(columns={"permno": "PERMNO"}),
                on=["PERMNO", "lm"], how="left")              # 결합
    e["volratio"] = e["rvol_d"] / e["rv_norm"]                # 변동성 축소비
    co_last = A.groupby("PERMCO")["m"].max()                  # 회사별 마지막 정상거래 월(모든 PERMNO)
    e["permco_cont"] = e["PERMCO"].map(co_last) > e["m"]      # 상폐 후에도 같은 회사가 거래되는지
    p = e["MthPrc"].abs()                                     # 직전월 주가
    perf = (p < 5) | (e["r12"] < -0.5) | (e["dret"] < -0.3)   # 일반 상폐 조건
    pinned = (e["volratio"] < 0.5) | ((e["jump6"] >= 0.20) & (e["volratio"] < 0.75))  # 현금 인수: 인수가에 가격 고정
    healthy = (p >= 10) & (e["r12"] >= -0.25) & (e["dret"] >= -0.10)  # 주식교환 합병·자진상폐: 부실 징후 없는 건강한 종목
    merger = pinned | healthy                                 # 인수합병 조건
    cash_deal = (e["volratio"] < 0.5) & (e["dret"] >= -0.10) & (e["r12"] >= -0.5)  # 저가주라도 인수가에 고정된 현금 인수
    e["class"] = np.select([cash_deal, perf, e["permco_cont"], merger],
                           ["merger", "perf", "reclass", "merger"], "unclear")  # 우선순위대로 분류
    e["applied_ret"] = np.where(e["class"].isin(["perf", "unclear"]), DELIST_RET, e["dret"])  # 적용 수익률
    return e.reset_index(drop=True)                           # 반환


if __name__ == "__main__":                                    # 단독 실행 시 분류 요약 출력
    import os, sys                                            # 경로
    if hasattr(sys.stdout, "reconfigure"):                    # 콘솔 한글
        sys.stdout.reconfigure(encoding="utf-8")              # UTF-8
    import selection_eval as se                               # 입력·출력 경로
    ev = classify_delistings(se.V2_PATH, se.DAILY_PATH)       # 분류
    s = ev.groupby("class").agg(건수=("dret", "size"), 상폐월수익_평균=("dret", "mean"), 상폐월수익_중앙=("dret", "median"),
                                직전주가_중앙=("MthPrc", lambda x: x.abs().median()), 변동성축소비_중앙=("volratio", "median"),
                                r12_중앙=("r12", "median"))   # 요약
    print(s.round(3))                                         # 출력
    os.makedirs(os.path.join(se.OUT_DIR, "v2"), exist_ok=True)  # 결과 폴더
    ev.to_csv(os.path.join(se.OUT_DIR, "v2", "delist_events.csv"), index=False, encoding="utf-8-sig")  # 저장
