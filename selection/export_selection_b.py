"""
B안(저위험·공왜도 방어형) 월별 선정 종목 → selection_output_template.csv 형식으로 저장
  factor_1 = 저베타 FP형(low_beta_fp) 백분위, factor_2 = 저변동성 12개월(low_vol_12m) 백분위, factor_3 = 음의 공왜도(coskew) 백분위
  composite_score = 세 백분위의 동일 가중 평균, selection_rank = 그 달 선정 30종목 안에서 composite_score 순위(1 = 최고)
  weight = 실제 보유 비중(형성월 시가총액 비례, 종목당 최대 2/30, 초과분은 나머지에 시총 비례 재배분, 월별 합 = 1)
  선정 규칙: Top-30, 버퍼 1.5(기존 보유는 유니버스 순위 45위 안이면 유지), 산업(SIC2) 최대 9종목 — 저위험_공왜도_방어형_전략.md 1장
실행(저장소 루트): python selection/export_selection_b.py
"""
import os                                    # 경로
os.environ["FR_PROFILE"] = "cap2"            # 시총가중·최대 2/k 프로필(전략 문서와 동일)
import sys                                   # 출력
import pandas as pd                          # 데이터프레임
import selection_eval as se                  # 프레임워크
import factor_research_v2 as fr              # 연구 도구

if hasattr(sys.stdout, "reconfigure"):                        # 콘솔 한글
    sys.stdout.reconfigure(encoding="utf-8")                  # UTF-8
NAMES = ["low_beta_fp", "low_vol_12m", "coskew"]              # factor_1~3
K, BUF, IND = 30, 1.5, 0.3                                    # 구성
OUT_CSV = os.path.join(se.OUT_DIR, "defensive_selection.csv")  # 결과 파일(data/selections/defensive_selection.csv와 같아야 함)
COLS = ["formation_month", "holding_month", "PERMNO", "TICKER", "selection_rank",
        "factor_1", "factor_2", "factor_3", "composite_score", "weight"]  # 템플릿 열 순서 + 보유 비중


def main():
    d = fr.load_data()                                        # 데이터
    R = {n: fr.ranks_of(n, d) for n in NAMES}                 # 유니버스 내 백분위(형성월)
    comp = fr.composite(d, NAMES)                             # 동일 가중 종합 점수
    w = fr.topk_weights(comp, d, K, buffer=BUF, ind_cap=IND)  # 선정(비중 > 0 = 선정)
    ife = pd.read_csv(se.IFE_PATH, usecols=["PERMNO", "date", "TICKER"])  # 월별 티커
    ife["month"] = pd.to_datetime(ife["date"]).dt.to_period("M")  # 월
    tick = ife.drop_duplicates(["PERMNO", "month"], keep="last").set_index(["month", "PERMNO"])["TICKER"]  # (월, 종목) → 티커
    rows = []                                                 # 결과
    for t in w.index:                                         # 형성월
        sel = w.loc[t][w.loc[t] > 0].index                    # 선정 종목
        sc = comp.loc[t, sel].sort_values(ascending=False)    # 점수 순
        for rank, p in enumerate(sc.index, 1):                # 순위
            rows.append({"formation_month": str(t), "holding_month": str(t + 1), "PERMNO": int(p),
                         "TICKER": tick.get((t, p), ""), "selection_rank": rank,
                         "factor_1": R[NAMES[0]].loc[t, p], "factor_2": R[NAMES[1]].loc[t, p],
                         "factor_3": R[NAMES[2]].loc[t, p], "composite_score": sc[p],
                         "weight": w.loc[t, p]})              # 행
    df = pd.DataFrame(rows, columns=COLS)                     # 표
    for c in ["factor_1", "factor_2", "factor_3", "composite_score", "weight"]:  # 소수 자리
        df[c] = df[c].round(6)                                # 6자리
    df.to_csv(OUT_CSV, index=False, encoding="utf-8", lineterminator="\r\n")         # 저장(템플릿과 같은 머리행)
    n = df.groupby("formation_month").size()                  # 월별 종목 수
    print(f"저장: {OUT_CSV}\n행 {len(df):,}, 형성월 {n.size}개 ({n.index.min()} ~ {n.index.max()}), 월별 종목 수 {n.min()}~{n.max()}")  # 요약
    print("결측 TICKER:", int((df["TICKER"] == "").sum() + df["TICKER"].isna().sum()), "/ 결측 팩터:", int(df[COLS[5:]].isna().sum().sum()))  # 점검
    print("점수 검산 최대 오차:", float((df[["factor_1", "factor_2", "factor_3"]].mean(1) - df["composite_score"]).abs().max()))  # 검산
    ws = df.groupby("formation_month")["weight"].sum()        # 월별 비중 합
    print(f"비중 합 범위 {ws.min():.6f} ~ {ws.max():.6f}, 최대 비중 {df['weight'].max():.6f} (상한 2/30 = {2/30:.6f})")  # 점검
    print(df.head(5).to_string(index=False))                  # 미리보기


if __name__ == "__main__":                                    # 실행
    main()                                                    # 메인
