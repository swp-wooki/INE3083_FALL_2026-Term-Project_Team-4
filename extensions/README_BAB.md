# BAB 확장 실험

이 브랜치는 AQR BAB 팩터를 쓴 두 확장 실험의 코드, 외부 데이터, 결과를 담고 있습니다. `src/`의 보고서 코드는 import만 하고 수정하지 않습니다.

- **E (`extE_factor_cov.py`)**: FF5, FF5+BAB 팩터 공분산으로 GMVP를 만들어 표본·LW 추정치와 비교
- **⑨ (`ext09_alpha_bab.py`)**: 보고서 19개 전략의 월별 수익률을 FF5와 FF5+BAB에 회귀해 알파 변화 확인

```powershell
pip install -e .
python extensions\extE_factor_cov.py
python extensions\ext09_alpha_bab.py
```

## 파일

| 경로 | 내용 |
|:--|:--|
| `extensions/external_data/aqr_bab_usa.csv` | AQR 공개 미국 BAB 월간 수익률 (2026-11-03 다운로드) |
| `extensions/external_data/aqr_bab_usa.source.json` | 출처 URL, 다운로드 시각, 원본 파일 SHA-256 |
| `extensions/extE_factor_cov.py`, `extensions/ext09_alpha_bab.py` | 실험 스크립트 |
| `extensions/common.py`, `solvers.py`, `ext01_window_cap.py`, `check_EA_conditions.py` | 두 스크립트가 import하는 공용 함수 |
| `results/extensions/extE_factor_cov/` | E의 수치, 쌍별 검정, 월별 수익률, 보유 비중, 솔버 진단, 해석(`notes.md`) |
| `results/extensions/ext09_alpha_bab/` | ⑨의 19개 전략 × 3개 기간 회귀 계수(`metrics.csv`), 요약, 해석(`notes.md`) |

## 주의

- BAB는 거래비용이 없는 롱숏 팩터이고, 전략 수익률은 비용 차감 후 롱온리입니다.
- 실행하면 `results/extensions/RESULTS.md`와 `all_results.csv`가 다시 만들어집니다. 이 브랜치에는 BAB 실험 두 개만 들어 있으므로, 그 파일에도 이 두 실험만 나옵니다.
- `.gitignore`가 `extensions/`와 `results/extensions/`를 제외하므로, 파일을 추가하려면 `git add -f`가 필요합니다.
