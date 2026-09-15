# TK-BIC × ENZ-ITO(23 nm) 비선형 activation — 전자온도(Te) 피드백 코드 공유 패키지

작성 2026-09-11. 공유한 세 그림(온도별 `T(λ;Te)`, 전달함수 스캔 `T(E_in;λp)`, `T(I)` RCWA vs TCMT)을
그린 **원본 코드 + 캐시 + 독립 실행 데모**입니다. 질문의 핵심인 "온도에 따라 ε 이 바뀌면 흡수·투과가 다시
바뀌는 피드백을 어떻게 처리했나"에 대한 답을 §2 에 정리했습니다.

## 0. 한 줄 요약

- ε_ITO(λ, Te) 는 측정 ε 에 Drude 항의 **변화분만** 얹은 delta-Drude(Kane) 식.
- 그 ε(Te) 를 RCWA 에 넣어 **Te 9개(300–8000 K)에서 스펙트럼 표(family) T, A(λ; Te)** 를 미리 만든다.
- 피드백 루프는 표 보간으로 닫는다. 두 가지 방식:
  - **(1) 준정적 에너지수지 고정점** — `∫C_e dTe = A(λp, Te)·F/t_ITO` 를 Te 에 대해 자기일관으로 푼다
    (논문 노트북, 공유한 세 그림이 전부 이 방식). RCWA 를 루프 안에서 다시 돌리지 않는다.
  - **(2) 2온도모델(TTM) 시간적분** — `C_e dTe/dt = A(λp, Te(t))·F·I(t)/t_ITO − g(Te−T_l)` 를 RK4 로
    1 fs 스텝 적분하며 매 스텝 A 를 Te(t) 로 재평가 ("time step 으로 나눠서" 하는 버전, 9월 4일 추가).
- `python tkbic_te_feedback_demo.py` 한 번(약 30 초, GPU/torcwa 불필요)에 그림 6장이 `figures/` 에 다시 나온다.

## 1. 세 그림 ↔ 파일 매핑

| 공유한 그림 | 노트북 (original/TKBIC_t23_paper.ipynb) | 빌더 줄 (original/_build_tkbic_t23_paper_nb.py) | 데모 출력 |
|---|---|---|---|
| (a) T(λ; Te) 온도별 스펙트럼 | §8 Fig 8(a) | 690–743 | figures/fig08_T_lambda_Te_family.png |
| (a) 전달함수 스캔 T(E_in; λp) 히트맵 | §9 Fig 10(a) | 937–1100 | figures/fig10_transfer_scan_T_E_lambda_p.png |
| 진단: RCWA(실선) vs 안정화 TCMT(파선), T vs peak intensity | §9 Fig 12 | 1149–1186 | figures/fig12_T_vs_I_RCWA_vs_TCMT.png |

빌더는 노트북을 **생성**하는 스크립트라 셀 코드가 `code(r'''…''')` 문자열 안에 그대로 들어 있습니다.
데모 스크립트 `tkbic_te_feedback_demo.py` 는 그 셀들을 순서대로 떼어내 하나의 .py 로 만든 것입니다
(§ 번호가 빌더와 대응, 주석에 물리 설명 추가). 노트북 자체는 jupyter 로 열면 실행 결과 그림이 전부 들어 있습니다.

## 2. 물리 사슬과 피드백을 닫는 방법

```
레이저 세기 I ──► 흡수 A(λp, Te) ──► 전자온도 Te ──► ε_ITO(λ, Te) ──► RCWA T, A(λ; Te) ──┐
       ▲                                                                                  │
       └──────────────────────────── (다시 A 가 바뀜: 피드백) ◄──────────────────────────┘
```

### 2.1 ε_ITO(λ, Te) — 측정-anchored delta-Drude (Kane 비포물선)  [데모 §2, `eps_ITO_at_Te`]

```
ε(λ; Te) = ε_meas(λ) − [ħω_p²(Te) − ħω_p0²] / (E (E + iħγ0)),      E = hc/λ
ħω_p(Te) = ħω_p0 · sqrt( m*(300)/m*(Te) ),   m*(Te)/m0 = 0.35 (1 + 2·0.4191 (1 + (π²/4)(k_B Te)²))
```
- ħω_p0 = 1.7759 eV, ħγ0 = 0.1268 eV 는 측정 ε(1000–1650 nm) 의 Drude fit. ε_∞ 는 식에 안 들어감.
- Te = 300 K 에서 측정값으로 **정확히 환원**(데모에 assert). 가열 → m* 증가 → ω_p 감소 → ENZ 적색이동.
- 예: ε(1292 nm) = 0.053+0.419i (300 K) → 0.286+0.388i (3000 K) → 1.224+0.264i (8000 K).
- 같은 식이 `ttm/_ito_eps_Te.py` 에 독립 모듈로도 있음 (Lumerical/meep 에도 동일 표 주입).

### 2.2 RCWA family — ε(Te) 가 솔버에 들어가는 지점  [데모 §5, 캐시 cache/tkbic_t23_Te*.npz]

- 구조: a-Si:H 원기둥(h 970.5, d 405.7 nm, 주기 588) / ITO 23 nm (균일층, ε(λ,Te)) / glass. torcwa, 차수 10, 격자 256.
- Te ∈ {300, 600, 1000, 1500, 2000, 3000, 4500, 6000, 8000} K 마다 λ 1240–1360 nm @1 nm 스펙트럼을 **독립적으로**
  계산 (Te 하나당 121 λ 점). 이것이 공유한 그림 (a) 이고, 흡수는 A = 1 − T − R.
- 결과: dip 1291.6 → 1298.5 nm 이동, FWHM 22.4 → 10.7 nm **선폭붕괴**, depth 0.66 → 0.89.
  (가열로 ENZ 가 비켜나 ITO 드레인 γ_nr 이 8.5 → 0.3 meV 로 붕괴하는 것이 주효과, 이동은 부수효과.)
- **RCWA 는 여기서 끝.** 이후 피드백 루프는 이 표를 (λ 선형 → Te 선형) 보간해서만 쓴다 (`_at(λp, Te, fam)`).
  Te 가 표 범위 밖이면 8000 K 값으로 clip (그래서 8000 K 이상은 "표 상한" 라벨).

### 2.3 방법 1 — 준정적 에너지수지 고정점 (노트북, 세 그림)  [데모 §6, `Te_balance`, `transfer`]

150 fs 펄스는 전자-격자 완화(수백 fs)보다 짧다고 보고 펄스 동안 ITO 전자계에 들어간 에너지가 전자온도로만 간다고 놓는다:
```
∫_300^Te C_e dT = A(λp, Te) · F / t_ITO ,   C_e = a_Ce·Te (Sommerfeld, a_Ce = 5.87 J m⁻³ K⁻²)
⇔  Te² − 300² = 2 A(λp, Te) F / (t_ITO a_Ce)          ← 우변에 Te 가 들어 있음 = 자기일관 방정식
```
- F = 입사 플루언스 [J/m²] = I_peak·τ_eff (τ_eff = 150 fs·√(π/2)), 픽셀 (8 µm)² 기준 1 nJ = 1.56 mJ/cm².
- 풀이 (`Te_balance`): ① 감쇠 고정점 반복 `Te ← (1−β)Te + β·sqrt(300² + 2A(Te)F/(t a))`, β = 0.35, 60회
  ② 그 근방 ±15 % 를 괄호로 잡아 이분법 40회. 왜 감쇠+이분법인가: λp 고정에서 A(Te) 가 가파르게 변해
  (dip 이 λp 를 지나가면) 순수 반복은 진동하고, 이분법은 괄호만 잡히면 무조건 수렴하기 때문.
- 연속법 (`transfer`): 세기를 낮은 값부터 올리며 **직전 Te\* 를 다음 초기값**으로 쓴다 (해가 둘일 때 가지 점프 방지).
- 출력: 세기 배열 → (T(λp,Te\*), A(λp,Te\*), Te\*). 이것이 활성화 함수 T(E_in).
- λp 스캔 (Fig 10(a)): λp 를 dip ±20 nm, 1 nm 간격으로 바꿔가며 위 전달함수를 반복. **RCWA 추가 계산 0** —
  family 가 1240–1360 nm 를 덮으니 λp 선택은 공짜. 41×49 고정점 해가 약 9 초.
- 자동 선정 결과: ReLU-like λp = 1289.6 nm (ΔT +0.48), sigmoid-like 1287.6 (+0.51), saturable 1299.6 (−0.39).

### 2.4 방법 2 — TTM 시간적분 ("time step" 버전)  [ttm/_te_fluence_map.py, 데모 §8]

```
C_e(Te) dTe/dt = A(λp, Te(t)) · F · I_norm(t) / t_ITO − g (Te − T_l)
C_L     dT_l/dt =                                     g (Te − T_l)      g = C_e/τ_ep (τ_ep 300–1000 fs) 또는 상수
```
- RK4, dt = 1 fs, 창 −1 … +3 ps, 가우스 펄스 (모듈 기본 FWHM 280 fs = 실측 레이저; 데모는 노트북과 같은 150 fs 로 맞춤).
- **매 RK4 스텝에서 `Afun(Te)` 를 현재 Te(t) 로 다시 평가** — 이것이 피드백을 시간축에서 닫는 것. Afun 은 같은 family 표 보간.
- 출력: Te_pk, 펄스평균 <Te>_I, 1 ps 후 Te, 격자온도, 흡수 에너지밀도, 펄스평균 투과 <T>_I, 역맵(목표 Te → F) 등.
- CLI 예: `python ttm/_te_fluence_map.py --family rcwa --lam-pump 1292 --tau 300,450,600,1000 --out figures`
  (cache/tkbic_angnl_th0.npz = 확정 설계(Al2O3 4 nm 포함)의 family, 1200–1420 nm; `--family fdtd` 는 Lumerical FDTD family).
- C_e 모델 4종(`--ce sommerfeld|min|fd|kane`) 과 τ_ep 를 바꿔 **Te 라벨의 모델 폭**을 볼 수 있다.

### 2.5 두 방법은 얼마나 다른가 — figures/fig13_quasistatic_vs_TTM.png (λp = 1289.6 nm, 1 nJ)

| 방법 | Te @ 1 nJ | 비고 |
|---|---|---|
| 준정적 고정점, 노트북 a_Ce = 5.87 | 7090 K | 공유한 그림들의 눈금 |
| 준정적 고정점, γ_e = 5.23 (TTM 모듈 규약) | 7303 K | 열용량 상수만 다름 |
| TTM g = 0 (e-ph 무손실) | 8390 K | 끝점 A(Te\*=7303 K) = 0.205 대신 **경로평균 A_eff = 0.271** 을 흡수 |
| TTM τ_ep = 1000 fs peak | 7537 K | |
| TTM τ_ep = 450 fs peak | 6929 K | 펄스 중 격자로 새는 만큼 낮아짐 |

- 준정적 고정점은 **끝점 흡수 A(Te\*)** 만 쓴다. ReLU-like 펌프(λp < dip)는 가열될수록 A 가 줄므로 차가울 때
  흡수한 몫을 빼먹는다. TTM(g=0) 을 경로평균 A_eff 로 준정적 식에 넣으면 8391 K 로 정확히 일치 (데모 판정 ①).
- e-ph 유출이 있으면 peak 가 내려오고 (판정 ②: g=0 ≥ τ1000 ≥ τ450), 결과적으로 노트북 눈금과 τ≈450 fs TTM 이
  비슷해 보이는 것은 두 근사가 반대 방향으로 상쇄된 우연이다. **절대 Te 눈금은 모델 라벨(×2 폭)** 로 인용할 것.
  실험 앵커는 E_sat 0.464 nJ ≙ Te_pk 4.7 kK (FDTD A_ITO + TTM τ450, 모델 띠 4.3–5.8 kK).
- 활성화 함수의 **모양**(문턱, 포화, 부호)은 두 방법이 같다. 다른 건 E_in 축의 눈금.

### 2.6 Fig 12 의 파선 — TCMT 로 같은 피드백을 해석적으로 닫기  [데모 §7]

- 300 K 스펙트럼에서 TCMT 파라미터 추출: Σ = FWHM, x = T_min/T_bg → γ_nr = Σ√x = 8.54, γ_r = Σ(1−√x) = 8.54 meV,
  ρ = γ_r/γ_nr = 1.00 (critical coupling), A_pk = 0.471 (폐쇄관계 2√x(1−√x) = 0.5).
- 1차 섭동(미공액 복소): δω̃(Te) = −(ω0/2)·Δε(Te)·∫_ITO E·E / ∫ εE·E — 실부 = 공진 이동, 허부 = 선폭 변화.
  겹침적분은 300 K 배경차감 공진장 3D 적분(캐시 cache/tkbic_t23paper_field3d.npz, |W_c| = 1.57 %, 위상 −57°).
  |E|² 대신 **E·E** 를 쓰는 이유: ENZ 층의 E_z 는 위상이 돌아 있어 |E|² 형식은 이동을 과대평가.
- 이 해석식 T, A(λ; Te) 를 같은 `Te_balance` 에 넣어 T(I) 를 만든 것이 파선. 자유 파라미터 0 으로 RCWA 실선과
  RMS 0.084 (스펙트럼 RMS 0.046). 예측 이동 +5.2 nm (RCWA +6.9), γ_nr 8.5 → 0.3 meV.

## 3. 파일 목록

```
tkbic_te_feedback_demo.py   독립 실행 데모 (세 그림 + Fig 9/11/13 재현, 판정 8개). 캐시 있으면 torch 불필요.
README.md / requirements.txt
Materials_data/             측정 nk: ITO_nk.csv (6열 헤더 csv — 열 순서 함정 주의), aSi_H_measured_Postech.txt, SiO2_substrate_measured.txt
cache/
  tkbic_t23_Te{300..8000}.npz     RCWA family (λ 1240–1360, T/R, meta P/H/D/TITO/MX/NX/TE)   ← 그림 (a)
  tkbic_t23_pol_h+95.npz          300 K 고해상 스펙트럼 1200–1400 nm (작동점 dip·TCMT 추출)
  tkbic_t23paper_field3d.npz      TCMT 겹침적분 (W_ITO, denom, 복소 W_ITO_c/denom_c)
  tkbic_angnl_th0.npz             확정 설계(Al2O3 4 nm) family 1200–1420 nm, p/s 편광 — TTM CLI 기본 입력
  tkbic_lumerical_Te_family.npz   Lumerical FDTD 2025 R2 Te family (mesh 4, 9 Te; A_ito 포함) — --family fdtd
  tkbic_meep_matfits.json         Drude 상수 (hwp 1.775882, hgam 0.126846 eV)
ttm/
  _te_fluence_map.py        TTM 시간적분 + F↔Te 역맵 + C_e 모델 4종 (경로만 패키지 상대경로로 패치)
  _ito_eps_Te.py            ε_ITO(λ,Te) 독립 모듈 + self-test (`python ttm/_ito_eps_Te.py`)
original/
  TKBIC_t23_paper.ipynb            논문 노트북 (실행 결과·그림 포함, 54셀)
  _build_tkbic_t23_paper_nb.py     노트북 빌더 (셀 원문)
  _full_nb_parts.json              빌더가 공유하는 재료/솔버 셀 (mat, sol)
figures/                    데모 출력 (fig08–13 png + demo_summary.npz) + TTM CLI 예시 출력 (tkbic_te_fluence_map.png/.txt/.npz,
                            `--nF 12` 소형 격자; 실제 사용 시 --nF 40 기본값으로 다시 돌릴 것)
```

## 4. 실행

```
pip install numpy scipy matplotlib          # 캐시 재사용 → 이것만으로 충분
python tkbic_te_feedback_demo.py            # 약 30 초, figures/ 에 저장. --show 로 화면 표시
python tkbic_te_feedback_demo.py --no-ttm   # §8 TTM 비교 생략
python tkbic_te_feedback_demo.py --recompute --threads 8   # RCWA 재계산 (torch+torcwa, GPU 권장, Te 9개×121 λ ≈ 30–60 분)
```
- 재계산 시 복소 고유값분해는 CPU(MKL) 에서 돌아 코어 수가 속도를 좌우. `--threads` 로 조정.
- 검증된 조합: python 3.10, torch 2.5.1+cu121, torcwa 0.1.4.2, matplotlib 3.7. `KMP_DUPLICATE_LIB_OK=TRUE` 는 스크립트가 설정.
- 한글 폰트: Malgun Gothic → NanumGothic → Noto Sans CJK KR 순으로 있는 것을 씀 (없으면 한글만 □ 로 나옴, 수치는 무관).

재현 확인용 수치 (데모 stdout, 2026-09-11):

| 항목 | 값 |
|---|---|
| ENZ (Re ε = 0) | 1302.28 nm |
| 작동점 dip / T_min / FWHM | 1291.59 nm / 0.214 / 23.0 nm |
| γ_r = γ_nr / ρ / A_pk | 8.54 meV / 1.00 / 0.471 |
| family FWHM, dip (300 → 8000 K) | 22.4 → 10.7 nm, 1291.6 → 1298.5 nm |
| λp ReLU / sigmoid / saturable | 1289.6 / 1287.6 / 1299.6 nm |
| ΔT (0 → 2.4 nJ) | +0.48 / +0.51 / −0.39 |
| TCMT 스펙트럼 RMS / T(I) RMS 중앙값 | 0.046 / 0.084 |

## 5. 주의·한계 (인용 전 확인)

1. **흡수 절대값**: torcwa S-matrix 의 1−T−R 은 ENZ 공진 흡수를 약 15 % 과대평가한다 (필드적분·fmmax·meep·fdtdx·
   Lumerical 5-방법 클러스터 0.46 vs torcwa 0.53). 여기 family 의 A 는 보정 전 값. E_sat 등 절대 눈금은
   FDTD 앵커(A_res 0.405, E_sat 0.464 nJ)를 쓰고, 비율·모양·λp 선택은 그대로 유효.
2. **Te 눈금은 모델 라벨**: C_e(Sommerfeld vs Kane/FD), τ_ep, 끝점-A vs 경로-A 에 따라 ×2 폭. §2.5 참조.
3. **τ_eff 규약**: 노트북은 F = I·150 fs·√(π/2) (=188 fs). 가우스 펄스의 정확한 ∫I dt = I_pk·FWHM·1.064 (=160 fs).
   I 축 라벨이 약 18 % 이동하는 정도이며 E_in(nJ) 축에는 영향 없음.
4. **펌프 파장 ≠ dip**: 시료 dip 이 1292 가 아니면 λp 선택을 측정 dip 기준 디튠(−4/0/+8 nm)으로 다시 잡을 것.
   1300 nm 펌프로 1292 dip 을 치면 자기제한 롤오프가 사라지고 단일빔 T 가 어두워진다.
5. 표 범위 8000 K 이상은 clip. 고에너지 끝(> ~1.5 nJ)은 표 상한의 영향을 받는다.
6. 실험 앵커 관련 상세(Lumerical Te family, TTM 역맵, 펌프-프로브 예측)는 원 프로젝트 HANDOFF.md
   "ITO 23 nm 핫전자 물리 검토"·"Lumerical Te-패밀리 FDTD 독립 검증" 절 참조 (이 패키지엔 미포함).
