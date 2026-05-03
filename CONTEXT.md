# 신고가 위치 판별 엔진 — 작업 맥락 문서

다른 PC나 새 세션에서 작업을 이어갈 때 이 파일을 Claude에게 보여주세요.

---

## 프로젝트 개요

**목적:** 한국 주식(KOSPI + KOSDAQ) 신고가 종목을 자동 스캔하여 위치(패턴)를 판별하는 웹 스크리너

**배포:**
- 백엔드(Flask API): Render.com → `https://new-high-screener-xxxx.onrender.com` (배포 중 또는 완료)
- 프론트엔드: GitHub Pages → `https://vipasset1004-lucky.github.io/new-high-screener/`
- 코드 레포: `https://github.com/vipasset1004-lucky/new-high-screener`

---

## 파일 구조

```
new-high-screener/
├── new_high_screener.py   # 핵심 분석 엔진 (v4)
├── web_server.py          # Flask SSE 백엔드 + Upstash Redis 캐시
├── index.html             # 프론트엔드 (다크테마, 모바일 최적화)
├── render.yaml            # Render.com 배포 설정
├── requirements.txt       # Python 의존성
└── CONTEXT.md             # 이 파일
```

---

## 분석 엔진 구조 (new_high_screener.py v4)

### 4가지 위치 분류
| 코드 | 이름 | 설명 |
|------|------|------|
| `BREAKOUT_BOTTOM` | 바닥탈출형 | 120일 박스 첫 돌파, 주도주 초기 단계 [최상급] |
| `PULLBACK_REBREAK` | 눌림재돌파형 | 1차 상승→조정→2차 시작 [우수] |
| `TREND_CONTINUE` | 추세진행형 | 추세 진행 중 [중립] |
| `OVERHEATED` | 과열 | 분배 구간, 자동 제외 |

### 7단계 분석 파이프라인
1. **STEP1** 신고가 탐지 (52주 + 120일 첫 돌파 여부)
2. **STEP2** 위치 판별 (4분류)
3. **STEP3** Minervini Trend Template (8기준, 최대 25점)
4. **STEP4** VCP 수렴 강도 — 진행 수축 패턴 (최대 20점)
5. **STEP5** 돌파 수급 — RVOL + 최초 거래대금 급증 (최대 20점)
6. **STEP6** 유지력 + RS — 3일 버팀 + Retest + 상대강도 (최대 25점)
7. **STEP7** 과열 감점 — RSI/급등/윗꼬리

### 주요 지표
- **ATH 감지**: 현재 52주 고가 = 역대 최고가 여부 (`is_ath`)
- **베이스 기간**: 돌파 전 수렴 일수 (`vcp_detail.base_days`)
- **Fibonacci 0.382**: 눌림 조정 깊이 (`fib_depth`)
- **MA20 이격도**: 20일선 대비 20% 초과 시 감점
- **Retest 패턴**: 신고가 닿고 3~8% 눌림 후 재돌파 (승률 76%)

### 종목 리스트
- `get_fallback_tickers(top_n=527)`: KOSPI + KOSDAQ 527종목 (하드코딩)
- `get_new_listings()`: KRX에서 신규상장 종목 자동수집 (1.2년~3년 내 상장, 7일 캐시)

---

## 백엔드 (web_server.py)

### API 엔드포인트
| 경로 | 설명 |
|------|------|
| `GET /` | index.html 서빙 |
| `GET /api/scan` | SSE 스트리밍 스캔 (`market`, `top_n`, `min_score`, `include_new` 파라미터) |
| `GET /api/cached-results` | 마지막 스캔 캐시 반환 |
| `GET /api/results` | 캐시 데이터 반환 |

### 캐시 시스템
- **Upstash Redis** (환경변수 있을 때): `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`
- **파일 폴백** (Redis 없을 때): `newhigh_cache.json`, `newhigh_cache_prev.json`
- 이전 스캔 결과도 `prev` 키로 보존

### 자동 스캔
- APScheduler로 **매일 21시(KST)** 자동 스캔
- 결과 캐시에 저장

---

## 프론트엔드 (index.html)

### UI 구성
- **다크테마** + **모바일 최적화** (iOS safe-area, viewport-fit=cover)
- **필터 탭**: 전체 / 바닥탈출형 / 눌림재돌파형 / 추세진행형 / 킬러필터 / Retest / 역사적신고가 / S·A+만
- **실시간 SSE 스트리밍**: 스캔 진행 상황 실시간 표시
- **카드 UI**: 종목명(22px), 점수 원, Minervini 진행바, 신호 배지, 수치 그리드, 52주 범위바
- **페이지 로드 시** 캐시된 결과 자동 표시

### 점수 등급
| 등급 | 기준 | 색상 |
|------|------|------|
| S | 80점 이상 | 금색 |
| A+ | 70점 이상 | 보라 |
| A | 60점 이상 | 파랑 |
| B | 50점 이상 | 초록 |

### 종목 선택 옵션
- 상위 150 / 300 / 450 / 527(전체) 종목
- 최소 점수: 40 / 50 / 60점
- 신규상장 포함 여부 체크박스

---

## 배포 설정 (render.yaml)

```yaml
services:
  - type: web
    name: new-high-screener
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn web_server:app --bind 0.0.0.0:$PORT --timeout 300 --workers 2 --threads 2
    envVars:
      - key: PYTHON_VERSION
        value: "3.12"
```

**Render에 추가할 환경변수 (선택):**
- `UPSTASH_REDIS_REST_URL`
- `UPSTASH_REDIS_REST_TOKEN`

---

## 다음 작업 후보

- [ ] Render 배포 URL 확인 및 테스트
- [ ] Upstash Redis 연결 (선택)
- [ ] 미국 주식 스캔 지원 추가 (S&P500 등)
- [ ] 종목 상세 페이지 (차트 연동)
- [ ] 텔레그램/슬랙 알림 연동

---

## 작업 이어가는 법

새 Claude Code 세션에서 이 파일을 열고:

```
이 CONTEXT.md를 읽고 new-high-screener 프로젝트 작업을 이어서 해줘.
GitHub: https://github.com/vipasset1004-lucky/new-high-screener
```
