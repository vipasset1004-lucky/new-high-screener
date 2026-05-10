# CLAUDE.md — New High Screener (신고가 스크리너)

## 프로젝트 개요

신고가(52주 + 120일 첫 돌파) 종목을 자동 스캔하여 위치(패턴) 판별 + PRIME-S 등급 부여하는 스크리너.

- **운영**: GitHub Actions (cron + workflow_dispatch) + GitHub Pages
- **이전 운영**: Render (2026-05-10 GitHub Actions로 마이그레이션, $25/월 → $0)
- **라이브 URL**: https://vipasset1004-lucky.github.io/new-high-screener/
- **자동 스캔**: 평일 KST 16:00 + 21:00

## 기술 스택

- **스캔 엔진**: `new_high_screener.py` (2,680줄) + `web_server.py` (helper 모듈)
- **standalone entry**: `scan_main.py` — GitHub Actions에서 실행
- **데이터**: pykrx + Naver supply (외국인/기관 수급)
- **호스팅**: GitHub Pages (gh-pages 브랜치)
- **비용**: $0 (public 레포)

## 자원 예산 (필수 준수)

> **"정해진 룸 안에서 최적의 안을 만드는 게 진짜 엔지니어링"**

- **타겟**: GitHub Actions ubuntu-latest (7GB RAM)
- **스캔 시간 목표**: 25~30분 이내 (timeout 35분)
- **유니버스**: 큐레이션 527 + 신규상장 (자동)

### 변경 시 체크리스트
1. 새 분석 추가 전 종목당 추가 메모리/시간 측정
2. universe 확장 전 외인/기관 데이터 의미 있는 구간인지 검증
3. SQLite live-tracking 의존 함수는 GitHub Actions 휘발성 FS 호환 확인
4. 출력 경로는 OS 무관 (Linux 호환)

## 작업 원칙

1. **자원 예산 우선** — 위 섹션 따를 것
2. **이론 → 백테스트 → 알고리즘 → 코드** 순서
3. **백테스트 통과 못하면 알고리즘 회귀** (단순 코드 수정 X)
4. UI/알고리즘 축소 금지 — 풀 보존이 기본

## 마이그레이션 메모 (2026-05-10)

Render → GitHub Actions 이전 시 변경:
- `web_server.py`: `scheduler.start()` 를 `SKIP_SCHEDULER=1` 환경변수로 가드
- `scan_main.py` 추가: CLI 진입점 (auto_scan_full 호출 → results.json)
- `frontend/index.html`: `/api/cached-results` → `./results.json`, `/api/scan` SSE → workflow_dispatch 링크 (admin)
- SQLite live-tracking 임시 제외 (필요 시 향후 gh-pages 영속화로 추가)
- 캐시(newhigh_cache.json, new_listings_cache.json)는 gh-pages에서 restore + commit

## 폴더 구조

```
new-high-screener/
├── new_high_screener.py        # 스캔 엔진 (변경 없음)
├── web_server.py               # 헬퍼 모듈 (Flask는 import 안 함)
├── scan_main.py                # GitHub Actions 진입점
├── frontend/index.html         # 정적 frontend
├── .github/workflows/
│   ├── scan.yml                # 평일 16:00·21:00 cron
│   └── deploy-frontend.yml     # frontend 변경 시 빠른 배포
└── CLAUDE.md                    # 이 파일
```
