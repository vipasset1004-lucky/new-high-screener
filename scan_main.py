"""GitHub Actions 용 standalone 실행 entrypoint.

기존 web_server.py 의 auto_scan_full 흐름을 CLI 모드로 추출:
- universe refresh
- 풀 스캔 (PRIME-S 발견 + K-NN + 진입 점수)
- 결과 → results.json 저장 (frontend 정적 fetch)

SQLite live-tracking 부분은 GitHub Actions 환경에서 제외 (휘발성 FS).
필요 시 향후 gh-pages 영속화로 추가 가능.
"""

from __future__ import annotations

import gc
import io
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# UTF-8 출력
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

# 스케줄러 자동 시작 방지 (web_server import 전에 설정)
os.environ.setdefault("SKIP_SCHEDULER", "1")

# 로깅 — INFO 레벨, 한국 시간
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# new_high_screener 모듈에서 분석 함수들 import
from new_high_screener import (
    fetch_index, get_market_regime, count_distribution_days,
    fetch_daily_batch, is_near_high_candidate, fetch_supply_batch,
    fetch_naver_breadth, get_fallback_tickers,
)
# web_server 의 보조 함수 (universe + 정밀분석 등)
import web_server  # noqa: E402  (전체 모듈 — auto_scan_full 의 의존 함수 참조)


def _save_results(cache_data: dict, output_path: str = "results.json") -> None:
    """결과를 JSON으로 저장 (frontend 가 이 파일을 fetch)."""
    Path(output_path).write_text(
        json.dumps(cache_data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(f"[저장] {output_path} ({len(cache_data.get('results', []))}종목)")


def run_scan() -> dict:
    """auto_scan_full 의 핵심 흐름을 CLI 모드로 실행.

    web_server.auto_scan_full 을 직접 호출 — 그 함수가 캐시에 저장하므로
    그 결과를 가져와 JSON 으로 export.
    """
    logger.info(f"[scan] 시작: {datetime.now()}")
    started = time.time()

    # universe refresh (오프라인 모드)
    try:
        web_server.refresh_dynamic_universe()
        logger.info("[scan] universe refresh 완료")
    except Exception as e:
        logger.warning(f"[scan] universe refresh 실패 (무시): {e}")

    # 풀 스캔
    web_server.auto_scan_full()

    # 캐시에서 결과 가져오기 — auto_scan_full 이 _cache_set 으로 저장
    cache_data = web_server._cache_get(web_server.CACHE_KEY)
    if not cache_data:
        logger.error("[scan] 캐시에서 결과 못 찾음")
        return {"results": [], "scan_date": datetime.now().isoformat(),
                "total_found": 0, "total_scanned": 0, "breadth_scan": 0}

    elapsed = time.time() - started
    logger.info(f"[scan] 완료 ({elapsed:.0f}s) → "
                f"{cache_data.get('total_found', 0)}종목 / "
                f"breadth {cache_data.get('breadth_scan', 0)}%")

    # prev cache 도 함께 export (frontend 의 "연속 등장" 비교용)
    prev = web_server._cache_get(web_server.CACHE_PREV_KEY)
    if prev:
        prev_tickers = [r.get("ticker") for r in (prev.get("results") or [])
                        if r.get("ticker")]
        cache_data["prev_tickers"] = prev_tickers
        cache_data["prev_scan_date"] = prev.get("scan_date")

    return cache_data


if __name__ == "__main__":
    payload = run_scan()
    output = os.environ.get("OUTPUT_FILE", "results.json")
    _save_results(payload, output)
    print(f"\n=== 결과 ===")
    print(f"scan_date: {payload.get('scan_date')}")
    print(f"total_found: {payload.get('total_found', 0)}")
    print(f"total_scanned: {payload.get('total_scanned', 0)}")
    print(f"breadth_scan: {payload.get('breadth_scan', 0)}%")
    print(f"naver_breadth: {payload.get('naver_breadth', 'N/A')}")
    rs = payload.get("results", [])
    if rs:
        print(f"\n상위 5종목:")
        for r in rs[:5]:
            name = r.get("name", "?")
            ticker = r.get("ticker", "?")
            score = r.get("entry_score", r.get("score", 0))
            grade = r.get("grade", "?")
            ps = "★" if r.get("is_prime_s") else (
                "●" if r.get("is_prime") else " ")
            print(f"  {ps} {ticker} {name:14s} 등급 {grade}  점수 {score}")
