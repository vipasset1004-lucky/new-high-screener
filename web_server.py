"""
신고가 지속 상승 스크리너 웹 서버
Flask + SSE 실시간 스캔
"""

from flask import Flask, Response, request, send_file, jsonify
from flask.json.provider import DefaultJSONProvider
import json
import time
import threading
import gc
import os
import sqlite3
from datetime import datetime, date
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import numpy as np
import pandas as pd

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, o):
        if isinstance(o, np.bool_):    return bool(o)
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)

def _np_default(o):
    if isinstance(o, np.bool_):    return bool(o)
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return None if (o != o) else float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    raise TypeError(f"{type(o)} not serializable")

import math as _math
def _sanitize(obj):
    if isinstance(obj, float) and (_math.isnan(obj) or _math.isinf(obj)):
        return None
    if isinstance(obj, dict):  return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_sanitize(v) for v in obj]
    return obj

def _dumps(obj):
    return json.dumps(_sanitize(obj), ensure_ascii=False, default=_np_default)

from new_high_screener import (
    get_fallback_tickers, get_dynamic_universe, analyze_stock, fetch_index,
    get_market_regime, post_process_results, fetch_naver_breadth,
    enrich_with_predict, check_gap_at_open, calculate_entry_score, add_entry_scores,
    fetch_daily_batch, is_near_high_candidate,
    fetch_supply_batch, count_distribution_days,
    add_multibagger_scores,
)

app = Flask(__name__)
app.json = NumpyJSONProvider(app)

# ── Redis 캐시 (Upstash) ──────────────────────────────
_redis = None
try:
    _url = os.environ.get("UPSTASH_REDIS_REST_URL")
    _tok = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    if _url and _tok:
        from upstash_redis import Redis
        _redis = Redis(url=_url, token=_tok)
        print("[cache] Upstash Redis 연결됨")
    else:
        print("[cache] Redis 없음 → 파일 폴백")
except Exception as e:
    print(f"[cache] Redis 초기화 실패: {e}")

CACHE_KEY      = "newhigh:cache"
CACHE_PREV_KEY = "newhigh:cache_prev"
UNIVERSE_KEY   = "newhigh:universe"
CACHE_TTL      = 60 * 60 * 48
CACHE_PATH      = "newhigh_cache.json"
CACHE_PREV_PATH = "newhigh_cache_prev.json"
UNIVERSE_PATH   = "newhigh_universe.json"


def _key_to_path(key):
    return {CACHE_KEY: CACHE_PATH, CACHE_PREV_KEY: CACHE_PREV_PATH,
            UNIVERSE_KEY: UNIVERSE_PATH}.get(key, CACHE_PATH)

import re as _re
def _safe_loads(s):
    if isinstance(s, bytes): s = s.decode("utf-8", errors="replace")
    return json.loads(_re.sub(r'\bNaN\b', 'null', s))

def _cache_get(key):
    if _redis:
        try:
            val = _redis.get(key)
            return _safe_loads(val) if val else None
        except Exception as e:
            print(f"[cache] get 오류: {e}")
    path = _key_to_path(key)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return _safe_loads(f.read())
        except:
            pass
    return None


def _cache_set(key, value):
    if _redis:
        try:
            _redis.set(key, _dumps(value), ex=CACHE_TTL)
            return
        except Exception as e:
            print(f"[cache] set 오류: {e}")
    path = _key_to_path(key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(_dumps(value))
    except Exception as e:
        print(f"[cache] 파일 저장 오류: {e}")


scan_state = {"running": False}


def get_merged_universe():
    """큐레이션 538 + 동적 유니버스(캐시) 합집합 반환"""
    curated = get_fallback_tickers(9999)
    curated_codes = {t["ticker"] for t in curated}

    cached = _cache_get(UNIVERSE_KEY)
    dynamic = cached.get("tickers", []) if isinstance(cached, dict) else []

    merged = list(curated)
    for t in dynamic:
        if t["ticker"] not in curated_codes:
            merged.append(t)

    return merged


def refresh_dynamic_universe():
    """매일 20:50 — KRX 유니버스 갱신 후 캐시 저장"""
    print(f"[20:50 유니버스] 갱신 시작: {datetime.now()}")
    try:
        tickers = get_dynamic_universe()
        if tickers:
            _cache_set(UNIVERSE_KEY, {
                "updated_at": datetime.now().isoformat(),
                "count": len(tickers),
                "tickers": tickers,
            })
            merged = get_merged_universe()
            print(f"[20:50 유니버스] 완료 → 동적 {len(tickers)}개 / 합산 {len(merged)}개")
        else:
            print("[20:50 유니버스] 결과 없음 — 큐레이션 유지")
    except Exception as e:
        print(f"[20:50 유니버스] 오류: {e}")


@app.route("/")
def index():
    response = send_file("index.html")
    # 모바일 브라우저 캐시 방지 — 항상 최신 HTML/JS 받도록
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/api/cached-results")
def cached_results():
    data = _cache_get(CACHE_KEY)
    prev = _cache_get(CACHE_PREV_KEY)
    if data:
        return jsonify({**data, "prev": prev, "from_cache": True})
    return jsonify({"results": [], "from_cache": False,
                    "message": "캐시 없음. 스캔을 실행해주세요."})


@app.route("/api/results")
def get_results():
    data = _cache_get(CACHE_KEY)
    if data:
        return jsonify(data)
    return jsonify({"results": [], "scan_date": None})


@app.route("/api/reset-scan", methods=["POST", "GET"])
def reset_scan():
    scan_state["running"] = False
    return jsonify({"ok": True, "message": "스캔 상태 초기화 완료"})


@app.route("/api/universe-status")
def universe_status():
    cached = _cache_get(UNIVERSE_KEY)
    if cached and isinstance(cached, dict):
        merged = get_merged_universe()
        return jsonify({
            "updated_at": cached.get("updated_at"),
            "dynamic_count": cached.get("count", 0),
            "merged_count": len(merged),
        })
    curated = get_fallback_tickers(9999)
    return jsonify({"updated_at": None, "dynamic_count": 0, "merged_count": len(curated)})


@app.route("/api/universe-refresh", methods=["POST", "GET"])
def universe_refresh():
    threading.Thread(target=refresh_dynamic_universe, daemon=True).start()
    return jsonify({"ok": True, "message": "유니버스 갱신 시작"})


ADMIN_KEY = os.environ.get("ADMIN_KEY", "vipasset1004")  # 관리자 키 (환경변수 또는 기본값)

@app.route("/api/scan")
def scan():
    import queue as _queue
    # 관리자 모드만 수동 스캔 가능 (일반 사용자는 자동 스캔만)
    admin = request.args.get("admin", "")
    if admin != ADMIN_KEY:
        def blocked():
            yield f"data: {_dumps({'type': 'error', 'message': '수동 스캔 비활성. 자동 스캔: 16:00, 21:00 (KST)'})}\n\n"
        return Response(blocked(), mimetype="text/event-stream")

    market = request.args.get("market", "KR")
    top_n_raw = request.args.get("top_n", "150")
    min_score = float(request.args.get("min_score", "40"))
    use_auto = (top_n_raw == "auto")
    top_n = 9999 if use_auto else int(top_n_raw)

    if scan_state["running"]:
        def already_running():
            yield f"data: {_dumps({'type': 'error', 'message': '이미 스캔이 진행 중입니다'})}\n\n"
        return Response(already_running(), mimetype="text/event-stream")

    msg_queue = _queue.Queue()

    def scan_worker():
        scan_state["running"] = True
        results = []
        try:
            tickers = get_merged_universe() if use_auto else get_fallback_tickers(top_n)
            total = len(tickers)

            msg_queue.put({'type': 'status', 'message': f'1단계: {total}개 종목 데이터 병렬 수집 중', 'progress': 2, 'total': total})

            index_df      = fetch_index("^KS11")
            base_regime   = get_market_regime(index_df)
            # Distribution Day 보정: 4+이면 NEUTRAL 강등, 6+이면 BEAR 강등
            dist_days = count_distribution_days(index_df)
            if base_regime == "BULL" and dist_days >= 6:
                market_regime = "BEAR"
            elif base_regime == "BULL" and dist_days >= 4:
                market_regime = "NEUTRAL"
            else:
                market_regime = base_regime
            print(f"[scan] regime={market_regime} (base={base_regime}, distribution_days={dist_days})")

            # [1단계] OHLCV 병렬 수집 (15 worker)
            def _fetch_progress(done, t):
                pct = int(2 + (done / t) * 28)  # 2% → 30%
                msg_queue.put({'type': 'progress', 'current': f'데이터 수집 {done}/{t}', 'index': done, 'total': total, 'progress': pct, 'found': 0})
            df_map = fetch_daily_batch(tickers, max_workers=15, is_korean=True, progress_cb=_fetch_progress)

            # [2단계] 1차 필터 (신고가 후보만 압축, detect_new_high와 동기화)
            candidates = [t for t in tickers if is_near_high_candidate(df_map.get(t["ticker"]))]
            n_cand = len(candidates)
            msg_queue.put({'type': 'status', 'message': f'2단계: {total}→{n_cand}개 후보 (외국인/기관 수급 수집)', 'progress': 32, 'total': n_cand})

            # [2.5단계] 수급 데이터 병렬 수집 (외국인/기관, 후보만)
            supply_map = fetch_supply_batch(candidates, max_workers=15)
            msg_queue.put({'type': 'status', 'message': f'3단계: {n_cand}개 정밀 분석', 'progress': 38, 'total': n_cand})

            # [3단계] 후보만 정밀 분석
            for i, t in enumerate(candidates):
                ticker = t["ticker"]
                name = t["name"]
                progress = int(38 + (i / max(n_cand,1)) * 57)  # 38% → 95%

                msg_queue.put({'type': 'progress', 'current': name, 'index': i+1, 'total': n_cand, 'progress': progress, 'found': len(results)})

                try:
                    result = analyze_stock(ticker, name, is_korean=True, index_df=index_df,
                                           themes=t.get("themes", []), market_regime=market_regime,
                                           df=df_map.get(ticker), supply_data=supply_map.get(ticker))
                    if result and result["score"] >= min_score:
                        results.append(result)
                        msg_queue.put({'type': 'result', 'data': result})
                except Exception:
                    pass
            supply_map = None

            df_map = None  # 메모리 회수
            gc.collect()

            results.sort(key=lambda x: (0 if x["type"] == "BREAKOUT_BOTTOM" else
                                        1 if x["type"] == "PULLBACK_REBREAK" else 2,
                                        -x["score"]))
            results = post_process_results(results)
            results = enrich_with_predict(results)
            results = add_entry_scores(results)
            results = add_multibagger_scores(results)

            breadth_scan  = round(len(results) / total * 100, 1) if total > 0 else 0
            naver_breadth = fetch_naver_breadth()
            prev_cache    = _cache_get(CACHE_KEY)
            prev_breadth  = prev_cache.get("breadth_scan", 0) if prev_cache else 0
            breadth_trend = ("▲" if breadth_scan > prev_breadth else
                             "▼" if breadth_scan < prev_breadth else "─")

            scan_date = datetime.now().isoformat()
            cache_data = {
                "scan_date"     : scan_date,
                "total_found"   : len(results),
                "total_scanned" : total,
                "breadth_scan"  : breadth_scan,
                "naver_breadth" : naver_breadth,
                "breadth_trend" : breadth_trend,
                "results"       : results,
            }

            if prev_cache:
                _cache_set(CACHE_PREV_KEY, prev_cache)
            _cache_set(CACHE_KEY, cache_data)

            try:
                saved_ps = save_prime_s_signals(results, scan_date)
                print(f"[live_db] PRIME-S {saved_ps}건 저장")
            except Exception as e:
                print(f"[live_db] 저장 오류: {e}")

            msg_queue.put({'type': 'done', 'total': len(results), 'scan_date': scan_date,
                           'results': results, 'breadth_scan': breadth_scan,
                           'naver_breadth': naver_breadth, 'breadth_trend': breadth_trend})

        except Exception as e:
            msg_queue.put({'type': 'error', 'message': str(e)})
        finally:
            scan_state["running"] = False
            msg_queue.put(None)  # sentinel
            gc.collect()

    t = threading.Thread(target=scan_worker, daemon=True)
    t.start()

    def generate():
        while True:
            try:
                msg = msg_queue.get(timeout=8)
                if msg is None:
                    break
                yield f"data: {_dumps(msg)}\n\n"
                if msg.get("type") in ("done", "error"):
                    break
            except _queue.Empty:
                yield ": keep-alive\n\n"  # Render 프록시 연결 유지

    return Response(generate(), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


# ══════════════════════════════════════════════════════════
# DB 무결성 체크
# ══════════════════════════════════════════════════════════

def check_db_integrity():
    """backtest_history.csv 중복 검사 — 이상 시 로그 출력"""
    db_path = "backtest_history.csv"
    if not os.path.exists(db_path):
        return True
    try:
        df = pd.read_csv(db_path, encoding="utf-8-sig")
        total  = len(df)
        unique = df.drop_duplicates(subset=["signal_date", "ticker"]).shape[0]
        if total != unique:
            print(f"[DB무결성] ⚠️ 중복 발견: 전체 {total}행 / 고유 {unique}행 / 중복 {total - unique}건")
            return False
        print(f"[DB무결성] ✅ 정상: {total}행 중복 없음")
        return True
    except Exception as e:
        print(f"[DB무결성] 오류: {e}")
        return False


# ══════════════════════════════════════════════════════════
# 자동 스캔 스케줄러 — 3개 (21:00 풀/07:00 빠른/16:00 수익)
# ══════════════════════════════════════════════════════════

def auto_scan_full():
    """매일 21:00 ⭐ 풀 스캔 — PRIME-S 발견 + K-NN + 진입 점수 + DB 저장"""
    print(f"[21:00 풀스캔] 시작: {datetime.now()}")
    if scan_state["running"]:
        print("[21:00 풀스캔] 이미 스캔 중 → 건너뜀")
        return
    scan_state["running"] = True
    results = []
    try:
        tickers  = get_merged_universe()
        index_df = fetch_index("^KS11")
        base_regime = get_market_regime(index_df)
        dist_days = count_distribution_days(index_df)
        if base_regime == "BULL" and dist_days >= 6:   market_regime = "BEAR"
        elif base_regime == "BULL" and dist_days >= 4: market_regime = "NEUTRAL"
        else: market_regime = base_regime
        print(f"[21:00 풀스캔] regime={market_regime} (base={base_regime}, dist_days={dist_days})")

        # [1단계] OHLCV 병렬 수집
        print(f"[21:00 풀스캔] 1단계: {len(tickers)}종목 병렬 fetch")
        df_map = fetch_daily_batch(tickers, max_workers=15, is_korean=True)

        # [2단계] 1차 필터 (신고가 후보만)
        candidates = [t for t in tickers if is_near_high_candidate(df_map.get(t["ticker"]))]
        print(f"[21:00 풀스캔] 2단계: {len(tickers)}→{len(candidates)}개 후보")

        # [2.5단계] 수급 데이터 병렬 수집
        supply_map = fetch_supply_batch(candidates, max_workers=15)
        print(f"[21:00 풀스캔] 2.5단계: 수급 {len(supply_map)}개 수집")

        # [3단계] 정밀 분석
        for t in candidates:
            try:
                r = analyze_stock(t["ticker"], t["name"], is_korean=True,
                                  index_df=index_df, themes=t.get("themes", []),
                                  market_regime=market_regime,
                                  df=df_map.get(t["ticker"]),
                                  supply_data=supply_map.get(t["ticker"]))
                if r and r["score"] >= 40:
                    results.append(r)
            except:
                pass
        df_map = None
        supply_map = None
        gc.collect()

        results.sort(key=lambda x: (0 if x["type"] == "BREAKOUT_BOTTOM" else
                                    1 if x["type"] == "PULLBACK_REBREAK" else 2,
                                    -x["score"]))
        results = post_process_results(results)
        results = enrich_with_predict(results)
        results = add_entry_scores(results)

        breadth_scan  = round(len(results) / len(tickers) * 100, 1) if tickers else 0
        naver_breadth = fetch_naver_breadth()
        prev_cache    = _cache_get(CACHE_KEY)
        prev_breadth  = prev_cache.get("breadth_scan", 0) if prev_cache else 0
        breadth_trend = ("▲" if breadth_scan > prev_breadth else
                         "▼" if breadth_scan < prev_breadth else "─")

        scan_date  = datetime.now().isoformat()
        cache_data = {
            "scan_date"    : scan_date,
            "total_found"  : len(results),
            "total_scanned": len(tickers),
            "breadth_scan" : breadth_scan,
            "naver_breadth": naver_breadth,
            "breadth_trend": breadth_trend,
            "results"      : results,
        }
        if prev_cache:
            _cache_set(CACHE_PREV_KEY, prev_cache)
        _cache_set(CACHE_KEY, cache_data)

        saved_ps = save_prime_s_signals(results, scan_date)
        ps_cnt = sum(1 for r in results if r.get("is_prime_s"))
        pm_cnt = sum(1 for r in results if r.get("is_prime") and not r.get("is_prime_s"))
        sb_cnt = sum(1 for r in results if r.get("entry_action") == "STRONG_BUY")
        b_cnt  = sum(1 for r in results if r.get("entry_action") == "BUY")
        try:
            con = sqlite3.connect(LIVE_DB)
            con.execute("""INSERT INTO scan_log
                (scan_date, total_found, prime_s_count, prime_count, breadth_scan)
                VALUES (?,?,?,?,?)""",
                (scan_date, len(results), ps_cnt, pm_cnt, breadth_scan))
            con.commit(); con.close()
        except:
            pass

        print(f"[21:00 풀스캔] 완료 → {len(results)}개 "
              f"/ PRIME-S {ps_cnt} / PRIME {pm_cnt} "
              f"/ STRONG_BUY {sb_cnt} / BUY {b_cnt}")
        check_db_integrity()
    except Exception as e:
        print(f"[21:00 풀스캔] 오류: {e}")
    finally:
        scan_state["running"] = False
        gc.collect()


def auto_scan_quick():
    """매일 07:00 — 어제 PRIME-S 종목 재스캔 (갭 체크 + 진입 점수 갱신)"""
    print(f"[07:00 빠른스캔] 시작: {datetime.now()}")
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        con = sqlite3.connect(LIVE_DB)
        cur = con.cursor()
        # 가장 최근 스캔일의 PRIME-S 종목
        cur.execute("""
            SELECT ticker, name FROM prime_s_signals
            WHERE live_date = (
                SELECT MAX(live_date) FROM prime_s_signals WHERE live_date < ?
            )
            ORDER BY score DESC
        """, (today,))
        rows = cur.fetchall()
        con.close()

        if not rows:
            print("[07:00 빠른스캔] 어제 PRIME-S 없음 → 건너뜀")
            return

        print(f"[07:00 빠른스캔] {len(rows)}개 PRIME-S 갭 체크")
        cache = _cache_get(CACHE_KEY) or {}
        results = cache.get("results", [])

        updated = 0
        for ticker_str, name in rows:
            try:
                gap = check_gap_at_open(ticker_str)
                # 캐시에서 해당 종목 찾아 갭 반영 + 진입 점수 재계산
                for i, r in enumerate(results):
                    if str(r.get("ticker")) == str(ticker_str):
                        r = dict(r)
                        r["gap_status"] = gap
                        es = calculate_entry_score(r)
                        r["entry_score"]     = es["score"]
                        r["entry_action"]    = es["action"]
                        r["entry_breakdown"] = es["breakdown"]
                        results[i] = r
                        updated += 1
                        break
                print(f"  {name}({ticker_str}): {gap['status']} / {gap['reason']}")
            except Exception as e:
                print(f"  [갭체크 오류] {ticker_str}: {e}")
            time.sleep(0.3)

        if updated > 0:
            cache["results"]         = results
            cache["quick_scan_time"] = datetime.now().isoformat()
            _cache_set(CACHE_KEY, cache)

        sb = sum(1 for r in results if r.get("entry_action") == "STRONG_BUY")
        b  = sum(1 for r in results if r.get("entry_action") == "BUY")
        print(f"[07:00 빠른스캔] 완료 → 갱신 {updated}개 "
              f"/ STRONG_BUY {sb} / BUY {b}")
    except Exception as e:
        print(f"[07:00 빠른스캔] 오류: {e}")


scheduler = BackgroundScheduler(timezone=pytz.timezone("Asia/Seoul"))
# 16:00 — 정규장 종가 직후 (당일 종가 반영)
scheduler.add_job(refresh_dynamic_universe, "cron", hour=15, minute=50, id="universe_pm")
scheduler.add_job(auto_scan_full,           "cron", hour=16, minute=0,  id="scan_pm")
# 21:00 — 시간외 단일가 종료 후 (외국인/기관 매매 반영, 다음날 매수 결정)
scheduler.add_job(refresh_dynamic_universe, "cron", hour=20, minute=50, id="universe_evening")
scheduler.add_job(auto_scan_full,           "cron", hour=21, minute=0,  id="scan_evening")
# 기존 07:00 퀵스캔 제거 — 21:00 결과로 다음날 매수 결정 충분
# SKIP_SCHEDULER=1 (GitHub Actions/CLI 모드) 시 스케줄러 안 띄움
if os.environ.get("SKIP_SCHEDULER") != "1":
    scheduler.start()


# ══════════════════════════════════════════════════════════
# 라이브 추적 시스템 — PRIME-S 종목 저장 + 실제 수익 추적
# ══════════════════════════════════════════════════════════
LIVE_DB = "prime_s_live.db"

def _init_live_db():
    con = sqlite3.connect(LIVE_DB)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prime_s_signals (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            live_date    TEXT NOT NULL,
            ticker       TEXT NOT NULL,
            name         TEXT,
            sub_label    TEXT,
            grade        TEXT,
            score        REAL,
            entry_price  REAL,
            stop_loss    REAL,
            target_1     REAL,
            target_2     REAL,
            market_regime TEXT,
            rvol         REAL,
            rsi          REAL,
            predict_rec  TEXT,
            predict_win20 REAL,
            predict_avg20 REAL,
            ret_5d       REAL,
            ret_10d      REAL,
            ret_20d      REAL,
            ret_60d      REAL,
            hit_stop     INTEGER DEFAULT 0,
            hit_tp1      INTEGER DEFAULT 0,
            hit_tp2      INTEGER DEFAULT 0,
            updated_at   TEXT,
            UNIQUE(live_date, ticker)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scan_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date  TEXT NOT NULL,
            total_found INTEGER,
            prime_s_count INTEGER,
            prime_count  INTEGER,
            breadth_scan REAL
        )
    """)
    con.commit()
    con.close()

_init_live_db()


def save_prime_s_signals(results, scan_date_str):
    """스캔 완료 후 PRIME-S 종목을 live DB에 저장"""
    con = sqlite3.connect(LIVE_DB)
    cur = con.cursor()
    today = scan_date_str[:10]
    saved = 0
    for r in results:
        if not r.get("is_prime_s"):
            continue
        p = r.get("predict") or {}
        try:
            cur.execute("""
                INSERT OR IGNORE INTO prime_s_signals
                (live_date, ticker, name, sub_label, grade, score,
                 entry_price, stop_loss, target_1, target_2,
                 market_regime, rvol, rsi,
                 predict_rec, predict_win20, predict_avg20,
                 updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                today,
                str(r.get("ticker","")),
                r.get("name",""),
                r.get("prime_s_sub",""),
                r.get("grade",""),
                r.get("score",0),
                r.get("entry_price"),
                r.get("stop_loss"),
                r.get("target_1"),
                r.get("target_2"),
                r.get("market_regime",""),
                r.get("rvol",0),
                r.get("rsi",0),
                p.get("rec"),
                p.get("win20"),
                p.get("avg20"),
                datetime.now().isoformat(),
            ))
            saved += 1
        except Exception as e:
            print(f"[live_db] save error {r.get('ticker')}: {e}")
    con.commit()
    con.close()
    return saved


def update_live_returns():
    """보유 종목 실제 수익률 자동 업데이트 (매일 21:30 실행)"""
    import yfinance as yf
    con = sqlite3.connect(LIVE_DB)
    cur = con.cursor()
    cur.execute("""
        SELECT id, ticker, live_date, entry_price
        FROM prime_s_signals
        WHERE ret_20d IS NULL OR ret_60d IS NULL
    """)
    rows = cur.fetchall()
    for row_id, ticker, live_date, entry_price in rows:
        if not entry_price:
            continue
        try:
            t = ticker if ticker.endswith(".KS") else ticker + ".KS"
            df = yf.download(t, period="90d", progress=False, auto_adjust=True)
            if df is None or len(df) < 2:
                continue
            start_dt = datetime.strptime(live_date, "%Y-%m-%d").date()
            # 신호일 이후 종가만 추출
            df_fwd = df[df.index.date > start_dt]
            if len(df_fwd) == 0:
                continue
            closes = df_fwd["Close"].values.flatten()
            e = float(entry_price)
            def fwd_ret(n):
                return round((float(closes[n-1]) / e - 1) * 100, 2) if len(closes) >= n else None
            # 손절/익절 달성 여부
            stop_p = e * 0.93
            tp1_p  = e * 1.20
            tp2_p  = e * 1.50
            hit_stop = any(float(c) <= stop_p for c in closes[:20])
            hit_tp1  = any(float(c) >= tp1_p  for c in closes[:60])
            hit_tp2  = any(float(c) >= tp2_p  for c in closes[:60])
            cur.execute("""
                UPDATE prime_s_signals
                SET ret_5d=?, ret_10d=?, ret_20d=?, ret_60d=?,
                    hit_stop=?, hit_tp1=?, hit_tp2=?, updated_at=?
                WHERE id=?
            """, (fwd_ret(5), fwd_ret(10), fwd_ret(20), fwd_ret(60),
                  int(hit_stop), int(hit_tp1), int(hit_tp2),
                  datetime.now().isoformat(), row_id))
        except Exception as e:
            print(f"[live_db] update error {ticker}: {e}")
    con.commit()
    con.close()
    print(f"[live_db] 수익률 업데이트 완료: {len(rows)}건 시도")


@app.route("/live-tracking")
def live_tracking_page():
    """추적 현황 HTML 페이지"""
    con = sqlite3.connect(LIVE_DB)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM prime_s_signals ORDER BY live_date DESC, score DESC LIMIT 200")
    signals = [dict(r) for r in cur.fetchall()]
    cur.execute("""
        SELECT substr(live_date,1,7) as month,
            COUNT(*) as cnt,
            ROUND(AVG(CASE WHEN ret_20d IS NOT NULL THEN ret_20d END), 2) as avg_ret20,
            ROUND(AVG(CASE WHEN ret_20d IS NOT NULL AND ret_20d > 0 THEN 1.0 ELSE 0.0 END)*100, 1) as win_rate,
            SUM(hit_tp1) as tp1_hits, SUM(hit_stop) as stop_hits
        FROM prime_s_signals WHERE ret_20d IS NOT NULL
        GROUP BY month ORDER BY month DESC LIMIT 12
    """)
    monthly = [dict(r) for r in cur.fetchall()]
    con.close()

    rows_html = ""
    for s in signals:
        ret20 = s.get("ret_20d")
        ret_str = f"{ret20:+.1f}%" if ret20 is not None else "-"
        ret_color = "#22c55e" if ret20 and ret20 > 0 else ("#ef4444" if ret20 and ret20 < 0 else "#6b7a99")
        stop = "🛑" if s.get("hit_stop") else ""
        tp1  = "🎯" if s.get("hit_tp1") else ""
        rows_html += f"""<tr>
            <td>{s.get('live_date','')}</td>
            <td><a href="https://finance.naver.com/item/main.naver?code={s.get('ticker','')}" target="_blank" style="color:#38bdf8;text-decoration:none">{s.get('name','')} <span style="color:#4a5a7a;font-size:11px">{s.get('ticker','')}</span></a></td>
            <td style="color:#f59e0b">{s.get('grade','')}</td>
            <td>{s.get('score','')}</td>
            <td>{s.get('entry_price',''):,}</td>
            <td style="color:{ret_color};font-weight:700">{ret_str}</td>
            <td>{stop}{tp1}</td>
        </tr>"""

    monthly_html = ""
    for m in monthly:
        wr = m.get("win_rate") or 0
        ar = m.get("avg_ret20") or 0
        wc = "#22c55e" if wr >= 55 else ("#f59e0b" if wr >= 45 else "#ef4444")
        ac = "#22c55e" if ar > 0 else "#ef4444"
        monthly_html += f"""<tr>
            <td>{m.get('month','')}</td>
            <td>{m.get('cnt','')}</td>
            <td style="color:{ac};font-weight:700">{ar:+.1f}%</td>
            <td style="color:{wc};font-weight:700">{wr}%</td>
            <td>{m.get('tp1_hits',0)}</td>
            <td>{m.get('stop_hits',0)}</td>
        </tr>"""

    html = f"""<!DOCTYPE html><html lang="ko"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>추적 현황 — PRIME-S</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0a1628;color:#c8d8f0;font-family:-apple-system,sans-serif;padding:16px}}
h1{{font-size:20px;font-weight:800;color:#f0f4ff;margin-bottom:4px}}
.sub{{font-size:13px;color:#4a5a7a;margin-bottom:20px}}
.back{{display:inline-block;margin-bottom:16px;color:#38bdf8;font-size:13px;text-decoration:none;padding:6px 14px;border:1px solid #1a3050;border-radius:8px}}
h2{{font-size:15px;font-weight:700;color:#94a3b8;margin:20px 0 10px;border-left:3px solid #3b82f6;padding-left:10px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#0d1829;color:#4a5a7a;padding:8px 10px;text-align:left;border-bottom:1px solid #1a2540;font-weight:600}}
td{{padding:8px 10px;border-bottom:1px solid #0d1829}}
tr:hover td{{background:#0d1829}}
@media(max-width:600px){{table{{font-size:11px}} td,th{{padding:6px 6px}}}}
</style></head><body>
<a href="/" class="back">← 스크리너로 돌아가기</a>
<h1>📊 추적 현황</h1>
<div class="sub">PRIME-S 신호 실시간 수익 추적</div>

<h2>월별 성과</h2>
<table><thead><tr><th>월</th><th>건수</th><th>평균수익(20일)</th><th>승률</th><th>1차익절</th><th>손절</th></tr></thead>
<tbody>{monthly_html if monthly_html else '<tr><td colspan="6" style="color:#4a5a7a;text-align:center;padding:20px">데이터 없음 (수익률 업데이트 대기 중)</td></tr>'}</tbody></table>

<h2>신호 목록 (최근 200건)</h2>
<table><thead><tr><th>날짜</th><th>종목</th><th>등급</th><th>점수</th><th>진입가</th><th>20일수익</th><th>결과</th></tr></thead>
<tbody>{rows_html if rows_html else '<tr><td colspan="7" style="color:#4a5a7a;text-align:center;padding:20px">신호 없음</td></tr>'}</tbody></table>
</body></html>"""
    return html


@app.route("/api/live-tracking")
def live_tracking():
    """PRIME-S 라이브 추적 현황 조회"""
    con = sqlite3.connect(LIVE_DB)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    # 최근 90일 신호
    cur.execute("""
        SELECT * FROM prime_s_signals
        ORDER BY live_date DESC, score DESC
        LIMIT 200
    """)
    signals = [dict(r) for r in cur.fetchall()]
    # 월별 통계
    cur.execute("""
        SELECT
            substr(live_date,1,7) as month,
            COUNT(*) as n,
            ROUND(AVG(CASE WHEN ret_20d IS NOT NULL THEN ret_20d END), 2) as avg_ret20,
            ROUND(AVG(CASE WHEN ret_20d IS NOT NULL AND ret_20d > 0 THEN 1.0 ELSE 0.0 END)*100, 1) as win_rate,
            SUM(hit_tp1) as tp1_hits,
            SUM(hit_stop) as stop_hits
        FROM prime_s_signals
        WHERE ret_20d IS NOT NULL
        GROUP BY month
        ORDER BY month DESC
        LIMIT 12
    """)
    monthly = [dict(r) for r in cur.fetchall()]
    con.close()
    return jsonify({"signals": signals, "monthly": monthly})


@app.route("/api/live-tracking/update", methods=["POST"])
def trigger_live_update():
    """수동으로 수익률 업데이트 트리거"""
    threading.Thread(target=update_live_returns, daemon=True).start()
    return jsonify({"status": "started"})


# 매일 16:00 자동 수익률 업데이트 (장 마감 30분 후)
scheduler.add_job(update_live_returns, "cron", hour=16, minute=0, id="returns_update")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
