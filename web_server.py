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
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    raise TypeError(f"{type(o)} not serializable")

def _dumps(obj):
    return json.dumps(obj, ensure_ascii=False, default=_np_default)

from new_high_screener import (
    get_fallback_tickers, analyze_stock, fetch_index, get_market_regime,
    post_process_results, fetch_naver_breadth, enrich_with_predict,
    check_gap_at_open, calculate_entry_score, add_entry_scores,
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
CACHE_TTL      = 60 * 60 * 48
CACHE_PATH      = "newhigh_cache.json"
CACHE_PREV_PATH = "newhigh_cache_prev.json"


def _cache_get(key):
    if _redis:
        try:
            val = _redis.get(key)
            return json.loads(val) if val else None
        except Exception as e:
            print(f"[cache] get 오류: {e}")
    path = CACHE_PATH if key == CACHE_KEY else CACHE_PREV_PATH
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
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
    path = CACHE_PATH if key == CACHE_KEY else CACHE_PREV_PATH
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(_dumps(value))
    except Exception as e:
        print(f"[cache] 파일 저장 오류: {e}")


scan_state = {"running": False}


@app.route("/")
def index():
    return send_file("index.html")


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


@app.route("/api/scan")
def scan():
    market = request.args.get("market", "KR")
    top_n = int(request.args.get("top_n", "150"))
    min_score = float(request.args.get("min_score", "40"))

    if scan_state["running"]:
        def already_running():
            yield f"data: {_dumps({'type': 'error', 'message': '이미 스캔이 진행 중입니다'})}\n\n"
        return Response(already_running(), mimetype="text/event-stream")

    def generate():
        scan_state["running"] = True
        results = []
        try:
            tickers = get_fallback_tickers(top_n)
            total = len(tickers)

            yield f"data: {_dumps({'type': 'status', 'message': f'{total}개 종목 신고가 분석 시작', 'progress': 2, 'total': total})}\n\n"

            index_df      = fetch_index("^KS11")
            market_regime = get_market_regime(index_df)

            for i, t in enumerate(tickers):
                ticker = t["ticker"]
                name = t["name"]
                progress = int(5 + (i / total) * 90)

                if i % 5 == 0:
                    yield f"data: {_dumps({'type': 'progress', 'current': name, 'index': i+1, 'total': total, 'progress': progress, 'found': len(results)})}\n\n"

                try:
                    result = analyze_stock(ticker, name, is_korean=True, index_df=index_df, themes=t.get("themes", []), market_regime=market_regime)
                    if result and result["score"] >= min_score:
                        results.append(result)
                        yield f"data: {_dumps({'type': 'result', 'data': result})}\n\n"
                except Exception as e:
                    pass

                gc.collect()
                time.sleep(0.08)

            # 정렬 + RS/유동성 랭킹 + combo_ok
            results.sort(key=lambda x: (0 if x["type"] == "BREAKOUT_BOTTOM" else
                                        1 if x["type"] == "PULLBACK_REBREAK" else 2,
                                        -x["score"]))
            results = post_process_results(results)
            results = enrich_with_predict(results)
            results = add_entry_scores(results)

            # ── (A) 신고가 확산도 (Market Breadth) ──
            breadth_scan  = round(len(results) / total * 100, 1) if total > 0 else 0
            naver_breadth = fetch_naver_breadth()   # 네이버 전체시장 신고가 수
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

            # PRIME-S 종목 라이브 DB에 저장
            try:
                saved_ps = save_prime_s_signals(results, scan_date)
                print(f"[live_db] PRIME-S {saved_ps}건 저장")
            except Exception as e:
                print(f"[live_db] 저장 오류: {e}")

            yield f"data: {_dumps({'type': 'done', 'total': len(results), 'scan_date': scan_date, 'results': results, 'breadth_scan': breadth_scan, 'naver_breadth': naver_breadth, 'breadth_trend': breadth_trend})}\n\n"

        except Exception as e:
            yield f"data: {_dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            scan_state["running"] = False
            gc.collect()

    return Response(generate(), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


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
        tickers  = get_fallback_tickers(500)
        index_df = fetch_index("^KS11")
        market_regime = get_market_regime(index_df)

        for t in tickers:
            try:
                r = analyze_stock(t["ticker"], t["name"], is_korean=True,
                                  index_df=index_df, themes=t.get("themes", []),
                                  market_regime=market_regime)
                if r and r["score"] >= 40:
                    results.append(r)
            except:
                pass
            time.sleep(0.08)
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
scheduler.add_job(auto_scan_full,  "cron", hour=21, minute=0,  id="scan_full")
scheduler.add_job(auto_scan_quick, "cron", hour=7,  minute=0,  id="scan_quick")
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
