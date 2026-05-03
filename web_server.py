"""
신고가 지속 상승 스크리너 웹 서버
Flask + SSE 실시간 스캔
"""

from flask import Flask, Response, request, send_file, jsonify
import json
import time
import threading
import gc
import os
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

from new_high_screener import (
    get_fallback_tickers, analyze_stock, fetch_index
)

app = Flask(__name__)

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
            _redis.set(key, json.dumps(value, ensure_ascii=False), ex=CACHE_TTL)
            return
        except Exception as e:
            print(f"[cache] set 오류: {e}")
    path = CACHE_PATH if key == CACHE_KEY else CACHE_PREV_PATH
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
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
            yield f"data: {json.dumps({'type': 'error', 'message': '이미 스캔이 진행 중입니다'})}\n\n"
        return Response(already_running(), mimetype="text/event-stream")

    def generate():
        scan_state["running"] = True
        results = []
        try:
            tickers = get_fallback_tickers(top_n)
            total = len(tickers)

            yield f"data: {json.dumps({'type': 'status', 'message': f'{total}개 종목 신고가 분석 시작', 'progress': 2, 'total': total}, ensure_ascii=False)}\n\n"

            index_df = fetch_index("^KS11")

            for i, t in enumerate(tickers):
                ticker = t["ticker"]
                name = t["name"]
                progress = int(5 + (i / total) * 90)

                if i % 5 == 0:
                    yield f"data: {json.dumps({'type': 'progress', 'current': name, 'index': i+1, 'total': total, 'progress': progress, 'found': len(results)}, ensure_ascii=False)}\n\n"

                try:
                    result = analyze_stock(ticker, name, is_korean=True, index_df=index_df)
                    if result and result["score"] >= min_score:
                        results.append(result)
                        yield f"data: {json.dumps({'type': 'result', 'data': result}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    pass

                gc.collect()
                time.sleep(0.08)

            # 정렬: 추세시작형 우선, 점수 내림차순
            results.sort(key=lambda x: (0 if x["type"] == "START" else 1, -x["score"]))

            scan_date = datetime.now().isoformat()
            cache_data = {
                "scan_date": scan_date,
                "total_found": len(results),
                "results": results,
            }

            prev = _cache_get(CACHE_KEY)
            if prev:
                _cache_set(CACHE_PREV_KEY, prev)
            _cache_set(CACHE_KEY, cache_data)

            yield f"data: {json.dumps({'type': 'done', 'total': len(results), 'scan_date': scan_date, 'results': results}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            scan_state["running"] = False
            gc.collect()

    return Response(generate(), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


# ── 자동 스캔 스케줄러 (매일 21시) ──────────────────────
def auto_scan_job():
    print(f"[스케줄] 자동 스캔 시작: {datetime.now()}")
    if scan_state["running"]:
        print("[스케줄] 이미 스캔 중 → 건너뜀")
        return

    scan_state["running"] = True
    results = []
    try:
        tickers = get_fallback_tickers(150)
        index_df = fetch_index("^KS11")
        for t in tickers:
            try:
                r = analyze_stock(t["ticker"], t["name"], is_korean=True, index_df=index_df)
                if r and r["score"] >= 40:
                    results.append(r)
            except:
                pass
            time.sleep(0.1)
            gc.collect()

        results.sort(key=lambda x: (0 if x["type"] == "START" else 1, -x["score"]))

        cache_data = {
            "scan_date": datetime.now().isoformat(),
            "total_found": len(results),
            "results": results,
        }
        prev = _cache_get(CACHE_KEY)
        if prev:
            _cache_set(CACHE_PREV_KEY, prev)
        _cache_set(CACHE_KEY, cache_data)
        print(f"[스케줄] 완료 → {len(results)}개 발견")
    except Exception as e:
        print(f"[스케줄] 오류: {e}")
    finally:
        scan_state["running"] = False
        gc.collect()


scheduler = BackgroundScheduler(timezone=pytz.timezone("Asia/Seoul"))
scheduler.add_job(auto_scan_job, "cron", hour=21, minute=0)
scheduler.start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
