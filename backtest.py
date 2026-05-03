# -*- coding: utf-8 -*-
"""
신고가 스크리너 히스토리컬 백테스트 v2
- 과거 신호 발생일을 찾아 forward 수익률 측정
- 전진 예측 편향(look-ahead bias) 없음
- 추가: regime 추적, 페널티 기록, drawdown, CORE/WATCH/EARLY 분류
"""
import sys, io, json, time, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from new_high_screener import (
    calc_indicators, score_trend_template, score_vcp,
    score_breakout_supply, score_holding_rs, calc_penalty, grade,
    fetch_index, HIGH_52W, BOX_WIN, classify_position,
    get_fallback_tickers, get_market_regime,
)

# ── 설정 ──
TOP_N               = 150
TEST_START_DAYS_AGO = 1500
MIN_FWD_DAYS        = 20
FWD_WINDOWS         = [5, 10, 20]

# 이전 시스템 기준 (페널티 강화 전) 비교용 기록 수동 입력
PREV_STATS = {
    "total": 1065,
    "win_20d": 49,   # % (추정)
    "avg_20d": 2.1,  # % (추정)
    "pct_50": 13,    # 60일 +50% 달성 %
}


def grade_tier(g):
    """S/A+/A/B → CORE/WATCH/EARLY/BASIC"""
    return {"S": "CORE", "A+": "CORE", "A": "WATCH", "B": "EARLY"}.get(g, "BASIC")


def is_prime_signal(grade_val, is_ath, regime):
    return bool(grade_val in ("S", "A+", "A") and is_ath and regime == "BULL")


def get_regime_at(idx_close_all, idx_dates, signal_date):
    """신호 발생 시점의 시장 regime 계산"""
    mask = idx_dates <= signal_date
    if mask.sum() < 200:
        return "NEUTRAL"
    c = idx_close_all[mask][-200:]
    ma200 = float(np.mean(c))
    ma50  = float(np.mean(c[-50:])) if len(c) >= 50 else ma200
    ret20 = (c[-1] / c[-20] - 1) * 100 if len(c) >= 20 else 0
    above_200 = c[-1] > ma200
    above_50  = c[-1] > ma50
    if above_200 and above_50 and ret20 > -5:
        return "BULL"
    elif above_200:
        return "NEUTRAL"
    else:
        return "BEAR"


def find_signals(df_full, idx_df):
    """전체 히스토리에서 신호 탐색 + regime/penalty/drawdown 캡처"""
    signals = []
    close_all = df_full["Close"].values
    high_all  = df_full["High"].values
    dates_all = df_full.index

    # 지수 데이터 준비
    idx_close = idx_df["Close"].values if idx_df is not None else None
    idx_dates = idx_df.index if idx_df is not None else None

    test_start_idx = max(HIGH_52W + 20, len(df_full) - TEST_START_DAYS_AGO)
    last_signal_idx = -99

    for i in range(test_start_idx, len(df_full) - MIN_FWD_DAYS):
        ws = max(0, i - HIGH_52W)
        prev_52h = float(np.max(high_all[ws:i]))
        if high_all[i] < prev_52h * 0.999:
            continue
        if i - last_signal_idx < 10:
            continue

        df_s = df_full.iloc[:i+1].copy()
        if len(df_s) < 200:
            continue
        df_s = calc_indicators(df_s)
        if df_s is None:
            continue

        avg_tv = float(df_s["tv"].iloc[-20:].mean()) if len(df_s) >= 20 else 0
        if avg_tv < 500_000_000:
            continue

        c  = df_s["Close"].values
        h  = df_s["High"].values
        lo = df_s["Low"].values
        v  = df_s["Volume"].values
        tv = df_s["tv"].values

        we = len(c)
        tv_20  = tv[max(0, we-20):we]
        avg_tv_local = float(np.mean(tv_20)) if len(tv_20) else 1.0
        rvol   = float(tv[-1]) / avg_tv_local if avg_tv_local > 0 else 1.0

        box_s  = max(0, we - BOX_WIN)
        prev_box_h = float(np.max(h[box_s:we-1])) if we-1 > box_s else 0
        first_breakout = prev_box_h < prev_52h * 0.97

        older_tv   = tv[max(0, we-70):max(0, we-20)]
        older_base = tv[max(0, we-100):max(0, we-70)]
        first_vol_surge = True
        if len(older_tv) > 0 and len(older_base) > 0:
            ob = float(np.mean(older_base))
            if ob > 0 and float(np.max(older_tv)) / ob >= 1.8:
                first_vol_surge = False

        retest = False
        ref_h = h[-1]
        look  = min(25, we - 1)
        sub_h = h[we-look-1 : we-1]
        sub_l = c[we-look-1 : we-1]
        if len(sub_h) > 3 and not first_breakout:
            pt = float(np.max(sub_h))
            ms = float(np.min(sub_l))
            pb = (pt - ms) / pt * 100
            if pt >= prev_52h * 0.99 and 3.0 <= pb <= 10.0:
                retest = True

        all_time_high = float(np.max(h))
        is_ath = float(h[-1]) >= all_time_high * 0.999

        nh = {
            "found": True, "days_ago": 0,
            "breakout_price": prev_52h,
            "breakout_high": float(h[-1]),
            "breakout_tv": float(tv[-1]),
            "avg_tv": avg_tv_local,
            "rvol": rvol,
            "first_breakout": first_breakout,
            "first_vol_surge": first_vol_surge,
            "retest": retest,
            "is_ath": is_ath,
        }

        rsi_v = float(df_s["rsi"].iloc[-1]) if "rsi" in df_s.columns else 50
        pos_type, pos_pct, ret_20d, above_30 = classify_position(df_s, nh)
        if pos_type == "OVERHEATED":
            continue

        ret_5d = (c[-1]/c[-6]-1)*100 if len(c)>=6 else 0

        s3, passed, failed, cnt = score_trend_template(df_s, pos_pct)
        s4, d4 = score_vcp(df_s, nh)
        s5, d5 = score_breakout_supply(nh)
        s6, d6 = score_holding_rs(df_s, nh, idx_df, pos_type=pos_type)
        type_bonus = {"PULLBACK_REBREAK":4,"TREND_CONTINUE":0}.get(pos_type, 0)
        pen, pen_r = calc_penalty(df_s, pos_type, rvol=nh["rvol"], days_ago=nh["days_ago"])

        stage1_ok = False
        if len(df_s) >= 126:
            h120 = float(df_s["High"].iloc[-120:].max())
            h126 = float(df_s["High"].iloc[-126:].max())
            cur_p = float(df_s["Close"].iloc[-1])
            stage1_ok = (h120 > 0 and cur_p/h120 <= 1.15 and
                         h126 > 0 and cur_p/h126 <= 1.25)
        vc_confirmed = bool(d4.get("vc_confirmed", False))
        is_ath_local = nh.get("is_ath", False)

        # Regime 계산 (백테스트용)
        sig_date_ts = pd.Timestamp(dates_all[i].date())
        regime_local = "BULL"
        if idx_df is not None and idx_df.index is not None:
            try:
                regime_local = get_regime_at(
                    idx_df["Close"].values, idx_df.index, sig_date_ts)
            except:
                pass

        context_bonus = (4 if stage1_ok else 0) + (3 if vc_confirmed else 0)
        if is_ath_local:                       context_bonus += 3
        if is_ath_local and regime_local == "BULL": context_bonus += 5

        raw   = s3 + s4 + s5 + s6 + type_bonus + context_bonus
        final = max(0, min(100, raw - pen))
        g_val, gl = grade(final, pos_type)
        if g_val == "D":
            continue

        # ── regime ──
        sig_date = pd.Timestamp(dates_all[i].date())
        regime = "BULL"
        if idx_close is not None and idx_dates is not None:
            regime = get_regime_at(idx_close, idx_dates, sig_date)

        # ── forward returns ──
        fwd = {}
        for fw in FWD_WINDOWS:
            if i + fw < len(close_all):
                fwd[f"ret_{fw}d"] = round((close_all[i+fw]/close_all[i]-1)*100, 2)
            else:
                fwd[f"ret_{fw}d"] = None

        max_fw = min(60, len(close_all) - i - 1)
        if max_fw > 0:
            fwd["ret_max"] = round((float(np.max(close_all[i+1:i+max_fw+1]))/close_all[i]-1)*100, 2)
        else:
            fwd["ret_max"] = None

        # ── max drawdown (신호 후 60일 이내) ──
        dd_fw = min(60, len(close_all) - i - 1)
        if dd_fw > 0:
            future_closes = close_all[i+1:i+dd_fw+1]
            peak = close_all[i]
            mdd = 0.0
            for fc in future_closes:
                peak = max(peak, fc)
                dd = (fc - peak) / peak * 100
                mdd = min(mdd, dd)
            fwd["mdd_60d"] = round(mdd, 2)
        else:
            fwd["mdd_60d"] = None

        # ── 매매규칙 시뮬레이션 (3 트랜치) ──
        # 트랜치1: -7% 손절 OR +20% 익절
        # 트랜치2: -7% 손절 OR +50% 익절
        # 트랜치3: 20일 보유 (단순) OR -7% 손절
        sim_fw = min(60, len(close_all) - i - 1)
        sim_ret = sim_mdd = None
        sim_stop = sim_tp1 = sim_tp2 = False
        if sim_fw > 0:
            entry     = close_all[i]
            sl_price  = entry * 0.93
            tp1_price = entry * 1.20
            tp2_price = entry * 1.50
            t1 = t2 = t3 = None
            sim_peak = entry
            sim_worst = 0.0
            for j in range(1, sim_fw + 1):
                cp = close_all[i + j]
                sim_peak = max(sim_peak, cp)
                sim_worst = min(sim_worst, (cp / sim_peak - 1) * 100)
                if t1 is None:
                    if cp >= tp1_price:  t1 = 0.20; sim_tp1 = True
                    elif cp <= sl_price: t1 = -0.07; sim_stop = True
                if t2 is None:
                    if cp >= tp2_price:  t2 = 0.50; sim_tp2 = True
                    elif cp <= sl_price: t2 = -0.07
                if t3 is None and j == 20:
                    t3 = (cp / entry - 1)
                if t1 is not None and t2 is not None and t3 is not None:
                    break
            last_ret = (close_all[i + sim_fw] / entry - 1)
            if t1 is None: t1 = last_ret
            if t2 is None: t2 = last_ret
            if t3 is None: t3 = last_ret
            sim_ret = round((t1 + t2 + t3) / 3 * 100, 2)
            sim_mdd = round(sim_worst, 2)
        fwd["sim_ret"]  = sim_ret
        fwd["sim_mdd"]  = sim_mdd
        fwd["sim_stop"] = sim_stop
        fwd["sim_tp1"]  = sim_tp1
        fwd["sim_tp2"]  = sim_tp2

        prime = is_prime_signal(g_val, is_ath, regime)

        # ── 슈퍼 알파 탐색용 추가 필드 ──
        # MA20 이격도 (±5% 이내 = 눌림 완성 구간)
        ma20_dist = None
        if "ma20" in df_s.columns:
            ma20_v = float(df_s["ma20"].iloc[-1])
            if ma20_v > 0:
                ma20_dist = round((c[-1] / ma20_v - 1) * 100, 2)

        # 거래량 드라이업 (최근 5일 < 20일 평균의 70%)
        vol_dryup = False
        if len(v) >= 20:
            v5  = float(np.mean(v[-5:]))
            v20 = float(np.mean(v[-20:]))
            if v20 > 0:
                vol_dryup = (v5 / v20) < 0.70

        # MA5 위 여부
        above_ma5 = False
        if "ma5" in df_s.columns:
            ma5_v = float(df_s["ma5"].iloc[-1])
            above_ma5 = c[-1] > ma5_v
        elif len(c) >= 5:
            above_ma5 = c[-1] > float(np.mean(c[-5:]))

        signals.append({
            "date":           str(dates_all[i].date()),
            "type":           pos_type,
            "grade":          g_val,
            "tier":           grade_tier(g_val),
            "is_prime":       prime,
            "score":          round(final, 1),
            "raw":            raw,
            "penalty":        pen,
            "pen_reasons":    pen_r,
            "regime":         regime,
            "rvol":           round(rvol, 2),
            "rsi":            round(rsi_v, 1),
            "first_breakout": first_breakout,
            "retest":         retest,
            "is_ath":         is_ath,
            "stage1_ok":      stage1_ok,
            "vc_confirmed":   vc_confirmed,
            "killer":         first_breakout and rvol >= 2.0,
            "ma20_dist":      ma20_dist,
            "vol_dryup":      vol_dryup,
            "above_ma5":      above_ma5,
            **fwd,
        })
        last_signal_idx = i

    return signals


# ── 통계 헬퍼 ────────────────────────────────────────────────

def stat(items, label="", fw_list=None):
    if fw_list is None:
        fw_list = FWD_WINDOWS
    n = len(items)
    if n == 0:
        return {"label": label, "N": 0}
    row = {"label": label, "N": n}
    for fw in fw_list:
        key = f"ret_{fw}d"
        vals = [x[key] for x in items if x.get(key) is not None]
        if vals:
            row[f"win_{fw}d"]  = round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1)
            row[f"avg_{fw}d"]  = round(sum(vals) / len(vals), 2)
        else:
            row[f"win_{fw}d"] = row[f"avg_{fw}d"] = None
    max_vals = [x["ret_max"] for x in items if x.get("ret_max") is not None]
    mdd_vals = [x["mdd_60d"] for x in items if x.get("mdd_60d") is not None]
    row["pct_50"]    = round(sum(1 for v in max_vals if v >= 50) / n * 100, 1) if max_vals else 0
    row["avg_mdd"]   = round(sum(mdd_vals) / len(mdd_vals), 2) if mdd_vals else 0
    row["worst_mdd"] = round(min(mdd_vals), 2) if mdd_vals else 0
    return row


def print_table(rows, title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    hdr = f"{'항목':<28} {'N':>5}  {'5일승률':>7} {'5일평균':>7}  {'10일승률':>8} {'10일평균':>8}  {'20일승률':>8} {'20일평균':>8}  {'50%달성':>7}  {'평균MDD':>7}"
    print(hdr)
    print("-" * 120)
    for r in rows:
        if r["N"] == 0:
            print(f"  {r['label']:<26} {'0':>5}")
            continue
        w5  = f"{r.get('win_5d',''):>6.1f}%" if r.get('win_5d') is not None else "     -"
        a5  = f"{r.get('avg_5d',''):>+6.2f}%" if r.get('avg_5d') is not None else "     -"
        w10 = f"{r.get('win_10d',''):>7.1f}%" if r.get('win_10d') is not None else "      -"
        a10 = f"{r.get('avg_10d',''):>+7.2f}%" if r.get('avg_10d') is not None else "      -"
        w20 = f"{r.get('win_20d',''):>7.1f}%" if r.get('win_20d') is not None else "      -"
        a20 = f"{r.get('avg_20d',''):>+7.2f}%" if r.get('avg_20d') is not None else "      -"
        p50 = f"{r.get('pct_50',0):>6.1f}%"
        mdd = f"{r.get('avg_mdd',0):>+6.2f}%"
        print(f"  {r['label']:<26} {r['N']:>5}  {w5} {a5}  {w10} {a10}  {w20} {a20}  {p50}  {mdd}")


def main():
    print("=" * 80)
    print(f"  신고가 스크리너 백테스트 v2  (상위 {TOP_N}종목 · 최근 1500거래일 · look-ahead 없음)")
    print("=" * 80)

    tickers_info = get_fallback_tickers(TOP_N)
    print(f"▶ {len(tickers_info)}종목 로드 완료")

    idx_df = fetch_index("^KS11", days=1500)
    print("▶ KOSPI 지수 로딩 완료\n")

    all_signals = []
    total = len(tickers_info)
    for i, t in enumerate(tickers_info):
        ticker = t["ticker"]
        name   = t["name"]
        themes = t.get("themes", [])
        print(f"  [{i+1:>3}/{total}] {name}({ticker}) ...", end=" ", flush=True)
        try:
            import yfinance as yf
            start = datetime.now() - timedelta(days=1500)
            df_full = None
            for sfx in [".KS", ".KQ"]:
                try:
                    sym = f"{ticker}{sfx}"
                    df = yf.download(sym, start=start, end=datetime.now(),
                                     auto_adjust=True, progress=False, timeout=15)
                    if df is not None and len(df) >= 200:
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        df = df.dropna(subset=["Close","Volume"])
                        df_full = df
                        break
                except:
                    continue

            if df_full is None:
                print("데이터없음")
                continue

            sigs = find_signals(df_full, idx_df)
            for s in sigs:
                s["ticker"] = ticker
                s["name"]   = name
                s["themes"] = themes
            all_signals.extend(sigs)
            print(f"신호 {len(sigs)}건  (누적 {len(all_signals)}건)")
            time.sleep(0.2)
        except Exception as e:
            print(f"오류: {e}")

    print(f"\n총 신호: {len(all_signals)}건\n")
    if not all_signals:
        print("신호 없음")
        return

    def _np(o):
        if isinstance(o, (np.bool_,)):      return bool(o)
        if isinstance(o, np.integer):       return int(o)
        if isinstance(o, np.floating):      return float(o)
        if isinstance(o, np.ndarray):       return o.tolist()
        raise TypeError(f"{type(o)} not serializable")

    with open("backtest_newhigh.json", "w", encoding="utf-8") as f:
        json.dump(all_signals, f, ensure_ascii=False, indent=2, default=_np)

    # ══════════════════════════════════════════════════════
    # 1. 이전 vs 새 시스템 비교
    # ══════════════════════════════════════════════════════
    new_all  = stat(all_signals, "새 시스템 (전체)")
    new_core = stat([s for s in all_signals if s["tier"] == "CORE"], "새 시스템 (CORE)")

    print("\n" + "="*80)
    print("  [1] 이전 vs 새 시스템 비교")
    print("="*80)
    print(f"  {'항목':<30} {'신호수':>6}  {'20일승률':>8}  {'20일평균':>8}  {'60일+50%':>8}  {'평균MDD':>8}")
    print("-"*80)

    prev = PREV_STATS
    print(f"  {'이전 시스템 (수동기록)':30} {prev['total']:>6}  {prev['win_20d']:>7.1f}%  {prev['avg_20d']:>+7.1f}%  {prev['pct_50']:>7.1f}%  {'N/A':>8}")

    def _row(lbl, items):
        r = stat(items)
        w20 = r.get("win_20d", 0) or 0
        a20 = r.get("avg_20d", 0) or 0
        p50 = r.get("pct_50", 0) or 0
        mdd = r.get("avg_mdd", 0) or 0
        print(f"  {lbl:<30} {r['N']:>6}  {w20:>7.1f}%  {a20:>+7.1f}%  {p50:>7.1f}%  {mdd:>+7.2f}%")

    _row("새 시스템 (전체)", all_signals)
    _row("새 시스템 (CORE만)", [s for s in all_signals if s["tier"]=="CORE"])
    _row("새 시스템 (페널티 없는 신호)", [s for s in all_signals if s["penalty"] == 0])
    _row("새 시스템 (페널티 받은 신호)", [s for s in all_signals if s["penalty"] > 0])

    # ══════════════════════════════════════════════════════
    # 2. CORE / WATCH / EARLY 등급별
    # ══════════════════════════════════════════════════════
    prime_sigs = [s for s in all_signals if s.get("is_prime")]
    tier_rows = [
        stat(prime_sigs,                                                       "🚀 PRIME (BULL×ATH)"),
        stat([s for s in all_signals if s["grade"] in ("S","A+") and not s.get("is_prime")], "CORE 비PRIME (S+A+)"),
        stat([s for s in all_signals if s["grade"] == "A+"],                   "  └ A+등급"),
        stat([s for s in all_signals if s["grade"] == "S"],                    "  └ S등급 (과열경고)"),
        stat([s for s in all_signals if s["grade"] == "A"],                    "WATCH (A)"),
        stat([s for s in all_signals if s["grade"] == "B"],                    "EARLY (B)"),
    ]
    print_table(tier_rows, "[2] PRIME / CORE / WATCH / EARLY 등급별 성과")

    # ══════════════════════════════════════════════════════
    # 3. 패턴별 (눌림 / 추세 / 킬러 등)
    # ══════════════════════════════════════════════════════
    pat_rows = [
        stat([s for s in all_signals if s["type"]=="PULLBACK_REBREAK"],  "눌림재돌파"),
        stat([s for s in all_signals if s["type"]=="TREND_CONTINUE"],    "추세진행"),
        stat([s for s in all_signals if s["type"]=="BREAKOUT_BOTTOM"],   "바닥탈출"),
        stat([s for s in all_signals if s.get("killer")],                "킬러(첫돌파+RVOL2x)"),
        stat([s for s in all_signals if s.get("retest")],                "Retest패턴"),
        stat([s for s in all_signals if s.get("is_ath")],                "역사적신고가(ATH)"),
        stat([s for s in all_signals if s.get("stage1_ok")],             "Stage1위치"),
        stat([s for s in all_signals if s.get("vc_confirmed")],          "VC 3중수축"),
    ]
    print_table(pat_rows, "[3] 패턴별 성과")

    # ══════════════════════════════════════════════════════
    # 4. 시장 regime별
    # ══════════════════════════════════════════════════════
    reg_rows = []
    for reg in ["BULL", "NEUTRAL", "BEAR"]:
        sub = [s for s in all_signals if s["regime"] == reg]
        reg_rows.append(stat(sub, f"{reg}장"))
    reg_rows.append(stat([s for s in all_signals if s["regime"]=="BULL" and s["tier"]=="CORE"], "BULL×CORE"))
    reg_rows.append(stat([s for s in all_signals if s["regime"]=="BULL" and s.get("is_ath")],  "BULL×ATH"))
    print_table(reg_rows, "[4] 시장 Regime별 성과")

    # ══════════════════════════════════════════════════════
    # 5. 최대 Drawdown 분석
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  [5] 최대 Drawdown 분석 (신호 후 60일)")
    print("="*80)

    all_mdd = [s["mdd_60d"] for s in all_signals if s.get("mdd_60d") is not None]
    if all_mdd:
        buckets = [
            ("MDD 0% 이상 (수익권)", [v for v in all_mdd if v >= 0]),
            ("MDD -5% 이내",        [v for v in all_mdd if -5 <= v < 0]),
            ("MDD -5~-10%",         [v for v in all_mdd if -10 <= v < -5]),
            ("MDD -10~-20%",        [v for v in all_mdd if -20 <= v < -10]),
            ("MDD -20% 이상 손실",  [v for v in all_mdd if v < -20]),
        ]
        print(f"  전체 평균 MDD: {sum(all_mdd)/len(all_mdd):+.2f}%  |  최악: {min(all_mdd):+.2f}%  |  중앙값: {sorted(all_mdd)[len(all_mdd)//2]:+.2f}%")
        print()
        for lbl, vals in buckets:
            pct = len(vals)/len(all_mdd)*100
            print(f"  {lbl:<25}  {len(vals):>4}건 ({pct:>5.1f}%)")

        print()
        for tier in ["CORE", "WATCH", "EARLY"]:
            sub_mdd = [s["mdd_60d"] for s in all_signals if s["tier"]==tier and s.get("mdd_60d") is not None]
            if sub_mdd:
                print(f"  {tier:<8} 평균MDD {sum(sub_mdd)/len(sub_mdd):+.2f}%  최악 {min(sub_mdd):+.2f}%")

    # ══════════════════════════════════════════════════════
    # 6. 페널티 종목 검증
    # ══════════════════════════════════════════════════════
    penalized = [s for s in all_signals if s["penalty"] > 0]
    clean     = [s for s in all_signals if s["penalty"] == 0]

    print(f"\n{'='*80}")
    print("  [6] 페널티 받은 종목 실제 성과 검증")
    print("="*80)
    _row2 = lambda lbl, items: stat(items, lbl)

    pen_rows = [
        stat(clean,                                                        "페널티 없음"),
        stat(penalized,                                                    "페널티 있음 (전체)"),
        stat([s for s in penalized if s["penalty"] >= 10],                "페널티 10점+"),
        stat([s for s in penalized if s["penalty"] >= 20],                "페널티 20점+"),
        stat([s for s in all_signals if "RVOL" in str(s.get("pen_reasons",""))], "RVOL 페널티"),
        stat([s for s in all_signals if "RSI"  in str(s.get("pen_reasons",""))], "RSI 페널티"),
        stat([s for s in all_signals if "경과"  in str(s.get("pen_reasons",""))], "일경과 페널티"),
        stat([s for s in all_signals if "하락"  in str(s.get("pen_reasons",""))], "신고가후하락 페널티"),
    ]
    print_table(pen_rows, "[6] 페널티 유형별 실제 성과")

    # 페널티 받았음에도 통과한 종목 top 손실
    if penalized:
        penalized_sorted = sorted(
            [s for s in penalized if s.get("ret_20d") is not None],
            key=lambda x: x["ret_20d"]
        )[:10]
        print("\n  페널티 받고도 통과한 종목 중 20일 최대 손실 TOP10:")
        print(f"  {'종목':<10} {'날짜':<12} {'grade':>6} {'pen':>5} {'pen_reason':<35} {'20일수익':>8}")
        print("  " + "-"*85)
        for s in penalized_sorted:
            pr = str(s.get("pen_reasons",""))[:33]
            print(f"  {s['name']:<10} {s['date']:<12} {s['grade']:>6} {s['penalty']:>5} {pr:<35} {s['ret_20d']:>+7.2f}%")

    # ── 전체 요약 ──
    print(f"\n{'='*80}")
    print("  전체 요약")
    print("="*80)
    prime_a = [s for s in prime_sigs if s["grade"] == "A+"]
    print_table([
        stat(all_signals,                                                  "전체"),
        stat(prime_sigs,                                                   "🚀 PRIME (BULL×ATH)"),
        stat(prime_a,                                                      "PRIME×A+"),
        stat([s for s in all_signals if s["tier"]=="CORE"],               "CORE (S+A+)"),
        stat([s for s in all_signals if s["penalty"] == 0],               "페널티없음"),
        stat([s for s in prime_sigs   if s["penalty"] == 0],              "PRIME×페널티없음"),
    ], "핵심 조합 요약")

    # ══════════════════════════════════════════════════════
    # 슈퍼 알파 탐색 — PRIME×A+ 내부 최강 필터 찾기
    # ══════════════════════════════════════════════════════
    def ma20_near(s):
        d = s.get("ma20_dist")
        return d is not None and abs(d) <= 5.0

    super_combos = [
        ("1. PRIME×A+ 기본",              prime_a),
        ("2. PRIME×A+×페널티없음",         [s for s in prime_a if s["penalty"] == 0]),
        ("3. PRIME×A+×RVOL 2x+",          [s for s in prime_a if s.get("rvol", 0) >= 2.0]),
        ("4. PRIME×A+×Stage1위치",         [s for s in prime_a if s.get("stage1_ok")]),
        ("5. PRIME×A+×MA20이격±5%",        [s for s in prime_a if ma20_near(s)]),
        ("6. PRIME×A+×VolumeDryup",        [s for s in prime_a if s.get("vol_dryup")]),
        ("7. PRIME×A+×MA5위",              [s for s in prime_a if s.get("above_ma5")]),
        # 복합
        ("8. PRIME×A+×페널티없음×MA20",    [s for s in prime_a if s["penalty"] == 0 and ma20_near(s)]),
        ("9. PRIME×A+×Stage1×페널티없음",  [s for s in prime_a if s.get("stage1_ok") and s["penalty"] == 0]),
        ("10. PRIME×A+×Dryup×MA5위",       [s for s in prime_a if s.get("vol_dryup") and s.get("above_ma5")]),
    ]

    print(f"\n{'='*80}")
    print("  [7] 슈퍼 알파 탐색 — PRIME×A+ 최강 필터 (N≥30 통계 신뢰)")
    print(f"{'='*80}")
    print(f"  {'조합':<35} {'N':>4}  {'20일승률':>8}  {'20일평균':>8}  {'50%달성':>8}  {'평균MDD':>8}")
    print("  " + "-"*75)

    best_combo = None
    best_avg   = 0.0

    for lbl, items in super_combos:
        r = stat(items, lbl)
        n  = r["N"]
        w20 = r.get("win_20d") or 0
        a20 = r.get("avg_20d") or 0
        p50 = r.get("pct_50")  or 0
        mdd = r.get("avg_mdd") or 0
        flag = "⚠️ 표본부족" if n < 30 else ("🏆 최강!" if a20 > 6.0 else ("✅" if a20 > 5.5 else ""))
        print(f"  {lbl:<35} {n:>4}  {w20:>7.1f}%  {a20:>+7.2f}%  {p50:>7.1f}%  {mdd:>+7.2f}%  {flag}")
        if n >= 30 and a20 > best_avg:
            best_avg   = a20
            best_combo = lbl

    # PRIME-S 판정
    print(f"\n  {'─'*75}")
    if best_combo and best_avg >= 5.5:
        print(f"\n  🚀🚀 PRIME-S 후보 발견: [{best_combo}]  20일 평균 {best_avg:+.2f}%")
        print(f"     → 이 조합이 PRIME-S 등급 기준으로 채택됩니다.")
    else:
        print(f"\n  ℹ️  현재 데이터에서 +5.5% 이상 N≥30 조합 미발견. PRIME×A+ 기본값 유지.")

    print(f"\n결과 저장: backtest_newhigh.json ({len(all_signals)}건)")

    # ══════════════════════════════════════════════════════
    # [8] 매매규칙 시뮬레이션 vs 단순보유 비교
    # ══════════════════════════════════════════════════════
    def sim_stat(items, label):
        sim_rets = [s["sim_ret"] for s in items if s.get("sim_ret") is not None]
        hold_rets = [s["ret_20d"] for s in items if s.get("ret_20d") is not None]
        sim_mdds  = [s["sim_mdd"] for s in items if s.get("sim_mdd") is not None]
        hold_mdds = [s["mdd_60d"] for s in items if s.get("mdd_60d") is not None]
        n = len(sim_rets)
        if n == 0:
            print(f"  {label:<32} {'0':>4}")
            return
        avg_sim   = round(sum(sim_rets) / n, 2)
        avg_hold  = round(sum(hold_rets) / len(hold_rets), 2) if hold_rets else 0
        mdd_sim   = round(sum(sim_mdds) / len(sim_mdds), 2) if sim_mdds else 0
        mdd_hold  = round(sum(hold_mdds) / len(hold_mdds), 2) if hold_mdds else 0
        win_sim   = round(sum(1 for v in sim_rets if v > 0) / n * 100, 1)
        win_hold  = round(sum(1 for v in hold_rets if v > 0) / len(hold_rets) * 100, 1) if hold_rets else 0
        stop_pct  = round(sum(1 for s in items if s.get("sim_stop")) / n * 100, 1)
        tp1_pct   = round(sum(1 for s in items if s.get("sim_tp1")) / n * 100, 1)
        tp2_pct   = round(sum(1 for s in items if s.get("sim_tp2")) / n * 100, 1)
        sharpe_sim  = round(avg_sim / max(abs(mdd_sim), 0.1), 2)
        sharpe_hold = round(avg_hold / max(abs(mdd_hold), 0.1), 2)
        print(f"  {label:<32} {n:>4} | 단순보유: 수익{avg_hold:>+6.2f}% 승률{win_hold:.1f}% MDD{mdd_hold:.1f}% SR{sharpe_hold:.2f}"
              f" | 매매규칙: 수익{avg_sim:>+6.2f}% 승률{win_sim:.1f}% MDD{mdd_sim:.1f}% SR{sharpe_sim:.2f}"
              f" | 손절:{stop_pct:.0f}% 1차익절:{tp1_pct:.0f}% 2차익절:{tp2_pct:.0f}%")

    prime_a = [s for s in all_signals if s.get("is_prime") and s.get("grade") == "A+"]
    prime_s_safe    = [s for s in prime_a if s["penalty"] == 0 and s["rvol"] < 2.0]
    prime_s_xpl     = [s for s in prime_a if s["rvol"] >= 2.0]
    prime_s_max     = [s for s in prime_a if s["penalty"] == 0 and s["rvol"] >= 2.0]
    non_prime_core  = [s for s in all_signals if not s.get("is_prime") and s.get("grade") in ("A+","S")]

    print(f"\n{'='*120}")
    print("  [8] 매매규칙 시뮬레이션 — 단순보유 vs 3트랜치(-7%/+20%/+50%)")
    print(f"{'='*120}")
    print(f"  {'조합':<32} {'N':>4}   {'──── 단순보유(20일) ────':^42}   {'──── 매매규칙 ────':^42}")
    print()
    sim_stat(prime_s_max,    "🔥🔥 PRIME-S MAX (페널티0+RVOL2x+)")
    sim_stat(prime_s_safe,   "🔥  PRIME-S Safe (페널티없음, RVOL<2)")
    sim_stat(prime_s_xpl,    "🔥  PRIME-S eXplosive (RVOL 2x+)")
    sim_stat(prime_a,        "🚀  PRIME×A+ (전체)")
    sim_stat(non_prime_core, "📊  비PRIME CORE (대조군)")
    sim_stat(all_signals,    "전체")

    # ══════════════════════════════════════════════════════
    # 누적 DB 저장 (backtest_history.csv)
    # 키: signal_date + ticker (최신 실행 결과로 덮어쓰기)
    # ══════════════════════════════════════════════════════
    run_date = datetime.now().strftime("%Y-%m-%d")
    DB_PATH  = "backtest_history.csv"

    rows = []
    for s in all_signals:
        rows.append({
            "run_date"       : run_date,
            "signal_date"    : s.get("date", ""),
            "ticker"         : s.get("ticker", ""),
            "name"           : s.get("name", ""),
            "score"          : s.get("score", 0),
            "grade"          : s.get("grade", ""),
            "tier"           : s.get("tier", ""),
            "is_prime"       : s.get("is_prime", False),
            "type"           : s.get("type", ""),
            "rvol"           : s.get("rvol", 0),
            "rsi"            : s.get("rsi", 0),
            "regime"         : s.get("regime", ""),
            "penalty"        : s.get("penalty", 0),
            "pen_reasons"    : str(s.get("pen_reasons", [])),
            "is_ath"         : s.get("is_ath", False),
            "first_breakout" : s.get("first_breakout", False),
            "retest"         : s.get("retest", False),
            "stage1_ok"      : s.get("stage1_ok", False),
            "vc_confirmed"   : s.get("vc_confirmed", False),
            "killer"         : s.get("killer", False),
            "ret_5d"         : s.get("ret_5d"),
            "ret_10d"        : s.get("ret_10d"),
            "ret_20d"        : s.get("ret_20d"),
            "ret_max"        : s.get("ret_max"),
            "mdd_60d"        : s.get("mdd_60d"),
            "reached_20pct"  : (s.get("ret_max") or 0) >= 20,
            "reached_50pct"  : (s.get("ret_max") or 0) >= 50,
        })

    new_df = pd.DataFrame(rows)

    if os.path.exists(DB_PATH):
        old_df = pd.read_csv(DB_PATH, encoding="utf-8-sig")
        # (signal_date, ticker) 페어 기준으로 정확히 제거 (isin 독립 체크 버그 방지)
        new_keys = set(
            zip(new_df["signal_date"].astype(str), new_df["ticker"].astype(str))
        )
        mask_keep = [
            (str(sd), str(tk)) not in new_keys
            for sd, tk in zip(old_df["signal_date"].astype(str), old_df["ticker"].astype(str))
        ]
        old_df = old_df[mask_keep]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # 최종 안전망: 혹시 남은 중복 제거 (score 높은 것 보존)
    combined = (combined
        .sort_values("score", ascending=False)
        .drop_duplicates(subset=["signal_date", "ticker"], keep="first")
        .sort_values(["signal_date", "ticker"])
        .reset_index(drop=True)
    )

    combined.sort_values(["signal_date", "ticker"], inplace=True)
    combined.to_csv(DB_PATH, index=False, encoding="utf-8-sig")
    print(f"누적 DB 저장: {DB_PATH} ({len(combined)}건 총, 이번 {len(new_df)}건 추가/갱신)")


if __name__ == "__main__":
    import os
    main()
