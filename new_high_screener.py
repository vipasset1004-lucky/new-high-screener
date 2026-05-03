"""
신고가 위치 판별 엔진 v3  ─  기관/전문가 수준
══════════════════════���════════════════════��═══════════════════════
학술·실전 근거:
  George & Hwang (2004)  ─ 52주 고가 모멘텀 0.65%/월, 장기 비역전
  Minervini SEPA         ─ 8개 Trend Template + VCP 진행수축 패턴
  O'Neil CAN SLIM        ─ RVOL 1.5~2.0x 이상 필수, 50일 이상 베이스
  Stan Weinstein         ─ Stage 2 돌파 + 30주선(150일) 상향
  Quantified Strategies  ─ RVOL 2x+ 시 후속 상승 확률 72~76%

4위치 분류 (위치 = 파동 내 좌표):
  ① BREAKOUT_BOTTOM   바닥 탈출형  ─ 장기 박스 첫 돌파, 주도주 초기  [최상급]
  ② PULLBACK_REBREAK  눌림 재돌파형 ─ 1차 상승→조정 완료→2차 시작    [우수]
  ③ TREND_CONTINUE    추세 진행형   ─ 추세 중, 추가 필터 필요         [중립]
  ④ OVERHEATED        과열 신고가   ─ 분배 구간, 자동 제외            [위험]

7단계 파이프라인:
  STEP1  신고가 탐지 (52주 + 120일 첫 돌파 여부)
  STEP2  위치 판별  (4분류)
  STEP3  Minervini Trend Template (8기준)
  STEP4  VCP 수렴 강도  (진행 수축 패턴)
  STEP5  돌파 수급     (RVOL + 최초 거래대금 급증)
  STEP6  유지력 + RS  (3일 버팀 + Retest + 상대강도)
  STEP7  과열 감점     (RSI/급등/윗꼬리)
═══════════════════════════════════════════════════════════════════
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime, timedelta
import time, warnings, os, json

warnings.filterwarnings("ignore")

SCAN_DAYS      = 400   # 넉넉히 (200MA + 여유)
NEW_HIGH_WIN   = 10    # 최근 N일 내 신고가 탐지
HIGH_52W       = 252   # 52주 거래일
BOX_WIN        = 120   # 첫 돌파 판별 기간


# ── 종목 리스트 ─────────────��──────────────────────────
def get_fallback_tickers(top_n=150):
    raw = [
        # KOSPI 대형/중형
        ("005930","삼성전자"),("000660","SK하이닉스"),("005380","현대차"),
        ("035420","NAVER"),("000270","기아"),("051910","LG화학"),
        ("006400","삼성SDI"),("035720","카카오"),("105560","KB금융"),
        ("055550","신한지주"),("086790","하나금융지주"),("032830","삼성생명"),
        ("028260","삼성물산"),("207940","삼성바이오로직스"),("068270","셀트리온"),
        ("373220","LG에너지솔루션"),("096770","SK이노베이션"),
        ("011200","HMM"),("003550","LG"),("010130","고려아연"),
        ("010950","S-Oil"),("012330","현대모비스"),("018260","삼성에스디에스"),
        ("009150","삼성전기"),("000100","유한양행"),("017670","SK텔레콤"),
        ("030200","KT"),("066570","LG전자"),("003490","대한항공"),
        ("009540","HD한국조선해양"),("267250","HD현대"),
        ("086280","현대글로비스"),("064350","현대로템"),
        ("298040","효성중공업"),("012450","한화에어로스페이스"),
        ("047810","한국항공우주"),("042660","한화오션"),
        ("010140","삼성중공업"),("004020","현대제철"),("005490","POSCO홀딩스"),
        ("247540","에코프로비엠"),("086520","에코프로"),("196170","알테오젠"),
        ("293490","카카오뱅크"),("041510","에스엠"),("352820","하이브"),
        ("003230","삼양식품"),("271560","오리온"),("002790","아모레퍼시픽"),
        ("128940","한미약품"),("006800","미래에셋증권"),("039490","키움증권"),
        ("071050","한국금융지주"),("145020","휴젤"),("214150","클래시스"),
        ("326030","SK바이오팜"),("272210","한화시스템"),
        ("161390","한국타이어앤테크놀로지"),("097950","CJ제일제당"),
        ("034730","SK"),("011790","SKC"),("082740","한화엔진"),
        ("018880","한온시스템"),("069620","대웅제약"),
        ("016360","삼성증권"),("000810","삼성화재"),
        ("088350","한화생명"),("139480","이마트"),
        ("011780","금호석유"),("004370","농심"),("011170","롯데케미칼"),
        # KOSDAQ
        ("263750","펄어비스"),("194480","데브시스터즈"),("251270","넷마블"),
        ("036570","엔씨소프트"),("112040","위메이드"),("095660","네오위즈"),
        ("067160","아프리카TV"),("237690","에스티팜"),("199800","툴젠"),
        ("357780","솔브레인"),("053800","안랩"),("192080","더블유게임즈"),
        ("225570","넥슨게임즈"),("035900","JYP Ent."),
        ("122870","와이지엔터테인먼트"),("259960","크래프톤"),
        ("091990","셀트리온헬스케어"),("086900","메디오젠"),
        ("036800","나이스정보통신"),("095700","제넥신"),
        ("080160","모두투어"),("011040","OCI홀딩스"),
        ("006650","대한유화"),("004370","농심"),
    ]
    seen, out = set(), []
    for code, name in raw:
        if code not in seen:
            out.append({"ticker": code, "name": name})
            seen.add(code)
    return out[:top_n]


# ── 데이터 수집 ────────────────────────────────────────
def fetch_daily(ticker, is_korean=True, days=SCAN_DAYS):
    suffixes = [".KS", ".KQ"] if is_korean else [""]
    for sfx in suffixes:
        try:
            sym = f"{ticker}{sfx}"
            start = datetime.now() - timedelta(days=days + 120)
            df = yf.download(sym, start=start, end=datetime.now(),
                             auto_adjust=True, progress=False, timeout=15)
            if df is None or len(df) < 120: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close","Volume"])
            if df["Close"].iloc[-1] <= 0: continue
            return df
        except: continue
    return None

# KOSPI 지수 (RS 계산용) ─ 한 번만 다운로드하고 캐시
_index_cache = {}
def fetch_index(symbol="^KS11", days=400):
    if symbol in _index_cache:
        return _index_cache[symbol]
    try:
        start = datetime.now() - timedelta(days=days + 60)
        df = yf.download(symbol, start=start, end=datetime.now(),
                         auto_adjust=True, progress=False, timeout=10)
        if df is None or len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        _index_cache[symbol] = df
        return df
    except:
        return None


# ── 지표 계산 ──────────────────────────────────────────
def calc_indicators(df):
    try:
        c, h, lo, v = df["Close"], df["High"], df["Low"], df["Volume"]
        df["rsi"]       = RSIIndicator(close=c, window=14).rsi()
        m = MACD(close=c)
        df["macd"]      = m.macd()
        df["macd_sig"]  = m.macd_signal()
        df["macd_hist"] = m.macd_diff()
        df["adx"]       = ADXIndicator(high=h, low=lo, close=c, window=14).adx()
        bb = BollingerBands(close=c, window=20, window_dev=2)
        df["bb_up"]     = bb.bollinger_hband()
        df["bb_lo"]     = bb.bollinger_lband()
        df["bb_mid"]    = bb.bollinger_mavg()
        df["bb_w"]      = (df["bb_up"] - df["bb_lo"]) / df["bb_mid"]
        df["atr"]       = AverageTrueRange(high=h, low=lo, close=c, window=14).average_true_range()
        for p in [5,10,20,50,60,120,150,200]:
            df[f"ma{p}"] = c.rolling(p).mean()
        df["tv"]        = c * v
        df["tv_ma20"]   = df["tv"].rolling(20).mean()
        df["v_ma20"]    = v.rolling(20).mean()
        return df
    except:
        return None


# ═══════════════��════════════════════════════════════════
# STEP 1  신고가 탐지
# ═════════════════════════════════════════════════════��══
def detect_new_high(df):
    """
    최근 NEW_HIGH_WIN일 내 52주 신고가 탐지.
    추가 판별:
      first_breakout   ─ 120일 내 이전 신고가 없음 (진짜 첫 돌파)
      first_vol_surge  ─ 직전 60일에 거래대금 2배 급증 없었음 (최초 세력 진입)
      retest_pattern   ─ 한 번 닿고 3~8% 눌린 후 재돌파 시도 (76% 승률 패턴)
    """
    if len(df) < HIGH_52W + NEW_HIGH_WIN:
        return {"found": False}

    close = df["Close"].values
    high  = df["High"].values
    tv    = df["tv"].values

    for days_ago in range(0, NEW_HIGH_WIN + 1):
        idx = -(days_ago + 1)
        if abs(idx) > len(close): break

        we = len(close) + idx                        # 당일 인덱스
        ws = max(0, we - HIGH_52W)
        if we - ws < 60: continue

        prev_52h = float(np.max(high[ws:we]))         # 52주 고가 (당일 제외)
        if high[idx] < prev_52h * 0.999: continue    # 신고가 아님

        # ── 거래대금 ──
        tv_20 = tv[max(0, we-20):we]
        avg_tv = float(np.mean(tv_20)) if len(tv_20) else 1.0
        rvol   = float(tv[idx]) / avg_tv if avg_tv > 0 else 1.0

        # ── 120일 첫 돌파 ──
        box_s = max(0, we - BOX_WIN)
        prev_box_h = float(np.max(high[box_s:we-1])) if we-1 > box_s else 0
        first_breakout = prev_box_h < prev_52h * 0.97

        # ── 최초 거래대금 급증 ──
        older_tv = tv[max(0, we-70):max(0, we-20)]
        older_base = tv[max(0, we-100):max(0, we-70)]
        first_vol_surge = True
        if len(older_tv) > 0 and len(older_base) > 0:
            older_avg = float(np.mean(older_base))
            if older_avg > 0 and float(np.max(older_tv)) / older_avg >= 1.8:
                first_vol_surge = False

        # ── Retest 패턴 (한 번 닿고 3~8% 눌림 후 재돌파) ──
        retest = False
        if not first_breakout and days_ago == 0:
            # 최근 5~25일 내 52주 신고가 근접 후 3~8% 조정 흔적 확인
            look = min(25, len(close) - 1)
            sub_h = high[-(look+1):-1]
            sub_l = close[-(look+1):-1]
            if len(sub_h) > 3:
                prev_touch = float(np.max(sub_h))
                min_since  = float(np.min(sub_l))
                pullback   = (prev_touch - min_since) / prev_touch * 100
                if (prev_touch >= prev_52h * 0.99) and (3.0 <= pullback <= 10.0):
                    retest = True

        return {
            "found"          : True,
            "days_ago"       : days_ago,
            "breakout_price" : float(prev_52h),
            "breakout_high"  : float(high[idx]),
            "breakout_tv"    : float(tv[idx]),
            "avg_tv"         : avg_tv,
            "rvol"           : rvol,
            "first_breakout" : first_breakout,
            "first_vol_surge": first_vol_surge,
            "retest"         : retest,
        }

    return {"found": False}


# ════════════════════════════════════════════════════════
# STEP 2  위치 판별 (4분류)
# ════════════════════════════════════════════════════════
def classify_position(df, nh):
    """
    4위치 분류 — 핵심: 파동 내 좌표 파악
    Minervini 기준 6번(저점 대비 30%+)도 함께 판별
    """
    close  = df["Close"].values
    high   = df["High"].values
    low    = df["Low"].values
    ma20   = df["ma20"].values

    rsi     = float(df["rsi"].iloc[-1])   if "rsi" in df.columns else 50
    ret_5d  = (close[-1]/close[-6]-1)*100  if len(close)>=6  else 0
    ret_20d = (close[-1]/close[-21]-1)*100 if len(close)>=21 else 0

    # 윗꼬리
    rng3  = high[-3:] - low[-3:]
    wick3 = high[-3:] - close[-3:]
    wick_r = float(np.mean(wick3/rng3)) if np.all(rng3>0) else 0

    # 52주 저점 대비 위치
    low_52w     = float(np.min(low[-HIGH_52W:])) if len(low)>=HIGH_52W else float(np.min(low))
    pos_pct     = (close[-1]-low_52w)/low_52w*100 if low_52w>0 else 999
    # Minervini 기준6: 저점 대비 30% 이상
    above_30pct = pos_pct >= 30.0

    # ── ④ 과열 ──
    if (ret_5d>25 or rsi>82 or
        (ret_5d>18 and rsi>76) or
        (wick_r>0.50 and rsi>72)):
        return "OVERHEATED", pos_pct, ret_20d, above_30pct

    fb   = nh.get("first_breakout", False)
    days = nh.get("days_ago", 0)

    # 60일 박스권 여부
    p60h = float(np.max(high[-61:-1])) if len(high)>61 else float(np.max(high[:-1]))
    p60l = float(np.min(low[-61:-1]))  if len(low)>61  else float(np.min(low[:-1]))
    box_range = (p60h-p60l)/p60l*100 if p60l>0 else 999
    was_in_box = box_range < 35

    # 눌림 여부 (MA20 근처까지 내려온 적 있는지)
    nh_idx  = len(close) - days - 1
    pb_s    = max(0, nh_idx-60)
    had_pb  = False
    if nh_idx > pb_s:
        sub_c  = close[pb_s:nh_idx]
        sub_ma = ma20[pb_s:nh_idx]
        valid  = ~np.isnan(sub_ma)
        if np.any(valid):
            had_pb = bool(np.any(sub_c[valid] <= sub_ma[valid]*1.02))

    # ① 바닥 탈출형
    if fb and (was_in_box or ret_20d<=25) and ret_20d<=38:
        return "BREAKOUT_BOTTOM", pos_pct, ret_20d, above_30pct

    # ② 눌림 재돌파형
    if had_pb and not fb and ret_20d<=38 and rsi<78:
        return "PULLBACK_REBREAK", pos_pct, ret_20d, above_30pct

    # ③ 추세 진행형
    if ret_20d<=45 and rsi<78:
        return "TREND_CONTINUE", pos_pct, ret_20d, above_30pct

    return "OVERHEATED", pos_pct, ret_20d, above_30pct


# ═════════════════════════════════════��══════════════════
# STEP 3  Minervini Trend Template (8기준)
# ══════════════════════════════════════���═════════════════
def score_trend_template(df, pos_pct):
    """
    Minervini 8가지 기준 점수화 (25점)
    기준1  가격 > 150MA AND 200MA            (5점)
    기준2  150MA > 200MA                     (3점)
    기준3  200MA 1개월 이상 상승 중            (3점)
    기준4  50MA > 150MA AND 200MA            (4점)
    기준5  가격 > 50MA                        (3점)
    기준6  저점 대비 30% 이상                  (3점)  ← Minervini 핵심
    기준7  52주 고가 25% 이내                  (2점)  ← 아직 상승 여력
    기준8  RS 상위 (ADX 대체)                 (2점)
    """
    score  = 0
    passed = []
    failed = []
    c = df["Close"].values

    def safe(key):
        v = df[key].iloc[-1] if key in df.columns else np.nan
        return float(v) if not pd.isna(v) else None

    price = float(c[-1])
    ma50  = safe("ma50")
    ma150 = safe("ma150")
    ma200 = safe("ma200")

    # 기준1: 가격 > 150MA, 200MA
    if ma150 and ma200 and price > ma150 and price > ma200:
        score += 5; passed.append("가격>150·200선")
    elif ma200 and price > ma200:
        score += 2; passed.append("가격>200선")
    else:
        failed.append("가격<200선")

    # 기준2: 150MA > 200MA
    if ma150 and ma200 and ma150 > ma200:
        score += 3; passed.append("150>200선")
    else:
        failed.append("150<200선")

    # 기준3: 200MA 1개월 상승 (20거래일 전 200MA 대비)
    if "ma200" in df.columns and len(df) >= 220:
        ma200_20ago = float(df["ma200"].iloc[-20])
        if not pd.isna(ma200_20ago) and ma200 and ma200 > ma200_20ago:
            score += 3; passed.append("200선상승중")
        else:
            failed.append("200선하락")
    else:
        failed.append("200선데이터부족")

    # 기준4: 50MA > 150MA, 200MA
    if ma50 and ma150 and ma200 and ma50 > ma150 and ma50 > ma200:
        score += 4; passed.append("50>150·200선")
    elif ma50 and ma200 and ma50 > ma200:
        score += 2; passed.append("50>200선")
    else:
        failed.append("50선<200선")

    # 기준5: 가격 > 50MA
    if ma50 and price > ma50:
        score += 3; passed.append("가격>50선")
    else:
        failed.append("가격<50선")

    # 기준6: 저점 대비 30% 이상 ← 가장 중요한 Minervini 조건
    if pos_pct >= 30:
        score += 3; passed.append("저점+30%↑")
    else:
        failed.append(f"저점+{pos_pct:.0f}%")

    # 기준7: 52주 고가 25% 이내 (아직 상승 여력)
    if len(df) >= HIGH_52W:
        h52 = float(df["High"].iloc[-HIGH_52W:].max())
        dist = (h52 - price) / h52 * 100 if h52 > 0 else 100
        if dist <= 25:
            score += 2; passed.append(f"52주고가{dist:.0f}%이내")
        else:
            failed.append(f"52주고가{dist:.0f}%초과")

    # 기준8: ADX (RS 대체)
    adx = float(df["adx"].iloc[-1]) if "adx" in df.columns and not pd.isna(df["adx"].iloc[-1]) else 0
    if adx > 25:
        score += 2; passed.append(f"ADX강세{adx:.0f}")

    total_criteria = len(passed)
    return score, passed, failed, total_criteria


# ═══════════════════════════════════��════════════════════
# STEP 4  VCP 수렴 강도 (Minervini VCP)
# ══════════════════���═════════════════════════��═══════════
def score_vcp(df, nh):
    """
    VCP(Volatility Contraction Pattern) 정밀 측정 (20점)

    Minervini 원칙:
      - 수축은 반드시 진행형이어야 함: 20%→12%→6% (각 수축이 이전보다 작아야)
      - 거래량도 각 수축마다 줄어야 함
      - 최소 2~3회 수축 확인

    구현:
      1) 돌파 직전 20일 구간을 5일씩 쪼개어 각 구간의 변동폭 측정
      2) 각 구간 변동폭이 이전 구간보다 줄어드는지 확인 (진행 수축)
      3) 거래량도 각 구간에서 감소 여부 확인
    """
    score  = 0
    detail = {}
    days_ago = nh.get("days_ago", 0)
    c  = df["Close"].values
    h  = df["High"].values
    lo = df["Low"].values
    v  = df["Volume"].values

    end_idx = max(1, len(c) - days_ago)

    # ── 진행 수축 측정 (5일 단위 4구간) ──
    seg = 5
    segs = []
    for i in range(4):
        s = max(0, end_idx - (i+1)*seg)
        e = max(0, end_idx - i*seg)
        if e > s:
            rng = float(np.mean(h[s:e] - lo[s:e]))
            vol = float(np.mean(v[s:e]))
            segs.append((rng, vol))

    segs.reverse()   # 오래된 → 최근 순

    progressive_price = 0   # 진행 수축 횟수 (가격)
    progressive_vol   = 0   # 진행 수축 횟수 (거래량)
    if len(segs) >= 2:
        for i in range(1, len(segs)):
            if segs[i][0] < segs[i-1][0] * 0.90:   # 10% 이상 수축
                progressive_price += 1
            if segs[i][1] < segs[i-1][1] * 0.90:
                progressive_vol += 1

    detail["progressive_price"] = progressive_price
    detail["progressive_vol"]   = progressive_vol

    # 진행 수축 점수
    if progressive_price >= 3:   score += 12
    elif progressive_price >= 2: score += 8
    elif progressive_price >= 1: score += 4

    if progressive_vol >= 2:     score += 5
    elif progressive_vol >= 1:   score += 3

    # ── BB 수축 보조 ──
    if "bb_w" in df.columns:
        bw = df["bb_w"].dropna()
        if len(bw) >= 30:
            rec = float(bw.iloc[-10:].mean())
            mx  = float(bw.iloc[-60:].max())
            bb_ratio = rec/mx if mx > 0 else 1.0
            if   bb_ratio < 0.40: score += 3
            elif bb_ratio < 0.55: score += 2
            elif bb_ratio < 0.70: score += 1
            detail["bb_ratio"] = round(bb_ratio, 2)

    # ── 거래량 드라이업 (Dry-Up) ──
    # O'Neil: 수렴 구간 거래량이 평균의 50% 미만 = 매도압력 없음
    if end_idx > 20:
        pre_vol  = v[max(0, end_idx-10):end_idx]
        base_vol = v[max(0, end_idx-40):max(0, end_idx-10)]
        if len(pre_vol) > 0 and len(base_vol) > 0:
            dry_ratio = float(np.mean(pre_vol)) / float(np.mean(base_vol))
            if dry_ratio < 0.50:
                score += 5; detail["vol_dryup"] = True
            elif dry_ratio < 0.70:
                score += 3; detail["vol_dryup"] = "partial"
            else:
                detail["vol_dryup"] = False
            detail["dry_ratio"] = round(dry_ratio, 2)

    return min(score, 20), detail


# ═══════════════════════════════════════════════��════════
# STEP 5  돌파 수급 (RVOL + 최초 급증)
# ════════════════════════════════════════��═══════════════
def score_breakout_supply(nh):
    """
    O'Neil: RVOL 1.5x 최소, 2.0x 이상 권장
    Weinstein: 돌파일 거래량 2~3x 이상
    첫 급증이면 +보너스 (최초 세력 진입 신호)
    (20점)
    """
    rvol      = nh.get("rvol", 1.0)
    first_fv  = nh.get("first_vol_surge", False)

    if   rvol >= 3.5: base = 20
    elif rvol >= 2.5: base = 16
    elif rvol >= 2.0: base = 13
    elif rvol >= 1.5: base = 8
    elif rvol >= 1.2: base = 4
    else:             base = 0

    bonus = 3 if (first_fv and rvol >= 1.5) else 0
    return min(base + bonus, 20), {"rvol": round(rvol,2), "first_vol": first_fv}


# ════════════════════════════════════════════════════════
# STEP 6  유지력 + RS + Retest
# ═══════════════════════════════════════════════���════════
def score_holding_rs(df, nh, index_df=None):
    """
    유지력 (George & Hwang: 돌파 후 3일 버팀이 승률 핵심)
    Retest 패턴 (76% 승률 패턴)
    RS 상대강도 (O'Neil: RS 70 이상 필수)
    (25점)
    """
    score  = 0
    detail = {}
    c    = df["Close"].values
    v    = df["Volume"].values
    ma5  = df["ma5"].values
    ma10 = df["ma10"].values

    days_ago       = nh.get("days_ago", 0)
    breakout_price = nh.get("breakout_price", c[-1])
    breakout_tv    = nh.get("breakout_tv", 0)
    retest         = nh.get("retest", False)

    # ── 유지력 (15점) ──
    hold = 0
    detail["price_held"]  = False
    detail["vol_dried"]   = False
    detail["above_ma5"]   = False
    detail["days_held"]   = days_ago

    if days_ago >= 1:
        post_c  = c[-(days_ago):]
        post_tv = (c[-(days_ago):] * v[-(days_ago):])
        price_held = bool(np.all(post_c >= breakout_price * 0.97))
        vol_dried  = bool(len(post_tv)>0 and float(np.mean(post_tv)) < breakout_tv*0.72) if breakout_tv>0 else False
        above_ma5  = bool(c[-1] > ma5[-1]) if not np.isnan(ma5[-1]) else False
        above_ma10 = bool(c[-1] > ma10[-1]) if not np.isnan(ma10[-1]) else False

        if price_held: hold += 6
        if vol_dried:  hold += 5
        if above_ma5:  hold += 2
        if above_ma10: hold += 2

        # 3일 이상 버텼으면 만점 보정
        mult = min(1.0, days_ago / 3)
        hold = int(hold * mult)

        detail["price_held"] = price_held
        detail["vol_dried"]  = vol_dried
        detail["above_ma5"]  = above_ma5
    else:
        # 당일 신고가: 5일선 위면 기본 3점
        above_ma5 = bool(c[-1] > ma5[-1]) if not np.isnan(ma5[-1]) else False
        hold = 3 if above_ma5 else 1
        detail["above_ma5"] = above_ma5

    score += hold
    detail["hold_score"] = hold

    # ── Retest 패턴 보너스 (5점) ── 76% 승률
    if retest:
        score += 5; detail["retest"] = True
    else:
        detail["retest"] = False

    # ── RS 상대강도 (5점) ──
    rs_score = 0
    if index_df is not None and len(df) >= HIGH_52W and len(index_df) >= HIGH_52W:
        try:
            # 12개월 수익률 vs KOSPI
            stock_ret  = (c[-1] / c[-HIGH_52W] - 1) if c[-HIGH_52W] > 0 else 0
            idx_close  = index_df["Close"].values
            idx_ret    = (idx_close[-1] / idx_close[-HIGH_52W] - 1) if idx_close[-HIGH_52W] > 0 else 0
            rs_diff    = stock_ret - idx_ret   # 초과수익률

            if   rs_diff > 0.30: rs_score = 5
            elif rs_diff > 0.15: rs_score = 3
            elif rs_diff > 0.00: rs_score = 2
            else:                rs_score = 0
            detail["rs_diff"] = round(rs_diff*100, 1)
        except:
            pass

    score += rs_score
    detail["rs_score"] = rs_score

    return min(score, 25), detail


# ═══════════════════════════════════════════════════════���
# STEP 7  과열 감점
# ════════════════════════════════════════════════════════
def calc_penalty(df, pos_type):
    c    = df["Close"].values
    h    = df["High"].values
    lo   = df["Low"].values
    rsi  = float(df["rsi"].iloc[-1])   if "rsi" in df.columns else 50
    r5   = (c[-1]/c[-6]-1)*100         if len(c)>=6  else 0
    r20  = (c[-1]/c[-21]-1)*100        if len(c)>=21 else 0

    rng3  = h[-3:] - lo[-3:]
    wick3 = h[-3:] - c[-3:]
    wr    = float(np.mean(wick3/rng3)) if np.all(rng3>0) else 0

    pen, rsn = 0, []
    # RSI
    if   rsi > 82: pen+=18; rsn.append(f"RSI극과열({rsi:.0f})")
    elif rsi > 78: pen+=10; rsn.append(f"RSI과열({rsi:.0f})")
    elif rsi > 74: pen+=4;  rsn.append(f"RSI주의({rsi:.0f})")
    # 5일 급등
    if   r5 > 22: pen+=15; rsn.append(f"5일급등({r5:.0f}%)")
    elif r5 > 16: pen+=8;  rsn.append(f"5일과속({r5:.0f}%)")
    elif r5 > 12: pen+=3;  rsn.append(f"5일빠름({r5:.0f}%)")
    # 윗꼬리
    if   wr > 0.50: pen+=10; rsn.append("윗꼬리과다")
    elif wr > 0.38: pen+=5;  rsn.append("윗꼬리주의")
    # 추세진행형 20일 과속 추가 감점
    if pos_type == "TREND_CONTINUE" and r20 > 32:
        pen+=6; rsn.append(f"20일과속({r20:.0f}%)")

    return pen, rsn


# ════════════════════��══════════════════════���════════════
# 점수 통합 & 등급
# ════════════════════════════════════════════════════��═══
MAX_SCORE = 90   # 3+4+5+6 합계 최대 (25+20+20+25)

def grade(score, pos_type):
    mins = {"BREAKOUT_BOTTOM":40, "PULLBACK_REBREAK":45, "TREND_CONTINUE":55}
    if score < mins.get(pos_type, 99):
        return "D", "△ 약함"
    if   score >= 80: return "S",  "🔥 최강"
    elif score >= 70: return "A+", "⭐ 주목"
    elif score >= 58: return "A",  "✓ 유효"
    else:             return "B",  "○ 보통"


TYPE_META = {
    "BREAKOUT_BOTTOM" : {"ko":"바닥 탈출형",   "emoji":"🏔️","color":"#f59e0b","desc":"장기 박스 첫 돌파 · 주도주 초기"},
    "PULLBACK_REBREAK": {"ko":"눌림 재돌파형", "emoji":"💎","color":"#3b82f6","desc":"1차 상승 조정 완료 · 2차 상승 시작"},
    "TREND_CONTINUE"  : {"ko":"추세 진행형",   "emoji":"📈","color":"#22c55e","desc":"추세 유지 중 · 추가 필터 확인 필요"},
    "OVERHEATED"      : {"ko":"과열형",        "emoji":"⛔","color":"#ef4444","desc":"분배 구간 가능성"},
}


# ═══════════════════���════════════════════════════════════
# 단일 종목 분석
# ═══════════════════════════════════════════���════════════
def analyze_stock(ticker, name, is_korean=True, index_df=None):
    try:
        df = fetch_daily(ticker, is_korean)
        if df is None or len(df) < 120: return None
        df = calc_indicators(df)
        if df is None: return None

        # STEP1
        nh = detect_new_high(df)
        if not nh["found"]: return None

        # STEP2
        pos_type, pos_pct, ret_20d, above_30 = classify_position(df, nh)
        if pos_type == "OVERHEATED": return None

        # STEP3 Minervini Template
        s3, passed, failed, criteria_cnt = score_trend_template(df, pos_pct)

        # STEP4 VCP
        s4, d4 = score_vcp(df, nh)

        # STEP5 돌파 수급
        s5, d5 = score_breakout_supply(nh)

        # STEP6 유지력+RS
        s6, d6 = score_holding_rs(df, nh, index_df)

        # 유형 기본 가점
        type_bonus = {"BREAKOUT_BOTTOM":8, "PULLBACK_REBREAK":4, "TREND_CONTINUE":0}[pos_type]

        # STEP7 감점
        pen, pen_r = calc_penalty(df, pos_type)

        raw   = s3 + s4 + s5 + s6 + type_bonus
        final = max(0, min(100, raw - pen))

        g, gl = grade(final, pos_type)
        if g == "D": return None

        meta = TYPE_META[pos_type]
        cur  = float(df["Close"].iloc[-1])
        rsi  = float(df["rsi"].iloc[-1])   if "rsi" in df.columns else 0
        adx  = float(df["adx"].iloc[-1])   if "adx" in df.columns else 0

        h52 = float(df["High"].iloc[-HIGH_52W:].max()) if len(df)>=HIGH_52W else float(df["High"].max())
        l52 = float(df["Low"].iloc[-HIGH_52W:].min())  if len(df)>=HIGH_52W else float(df["Low"].min())
        pos_range = (cur-l52)/(h52-l52)*100 if (h52-l52)>0 else 50

        # Minervini 충족 기준 수 → 신뢰도 표시
        template_pct = round(criteria_cnt/8*100)

        return {
            "ticker"         : ticker,
            "name"           : name,
            "score"          : round(final,1),
            "grade"          : g,
            "grade_label"    : gl,
            "type"           : pos_type,
            "type_ko"        : meta["ko"],
            "type_emoji"     : meta["emoji"],
            "type_color"     : meta["color"],
            "type_desc"      : meta["desc"],
            "current_price"  : cur,
            "rsi"            : round(rsi,1),
            "adx"            : round(adx,1),
            "rvol"           : round(nh["rvol"],2),
            "days_ago"       : nh["days_ago"],
            "breakout_price" : round(nh["breakout_price"],0),
            "first_breakout" : nh["first_breakout"],
            "first_vol_surge": nh["first_vol_surge"],
            "retest"         : nh["retest"],
            "ret_20d"        : round(ret_20d,1),
            "pos_pct"        : round(pos_pct,1),
            "above_30pct"    : above_30,
            "pos_in_range"   : round(pos_range,1),
            "high_52w"       : round(h52,0),
            "low_52w"        : round(l52,0),
            "template_cnt"   : criteria_cnt,
            "template_pct"   : template_pct,
            "template_passed": passed,
            "template_failed": failed,
            "price_held"     : d6.get("price_held",False),
            "vol_dried"      : d6.get("vol_dried",False),
            "above_ma5"      : d6.get("above_ma5",False),
            "retest_bonus"   : d6.get("retest",False),
            "rs_diff"        : d6.get("rs_diff",0),
            "score_detail"   : {
                "s3_template"  : s3,
                "s4_vcp"       : s4,
                "s5_supply"    : s5,
                "s6_holding"   : s6,
                "type_bonus"   : type_bonus,
                "penalty"      : -pen,
                "penalty_rsn"  : pen_r,
                "vcp_detail"   : d4,
            },
        }
    except:
        return None


# ═════════════��══════════════════════════════════════════
# 전체 스캔
# ══════════════════════���═════════════════════════════════
def screen_stocks(top_n=150, progress_cb=None):
    tickers  = get_fallback_tickers(top_n)
    idx_df   = fetch_index("^KS11")   # KOSPI RS 계산용
    results  = []
    total    = len(tickers)

    for i, t in enumerate(tickers):
        if progress_cb: progress_cb(i, total, t["name"], len(results))
        r = analyze_stock(t["ticker"], t["name"], is_korean=True, index_df=idx_df)
        if r: results.append(r)
        time.sleep(0.08)

    order = {"BREAKOUT_BOTTOM":0, "PULLBACK_REBREAK":1, "TREND_CONTINUE":2}
    results.sort(key=lambda x: (order.get(x["type"],9), -x["score"]))
    return results


if __name__ == "__main__":
    print("=== 신고가 위치 판별 엔진 v3 ===")
    results = screen_stocks(top_n=40)
    print(f"\n발견: {len(results)}개\n")
    for r in results[:10]:
        fb = "★첫돌파" if r["first_breakout"] else ""
        rt = "↩Retest" if r["retest"]         else ""
        print(f"[{r['grade_label']}] {r['name']} {r['type_emoji']}{r['type_ko']} "
              f"점수:{r['score']} Template:{r['template_pct']}% "
              f"RVOL:{r['rvol']}x {fb}{rt}")
