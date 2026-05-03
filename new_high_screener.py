"""
신고가 위치 판별 엔진 (New High Position Engine)
==================================================
핵심 철학: "신고가 여부"가 아니라 "신고가의 위치+구조+수급+확률"을 본다

4위치 분류:
  ① BREAKOUT_BOTTOM  바닥 탈출 신고가   - 최상급 (주도주 초기)
  ② PULLBACK_REBREAK 눌림 후 재돌파     - 우수   (2차 상승 시작)
  ③ TREND_CONTINUE   추세 진행 중       - 중립   (추가 필터 필요)
  ④ OVERHEATED       과열 신고가        - 제외   (분배 구간)

6단계 파이프라인:
  STEP1 신고가 탐지
  STEP2 신고가 위치 판별 (4분류)
  STEP3 돌파 강도 점수
  STEP4 수급 점수
  STEP5 섹터 점수
  STEP6 과열 제거
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime, timedelta
import json
import time
import requests
import warnings
import os

warnings.filterwarnings("ignore")

SCAN_DAYS      = 320   # 일봉 데이터 (120일 박스 판별 + 여유)
NEW_HIGH_WINDOW = 10   # 최근 N일 내 신고가 탐지
HIGH_52W_DAYS  = 252   # 52주 = 약 252 거래일
BOX_WINDOW     = 120   # 박스권 판별 기준 기간 (일)


# ── 종목 리스트 ────────────────────────────────────────
def get_fallback_tickers(top_n=150):
    """주요 종목 기본 리스트"""
    tickers = [
        # KOSPI 대형주
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
        # KOSPI 중형주 모멘텀
        ("247540","에코프로비엠"),("086520","에코프로"),("196170","알테오젠"),
        ("293490","카카오뱅크"),("041510","에스엠"),("352820","하이브"),
        ("003230","삼양식품"),("271560","오리온"),("002790","아모레퍼시픽"),
        ("128940","한미약품"),("006800","미래에셋증권"),("039490","키움증권"),
        ("071050","한국금융지주"),("145020","휴젤"),("214150","클래시스"),
        ("326030","SK바이오팜"),("272210","한화시스템"),
        ("161390","한국타이어앤테크놀로지"),("097950","CJ제일제당"),
        ("034730","SK"),("011790","SKC"),
        # KOSDAQ
        ("263750","펄어비스"),("194480","데브시스터즈"),("251270","넷마블"),
        ("036570","엔씨소프트"),("112040","위메이드"),("095660","네오위즈"),
        ("067160","아프리카TV"),("237690","에스티팜"),("199800","툴젠"),
        ("357780","솔브레인"),("053800","안랩"),("192080","더블유게임즈"),
        ("225570","넥슨게임즈"),("041510","에스엠"),("035900","JYP Ent."),
        ("122870","와이지엔터테인먼트"),("259960","크래프톤"),
        ("091990","셀트리온헬스케어"),("086900","메디오젠"),
        ("036800","나이스정보통신"),("095700","제넥신"),
        ("108675","LX하우시스"),("080160","모두투어"),
        # 추가 거래대금 상위
        ("011040","OCI홀딩스"),("082740","한화엔진"),
        ("018880","한온시스템"),("069620","대웅제약"),
        ("016360","삼성증권"),("000810","삼성화재"),
        ("088350","한화생명"),("139480","이마트"),
        ("011780","금호석유"),("004370","농심"),
        ("006650","대한유화"),("011170","롯데케미칼"),
    ]
    seen, result = set(), []
    for code, name in tickers:
        if code not in seen:
            result.append({"ticker": code, "name": name})
            seen.add(code)
    return result[:top_n]


# ── 데이터 수집 ────────────────────────────────────────
def fetch_daily_data(ticker_code, is_korean=True, days=SCAN_DAYS):
    suffixes = [".KS", ".KQ"] if is_korean else [""]
    for suffix in suffixes:
        try:
            symbol = f"{ticker_code}{suffix}"
            end = datetime.now()
            start = end - timedelta(days=days + 90)
            df = yf.download(symbol, start=start, end=end,
                             auto_adjust=True, progress=False, timeout=15)
            if df is None or len(df) < 80:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close", "Volume"])
            if df["Close"].iloc[-1] <= 0:
                continue
            return df
        except:
            continue
    return None


# ── 지표 계산 ──────────────────────────────────────────
def calculate_indicators(df):
    try:
        c, h, lo, v = df["Close"], df["High"], df["Low"], df["Volume"]

        df["rsi"]        = RSIIndicator(close=c, window=14).rsi()
        m = MACD(close=c, window_slow=26, window_fast=12, window_sign=9)
        df["macd"]       = m.macd()
        df["macd_signal"]= m.macd_signal()
        df["macd_hist"]  = m.macd_diff()
        adx = ADXIndicator(high=h, low=lo, close=c, window=14)
        df["adx"]        = adx.adx()
        bb = BollingerBands(close=c, window=20, window_dev=2)
        df["bb_upper"]   = bb.bollinger_hband()
        df["bb_lower"]   = bb.bollinger_lband()
        df["bb_mid"]     = bb.bollinger_mavg()
        df["bb_width"]   = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["atr"]        = AverageTrueRange(high=h, low=lo, close=c, window=14).average_true_range()
        df["ma5"]        = c.rolling(5).mean()
        df["ma10"]       = c.rolling(10).mean()
        df["ma20"]       = c.rolling(20).mean()
        df["ma60"]       = c.rolling(60).mean()
        df["ma120"]      = c.rolling(120).mean()
        df["vol_ma20"]   = v.rolling(20).mean()
        df["trade_value"]= c * v
        df["tv_ma20"]    = df["trade_value"].rolling(20).mean()
        return df
    except:
        return None


# ══════════════════════════════════════════════════════
# STEP 1 : 신고가 탐지
# ══════════════════════════════════════════════════════
def detect_new_high(df):
    """
    최근 NEW_HIGH_WINDOW일 내 52주 신고가 발생 여부 탐지.
    추가: 120일 내 이전 신고가 존재 여부 (첫 돌파 판별용)
    """
    if len(df) < HIGH_52W_DAYS + NEW_HIGH_WINDOW:
        return {"found": False}

    close  = df["Close"].values
    high   = df["High"].values
    tv     = df["trade_value"].values

    for days_ago in range(0, NEW_HIGH_WINDOW + 1):
        idx = -(days_ago + 1)
        if abs(idx) > len(close):
            break

        window_end   = len(close) + idx
        window_start = max(0, window_end - HIGH_52W_DAYS)
        if window_end - window_start < 60:
            continue

        prev_52w_high = np.max(high[window_start:window_end])
        today_high    = high[idx]

        if today_high >= prev_52w_high * 0.999:
            # 20일 평균 거래대금 & RVOL
            tv_win   = tv[max(0, window_end - 20):window_end]
            avg_tv   = np.mean(tv_win) if len(tv_win) > 0 else 1
            rvol     = tv[idx] / avg_tv if avg_tv > 0 else 1

            # ── 120일 첫 돌파 여부 ──
            # 신고가 발생 시점 기준 이전 120일 중 52주 고가 근접(97%) 여부
            box_start = max(0, window_end - BOX_WINDOW)
            box_end   = max(0, window_end - 1)
            prev_box_high = np.max(high[box_start:box_end]) if box_end > box_start else 0
            first_breakout = prev_box_high < prev_52w_high * 0.97  # 120일간 신고가 없었음

            # ── 직전 거래대금 급증 최초 여부 ──
            # 60일 전 구간에서 RVOL 2배 이상 있었는지
            older_tv_win = tv[max(0, window_end - 60):max(0, window_end - 20)]
            older_avg_tv = np.mean(tv[max(0, window_end - 80):max(0, window_end - 60)]) if window_end > 80 else avg_tv
            first_vol_surge = True
            if len(older_tv_win) > 0 and older_avg_tv > 0:
                older_rvol = np.max(older_tv_win) / older_avg_tv
                if older_rvol >= 1.8:
                    first_vol_surge = False  # 이미 전에 거래대금 급증 있었음

            return {
                "found"         : True,
                "days_ago"      : days_ago,
                "breakout_price": float(prev_52w_high),
                "breakout_high" : float(today_high),
                "breakout_tv"   : float(tv[idx]),
                "avg_tv_20d"    : float(avg_tv),
                "rvol"          : float(rvol),
                "first_breakout": bool(first_breakout),   # ★ 120일 첫 돌파
                "first_vol_surge": bool(first_vol_surge), # ★ 거래대금 최초 급증
            }

    return {"found": False}


# ══════════════════════════════════════════════════════
# STEP 2 : 신고가 위치 판별 (4분류)
# ══════════════════════════════════════════════════════
def classify_position(df, nh):
    """
    4위치 분류:
      BREAKOUT_BOTTOM  바닥 탈출 신고가  ① 최상급
      PULLBACK_REBREAK 눌림 후 재돌파    ② 우수
      TREND_CONTINUE   추세 진행 중      ③ 중립
      OVERHEATED       과열 신고가       ④ 제외
    """
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    ma20  = df["ma20"].values
    ma60  = df["ma60"].values

    # ── 기본 수치 ──
    rsi     = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50
    adx     = float(df["adx"].iloc[-1]) if "adx" in df.columns else 15
    ret_5d  = (close[-1] / close[-6]  - 1) * 100 if len(close) >= 6  else 0
    ret_20d = (close[-1] / close[-21] - 1) * 100 if len(close) >= 21 else 0

    # 윗꼬리 비율 (최근 3일 평균)
    rng  = high[-3:] - low[-3:]
    wick = high[-3:] - close[-3:]
    wick_ratio = float(np.mean(wick / rng)) if np.all(rng > 0) else 0

    # 52주 저점 대비 현재 위치
    low_52w     = float(np.min(low[-HIGH_52W_DAYS:])) if len(low) >= HIGH_52W_DAYS else float(np.min(low))
    position_pct = (close[-1] - low_52w) / low_52w * 100 if low_52w > 0 else 999

    # ── ④ 과열 판정 (가장 먼저) ──
    overheated = (
        ret_5d  > 25 or
        rsi     > 82 or
        (ret_5d > 18 and rsi > 76) or
        (wick_ratio > 0.5 and rsi > 72)
    )
    if overheated:
        return "OVERHEATED", position_pct, ret_20d

    first_breakout  = nh.get("first_breakout", False)
    first_vol_surge = nh.get("first_vol_surge", False)
    rvol            = nh.get("rvol", 1.0)
    days_ago        = nh.get("days_ago", 0)

    # ── 박스권 여부 (60일 변동폭 < 30%) ──
    p60_high = float(np.max(high[-61:-1])) if len(high) > 61 else float(np.max(high[:-1]))
    p60_low  = float(np.min(low[-61:-1]))  if len(low)  > 61 else float(np.min(low[:-1]))
    box_range = (p60_high - p60_low) / p60_low * 100 if p60_low > 0 else 999
    was_in_box = box_range < 35

    # ── 눌림 여부 (신고가 전 20~60일 중 MA20 하단 터치) ──
    nh_idx = len(close) - days_ago - 1
    pb_start = max(0, nh_idx - 60)
    pb_end   = max(0, nh_idx)
    had_pullback = False
    if pb_end > pb_start:
        sub_c   = close[pb_start:pb_end]
        sub_ma  = ma20[pb_start:pb_end]
        valid   = ~np.isnan(sub_ma)
        if np.any(valid):
            # MA20 ± 2% 이내로 내려온 적 있으면 눌림으로 인정
            had_pullback = np.any(sub_c[valid] <= sub_ma[valid] * 1.02)

    # ── ① 바닥 탈출 신고가 ──
    # 핵심 킬러 필터: 120일 첫 돌파 + 거래대금 2배 + 박스권 또는 저점 초기
    if (first_breakout and
        (was_in_box or ret_20d <= 25) and
        ret_20d <= 35):
        return "BREAKOUT_BOTTOM", position_pct, ret_20d

    # ── ② 눌림 후 재돌파 ──
    if (had_pullback and
        not first_breakout and
        ret_20d <= 35 and
        rsi < 78):
        return "PULLBACK_REBREAK", position_pct, ret_20d

    # ── ③ 추세 진행 중 ──
    if ret_20d <= 45 and rsi < 78:
        return "TREND_CONTINUE", position_pct, ret_20d

    # 기본: 과열에 가까운 경계
    return "OVERHEATED", position_pct, ret_20d


# ══════════════════════════════════════════════════════
# STEP 3 : 돌파 강도 점수
# ══════════════════════════════════════════════════════
def score_breakout_strength(df, nh):
    """
    수렴 강도 + 돌파 거래대금 + 첫 거래대금 급증 여부
    반환: 점수(0-40), 상세
    """
    score  = 0
    detail = {}

    # ── 수렴 점수 (20점): 돌파 전 10일 변동폭 수축 ──
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values

    # 신고가 발생 전 10일 일봉 변동폭 vs 직전 30일 평균 변동폭
    days_ago  = nh.get("days_ago", 0)
    end_idx   = max(1, len(close) - days_ago)
    pre10_rng = (high[max(0,end_idx-10):end_idx] - low[max(0,end_idx-10):end_idx])
    pre30_rng = (high[max(0,end_idx-30):end_idx] - low[max(0,end_idx-30):end_idx])

    conv_score = 0
    if len(pre10_rng) > 0 and len(pre30_rng) > 0:
        avg10 = float(np.mean(pre10_rng))
        avg30 = float(np.mean(pre30_rng))
        ratio = avg10 / avg30 if avg30 > 0 else 1.0
        if   ratio < 0.45: conv_score = 20
        elif ratio < 0.60: conv_score = 14
        elif ratio < 0.75: conv_score = 8
        elif ratio < 0.90: conv_score = 3
        detail["conv_ratio"] = round(ratio, 2)
    else:
        detail["conv_ratio"] = 1.0

    # BB 수축 보조 확인
    if "bb_width" in df.columns:
        bb = df["bb_width"].dropna()
        if len(bb) >= 30:
            recent_bb = float(bb.iloc[-10:].mean())
            max_bb60  = float(bb.iloc[-60:].max())
            bb_ratio  = recent_bb / max_bb60 if max_bb60 > 0 else 1.0
            if bb_ratio < 0.5 and conv_score < 20:
                conv_score = max(conv_score, 12)
            detail["bb_ratio"] = round(bb_ratio, 2)

    score += conv_score
    detail["convergence"] = conv_score

    # ── 돌파 거래대금 점수 (20점) ──
    rvol             = nh.get("rvol", 1.0)
    first_vol_surge  = nh.get("first_vol_surge", False)

    if   rvol >= 3.0: vol_score = 20
    elif rvol >= 2.0: vol_score = 15
    elif rvol >= 1.5: vol_score = 10
    elif rvol >= 1.2: vol_score = 5
    else:             vol_score = 0

    # 최초 거래대금 급증이면 +3 보너스
    if first_vol_surge and rvol >= 1.5:
        vol_score = min(20, vol_score + 3)

    score += vol_score
    detail["breakout_vol"] = vol_score
    detail["rvol"]         = round(rvol, 2)
    detail["first_vol"]    = first_vol_surge

    return score, detail


# ══════════════════════════════════════════════════════
# STEP 4 : 수급 점수
# ══════════════════════════════════════════════════════
def score_supply_demand(df, nh):
    """
    유지력 + MA 정배열 + 거래량 추세
    반환: 점수(0-35), 상세
    """
    score  = 0
    detail = {}
    close  = df["Close"].values
    high   = df["High"].values
    volume = df["Volume"].values
    ma5    = df["ma5"].values
    ma20   = df["ma20"].values
    ma60   = df["ma60"].values
    ma120  = df["ma120"].values

    days_ago       = nh.get("days_ago", 0)
    breakout_price = nh.get("breakout_price", close[-1])
    breakout_tv    = nh.get("breakout_tv", 0)

    # ── 유지력 점수 (20점): 신고가 후 N일 버팀 ──
    hold_score = 0
    if days_ago >= 1:
        post_close = close[-(days_ago):]
        post_vol   = volume[-(days_ago):]

        # 가격 유지 (신고가 기준가 -3% 이내)
        price_held = bool(np.all(post_close >= breakout_price * 0.97))

        # 거래량 감소 패턴 (눌림에서 거래량 줄어야 좋음)
        post_tv   = (post_close * post_vol)
        vol_dried = bool(len(post_tv) > 0 and float(np.mean(post_tv)) < breakout_tv * 0.75) if breakout_tv > 0 else False

        # 5일선 위 유지
        above_ma5 = bool(close[-1] > ma5[-1]) if not np.isnan(ma5[-1]) else False

        if price_held: hold_score += 10
        if vol_dried:  hold_score += 6
        if above_ma5:  hold_score += 4

        # 3일 이상 버텼으면 만점 가능
        if days_ago < 3:
            hold_score = int(hold_score * (days_ago / 3))

        detail["price_held"] = price_held
        detail["vol_dried"]  = vol_dried
        detail["above_ma5"]  = above_ma5
    else:
        # 신고가 당일 → 아직 검증 전, 기본 5점
        hold_score = 5
        detail["price_held"] = True
        detail["vol_dried"]  = False
        detail["above_ma5"]  = bool(close[-1] > ma5[-1]) if not np.isnan(ma5[-1]) else True

    score += hold_score
    detail["holding"] = hold_score

    # ── MA 정배열 점수 (10점) ──
    def safe(arr): return float(arr[-1]) if not np.isnan(arr[-1]) else None
    p, m5, m20, m60, m120 = (safe(close), safe(ma5), safe(ma20), safe(ma60), safe(ma120))

    ma_score = 0
    if all(v is not None for v in [p, m5, m20, m60, m120]):
        if p > m5 > m20 > m60 > m120: ma_score = 10
        elif p > m5 > m20 > m60:      ma_score = 8
        elif p > m20 > m60:            ma_score = 5
        elif p > m60:                  ma_score = 2
    elif all(v is not None for v in [p, m5, m20, m60]):
        if p > m5 > m20 > m60: ma_score = 8
        elif p > m20 > m60:    ma_score = 5

    score += ma_score
    detail["ma_alignment"] = ma_score

    # ── 거래량 추세 점수 (5점): 최근 5일 평균 > 20일 평균 ──
    vol_ma20 = df["vol_ma20"].values
    vol_trend_score = 0
    if not np.isnan(vol_ma20[-1]):
        vol_5d_avg = float(np.mean(volume[-5:]))
        if vol_5d_avg > float(vol_ma20[-1]) * 1.3:
            vol_trend_score = 5
        elif vol_5d_avg > float(vol_ma20[-1]) * 1.1:
            vol_trend_score = 3
    score += vol_trend_score
    detail["vol_trend"] = vol_trend_score

    return score, detail


# ══════════════════════════════════════════════════════
# STEP 5 : 섹터 점수 (ADX + 추세 강도로 대체)
# ══════════════════════════════════════════════════════
def score_sector_trend(df):
    """
    ADX 추세 강도 + MACD 전환 + 상대위치 점수
    (섹터 데이터 미확보 시 ADX 기반 대체)
    반환: 점수(0-15), 상세
    """
    score  = 0
    detail = {}
    close  = df["Close"].values

    # ADX 추세 강도 (8점)
    adx = float(df["adx"].iloc[-1]) if "adx" in df.columns else 15
    if   adx > 35: adx_score = 8
    elif adx > 25: adx_score = 6
    elif adx > 20: adx_score = 4
    elif adx > 15: adx_score = 2
    else:          adx_score = 0
    score += adx_score
    detail["adx"] = round(adx, 1)
    detail["adx_score"] = adx_score

    # MACD 전환 점수 (4점)
    if "macd_hist" in df.columns and len(df) > 3:
        mh      = float(df["macd_hist"].iloc[-1])
        mh_prev = float(df["macd_hist"].iloc[-2])
        macd_score = 0
        if mh > 0 and mh > mh_prev:  macd_score = 4
        elif mh > 0:                  macd_score = 2
        elif mh > mh_prev:            macd_score = 1  # 히스토그램 증가 중 (전환 준비)
        score += macd_score
        detail["macd_score"] = macd_score

    # 52주 위치 점수 (3점): 저점 대비 30~100%가 가장 좋음
    low_52w = float(df["Low"].iloc[-HIGH_52W_DAYS:].min()) if len(df) >= HIGH_52W_DAYS else float(df["Low"].min())
    pos_pct  = (close[-1] - low_52w) / low_52w * 100 if low_52w > 0 else 999
    if   30 <= pos_pct <= 100: pos_score = 3
    elif 20 <= pos_pct <= 150: pos_score = 2
    elif pos_pct < 20:         pos_score = 1
    else:                      pos_score = 0
    score += pos_score
    detail["position_score"] = pos_score

    return score, detail


# ══════════════════════════════════════════════════════
# STEP 6 : 과열 감점
# ══════════════════════════════════════════════════════
def calc_penalty(df, position_type):
    """과열 신호에 따른 감점 계산"""
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    rsi   = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50
    ret_5d  = (close[-1] / close[-6]  - 1) * 100 if len(close) >= 6  else 0
    ret_20d = (close[-1] / close[-21] - 1) * 100 if len(close) >= 21 else 0

    rng  = high[-3:] - low[-3:]
    wick = high[-3:] - close[-3:]
    wick_ratio = float(np.mean(wick / rng)) if np.all(rng > 0) else 0

    penalty = 0
    reasons = []

    if   rsi > 80:  penalty += 15; reasons.append(f"RSI과열({rsi:.0f})")
    elif rsi > 76:  penalty += 8;  reasons.append(f"RSI주의({rsi:.0f})")
    elif rsi > 72:  penalty += 3;  reasons.append(f"RSI높음({rsi:.0f})")

    if   ret_5d > 20: penalty += 12; reasons.append(f"5일급등({ret_5d:.0f}%)")
    elif ret_5d > 15: penalty += 7;  reasons.append(f"5일과속({ret_5d:.0f}%)")
    elif ret_5d > 12: penalty += 3;  reasons.append(f"5일빠름({ret_5d:.0f}%)")

    if   wick_ratio > 0.45: penalty += 8; reasons.append("윗꼬리과다")
    elif wick_ratio > 0.35: penalty += 4; reasons.append("윗꼬리주의")

    # 추세 진행형은 추가 감점
    if position_type == "TREND_CONTINUE" and ret_20d > 30:
        penalty += 5; reasons.append(f"20일과속({ret_20d:.0f}%)")

    return penalty, reasons


# ══════════════════════════════════════════════════════
# 최종 점수 통합
# ══════════════════════════════════════════════════════
def compute_final_score(df, nh, position_type, position_pct):
    """
    전체 6단계 점수 통합
    - BREAKOUT_BOTTOM  : 기본 가점 +10 (최상급 보너스)
    - PULLBACK_REBREAK : 기본 가점 +5
    - TREND_CONTINUE   : 기본 가점 0
    """
    s3, d3 = score_breakout_strength(df, nh)
    s4, d4 = score_supply_demand(df, nh)
    s5, d5 = score_sector_trend(df)
    penalty, pen_reasons = calc_penalty(df, position_type)

    type_bonus = {"BREAKOUT_BOTTOM": 10, "PULLBACK_REBREAK": 5, "TREND_CONTINUE": 0}.get(position_type, 0)

    raw = s3 + s4 + s5 + type_bonus
    final = max(0, min(100, raw - penalty))

    detail = {
        **d3, **d4, **d5,
        "type_bonus": type_bonus,
        "penalty": -penalty,
        "penalty_reasons": pen_reasons,
        "s3_breakout": s3,
        "s4_supply":   s4,
        "s5_sector":   s5,
    }
    return final, detail


# ── 등급 ──────────────────────────────────────────────
def get_grade(score, position_type):
    type_min = {"BREAKOUT_BOTTOM": 45, "PULLBACK_REBREAK": 50, "TREND_CONTINUE": 60}.get(position_type, 999)
    if score < type_min:
        return "D", "△ 약함"
    if   score >= 80: return "A+", "🔥 최강"
    elif score >= 70: return "A",  "⭐ 강력"
    elif score >= 58: return "B",  "✓ 유효"
    else:             return "C",  "○ 보통"


# ── 유형 메타 ──────────────────────────────────────────
TYPE_META = {
    "BREAKOUT_BOTTOM" : {"ko": "바닥 탈출형", "emoji": "🏔️", "color": "#f59e0b", "desc": "장기 박스 후 첫 돌파 · 주도주 초기"},
    "PULLBACK_REBREAK": {"ko": "눌림 재돌파형", "emoji": "💎", "color": "#3b82f6", "desc": "1차 상승 후 조정 완료 · 2차 상승 시작"},
    "TREND_CONTINUE"  : {"ko": "추세 진행형", "emoji": "📈", "color": "#22c55e", "desc": "추세 유지 중 · 추가 필터 확인 필요"},
    "OVERHEATED"      : {"ko": "과열형", "emoji": "⛔", "color": "#ef4444", "desc": "분배 구간 가능성"},
}


# ── 단일 종목 분석 ─────────────────────────────────────
def analyze_stock(ticker_code, name, is_korean=True):
    try:
        df = fetch_daily_data(ticker_code, is_korean=is_korean)
        if df is None or len(df) < 100:
            return None

        df = calculate_indicators(df)
        if df is None:
            return None

        # STEP 1: 신고가 탐지
        nh = detect_new_high(df)
        if not nh["found"]:
            return None

        # STEP 2: 위치 판별
        position_type, position_pct, ret_20d = classify_position(df, nh)
        if position_type == "OVERHEATED":
            return None

        # STEP 3~6: 점수 계산
        score, detail = compute_final_score(df, nh, position_type, position_pct)

        grade, grade_label = get_grade(score, position_type)
        if grade == "D":
            return None

        meta = TYPE_META[position_type]
        current = float(df["Close"].iloc[-1])
        rsi = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 0
        adx = float(df["adx"].iloc[-1]) if "adx" in df.columns else 0

        high_52w = float(df["High"].iloc[-HIGH_52W_DAYS:].max()) if len(df) >= HIGH_52W_DAYS else float(df["High"].max())
        low_52w  = float(df["Low"].iloc[-HIGH_52W_DAYS:].min())  if len(df) >= HIGH_52W_DAYS else float(df["Low"].min())
        rng_52w  = high_52w - low_52w
        pos_range = (current - low_52w) / rng_52w * 100 if rng_52w > 0 else 50

        return {
            "ticker"        : ticker_code,
            "name"          : name,
            "score"         : round(score, 1),
            "grade"         : grade,
            "grade_label"   : grade_label,
            "type"          : position_type,
            "type_ko"       : meta["ko"],
            "type_emoji"    : meta["emoji"],
            "type_color"    : meta["color"],
            "type_desc"     : meta["desc"],
            "current_price" : current,
            "rsi"           : round(rsi, 1),
            "adx"           : round(adx, 1),
            "rvol"          : round(nh["rvol"], 2),
            "days_ago"      : nh["days_ago"],
            "breakout_price": round(nh["breakout_price"], 0),
            "first_breakout": nh["first_breakout"],
            "first_vol_surge": nh["first_vol_surge"],
            "ret_20d"       : round(ret_20d, 1),
            "position_pct"  : round(position_pct, 1),
            "pos_in_range"  : round(pos_range, 1),
            "high_52w"      : round(high_52w, 0),
            "low_52w"       : round(low_52w, 0),
            "price_held"    : detail.get("price_held", False),
            "vol_dried"     : detail.get("vol_dried", False),
            "above_ma5"     : detail.get("above_ma5", False),
            "score_detail"  : detail,
        }

    except:
        return None


# ── 전체 스캔 ──────────────────────────────────────────
def screen_stocks(top_n=150, progress_cb=None):
    tickers = get_fallback_tickers(top_n)
    results = []
    total   = len(tickers)

    for i, t in enumerate(tickers):
        if progress_cb:
            progress_cb(i, total, t["name"], len(results))
        r = analyze_stock(t["ticker"], t["name"], is_korean=True)
        if r:
            results.append(r)
        time.sleep(0.08)

    # 정렬: 바닥탈출 > 눌림재돌파 > 추세진행, 같은 유형 내 점수 내림차순
    order = {"BREAKOUT_BOTTOM": 0, "PULLBACK_REBREAK": 1, "TREND_CONTINUE": 2}
    results.sort(key=lambda x: (order.get(x["type"], 9), -x["score"]))
    return results


if __name__ == "__main__":
    print("=== 신고가 위치 판별 엔진 ===")
    results = screen_stocks(top_n=50)
    print(f"\n발견: {len(results)}개\n")
    for r in results[:10]:
        fb = "★첫돌파" if r["first_breakout"] else ""
        fv = "★첫급증" if r["first_vol_surge"] else ""
        print(f"[{r['grade_label']}] {r['name']}({r['ticker']}) "
              f"{r['type_emoji']}{r['type_ko']} 점수:{r['score']} "
              f"RVOL:{r['rvol']}x {fb}{fv} "
              f"신고가:{r['days_ago']}일전")
