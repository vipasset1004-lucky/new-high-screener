"""
신고가 지속 상승 스크리너 (New High Continuation Screener)
- 신고가 후 계속 오를 종목 vs 꺾일 종목 구분
- 3가지 유형 분류: ① 추세시작형 ② 추세중간형 ③ 고점형(제외)
- 핵심 필터: 수렴 + 돌파강도 + 3일 유지력 + 과열 제거
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime, timedelta
import json
import time
import requests
import warnings
import os

warnings.filterwarnings("ignore")

# ── 설정 ──────────────────────────────────────────────
SCAN_DAYS = 300          # 일봉 데이터 기간
NEW_HIGH_WINDOW = 10     # 신고가 탐지 기간 (최근 N일 내)
HIGH_52W_DAYS = 252      # 52주 = 약 252 거래일


# ── 종목 리스트 ────────────────────────────────────────
def get_fallback_tickers(top_n=150):
    """주요 종목 기본 리스트 (거래대금 상위)"""
    kospi = [
        # 시가총액 상위
        ("005930", "삼성전자"), ("000660", "SK하이닉스"), ("005380", "현대차"),
        ("035420", "NAVER"), ("000270", "기아"), ("051910", "LG화학"),
        ("006400", "삼성SDI"), ("035720", "카카오"), ("105560", "KB금융"),
        ("055550", "신한지주"), ("086790", "하나금융지주"), ("032830", "삼성생명"),
        ("028260", "삼성물산"), ("207940", "삼성바이오로직스"), ("068270", "셀트리온"),
        ("247540", "에코프로비엠"), ("373220", "LG에너지솔루션"), ("096770", "SK이노베이션"),
        ("011200", "HMM"), ("003550", "LG"), ("010130", "고려아연"),
        ("010950", "S-Oil"), ("012330", "현대모비스"), ("018260", "삼성에스디에스"),
        ("009150", "삼성전기"), ("000100", "유한양행"), ("011170", "롯데케미칼"),
        ("017670", "SK텔레콤"), ("030200", "KT"), ("032640", "LG유플러스"),
        ("066570", "LG전자"), ("003490", "대한항공"), ("010140", "삼성중공업"),
        ("042660", "한화오션"), ("009540", "HD한국조선해양"), ("267250", "HD현대"),
        ("086280", "현대글로비스"), ("000810", "삼성화재"), ("088350", "한화생명"),
        ("139480", "이마트"), ("004020", "현대제철"), ("005490", "POSCO홀딩스"),
        ("097950", "CJ제일제당"), ("011780", "금호석유"), ("161390", "한국타이어앤테크놀로지"),
        ("082740", "한화엔진"), ("064350", "현대로템"), ("298040", "효성중공업"),
        ("012450", "한화에어로스페이스"), ("047810", "한국항공우주"),
        # 중형주 모멘텀
        ("247540", "에코프로비엠"), ("086520", "에코프로"), ("011040", "OCI홀딩스"),
        ("293490", "카카오뱅크"), ("035760", "CJ ENM"), ("041510", "에스엠"),
        ("352820", "하이브"), ("003230", "삼양식품"), ("007310", "오뚜기"),
        ("271560", "오리온"), ("002790", "아모레퍼시픽"), ("090430", "아모레G"),
        ("069620", "대웅제약"), ("128940", "한미약품"), ("006800", "미래에셋증권"),
        ("016360", "삼성증권"), ("071050", "한국금융지주"), ("039490", "키움증권"),
        ("034730", "SK"), ("018880", "한온시스템"), ("196170", "알테오젠"),
        ("145020", "휴젤"), ("091990", "셀트리온헬스케어"), ("000120", "CJ대한통운"),
        ("097520", "중앙첨단소재"), ("011790", "SKC"), ("006650", "대한유화"),
    ]
    kosdaq = [
        ("247540", "에코프로비엠"), ("086520", "에코프로"), ("091990", "셀트리온헬스케어"),
        ("041510", "에스엠"), ("035900", "JYP Ent."), ("122870", "와이지엔터테인먼트"),
        ("293490", "카카오뱅크"), ("259960", "크래프톤"),
        ("263750", "펄어비스"), ("194480", "데브시스터즈"), ("251270", "넷마블"),
        ("036570", "엔씨소프트"), ("112040", "위메이드"), ("095660", "네오위즈"),
        ("045360", "나스미디어"), ("067160", "아프리카TV"), ("035080", "인터파크트리플"),
        ("145020", "휴젤"), ("006840", "AK홀딩스"), ("214150", "클래시스"),
        ("196170", "알테오젠"), ("326030", "SK바이오팜"), ("272210", "한화시스템"),
        ("237690", "에스티팜"), ("199800", "툴젠"), ("086900", "메디오젠"),
        ("950130", "엑스페릭스"), ("241590", "화승엔터프라이즈"),
        ("357780", "솔브레인"), ("036800", "나이스정보통신"),
        ("053800", "안랩"), ("192080", "더블유게임즈"), ("080160", "모두투어"),
        ("095700", "제넥신"), ("108675", "LX하우시스"),
        ("158430", "아톤"), ("225570", "넥슨게임즈"),
    ]

    all_tickers = []
    seen = set()
    for code, name in kospi + kosdaq:
        if code not in seen:
            all_tickers.append({"ticker": code, "name": name})
            seen.add(code)

    return all_tickers[:top_n]


def get_krx_tickers(market="ALL", top_n=200):
    """KRX 전체 종목 + 거래대금 기준 정렬"""
    try:
        print("  KRX 종목 리스트 다운로드 중...")
        url = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
        tables = pd.read_html(url, encoding="euc-kr")
        df = tables[0]
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
        df = df[df["종목코드"].str.match(r"^\d{6}$")]
        print(f"  KRX 전체 종목: {len(df)}개")

        cap_data = []
        for _, row in df.iterrows():
            ticker = row["종목코드"]
            name = row["회사명"]
            for suffix in [".KS", ".KQ"]:
                try:
                    info = yf.Ticker(f"{ticker}{suffix}")
                    hist = info.history(period="5d")
                    if hist.empty:
                        continue
                    avg_vol = hist["Volume"].mean()
                    price = hist["Close"].iloc[-1]
                    mkt = "KOSPI" if suffix == ".KS" else "KOSDAQ"
                    if market != "ALL" and mkt != market:
                        continue
                    cap_data.append({
                        "ticker": ticker, "name": name,
                        "trade_value": avg_vol * price,
                    })
                    break
                except:
                    continue
            if len(cap_data) >= top_n * 3:
                break
            time.sleep(0.1)

        cap_data.sort(key=lambda x: x["trade_value"], reverse=True)
        return [{"ticker": x["ticker"], "name": x["name"]} for x in cap_data[:top_n]]
    except Exception as e:
        print(f"[ERROR] KRX 수집 실패: {e} → fallback 사용")
        return get_fallback_tickers(top_n)


# ── 데이터 수집 ────────────────────────────────────────
def fetch_daily_data(ticker_code, is_korean=True, days=SCAN_DAYS):
    """일봉 데이터 수집"""
    suffixes = [".KS", ".KQ"] if is_korean else [""]
    for suffix in suffixes:
        try:
            symbol = f"{ticker_code}{suffix}"
            end = datetime.now()
            start = end - timedelta(days=days + 60)
            df = yf.download(symbol, start=start, end=end,
                             auto_adjust=True, progress=False, timeout=15)
            if df is None or len(df) < 60:
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
    """RSI, MACD, ADX, BB, ATR 계산"""
    try:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # RSI
        rsi = RSIIndicator(close=close, window=14)
        df["rsi"] = rsi.rsi()

        # MACD
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # ADX
        adx = ADXIndicator(high=high, low=low, close=close, window=14)
        df["adx"] = adx.adx()

        # Bollinger Bands
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        # ATR
        atr = AverageTrueRange(high=high, low=low, close=close, window=14)
        df["atr"] = atr.average_true_range()

        # 이동평균
        df["ma5"]   = close.rolling(5).mean()
        df["ma10"]  = close.rolling(10).mean()
        df["ma20"]  = close.rolling(20).mean()
        df["ma60"]  = close.rolling(60).mean()
        df["ma120"] = close.rolling(120).mean()

        # 거래대금 (원)
        df["trade_value"] = close * volume

        return df
    except Exception as e:
        return None


# ── 신고가 탐지 ────────────────────────────────────────
def detect_new_high(df):
    """
    최근 NEW_HIGH_WINDOW일 내 52주 신고가 발생 여부 탐지.
    반환: {
        "found": bool,
        "days_ago": int,          # 신고가 발생 며칠 전인지 (0=오늘)
        "breakout_price": float,  # 신고가 돌파 기준가
        "breakout_volume": float, # 신고가 발생일 거래대금
        "avg_volume_20d": float,  # 20일 평균 거래대금
        "rvol": float,            # 상대거래량 비율
    }
    """
    if len(df) < HIGH_52W_DAYS + NEW_HIGH_WINDOW:
        return {"found": False}

    close = df["Close"].values
    high = df["High"].values
    trade_val = df["trade_value"].values

    result = {"found": False}

    for days_ago in range(0, NEW_HIGH_WINDOW + 1):
        idx = -(days_ago + 1)  # -1=어제, -2=이틀전...
        if abs(idx) > len(close):
            break

        # 해당일 기준 52주 전 ~ 전날 고가
        window_end = len(close) + idx       # 해당일 인덱스
        window_start = max(0, window_end - HIGH_52W_DAYS)
        if window_end - window_start < 50:
            continue

        prev_52w_high = np.max(high[window_start:window_end])
        today_close = close[idx]
        today_high = high[idx]

        # 당일 고가가 52주 고가를 돌파하면 신고가
        if today_high >= prev_52w_high * 0.999:
            # 20일 평균 거래대금
            tv_window = trade_val[max(0, window_end - 20):window_end]
            avg_tv = np.mean(tv_window) if len(tv_window) > 0 else 1
            breakout_tv = trade_val[idx]
            rvol = breakout_tv / avg_tv if avg_tv > 0 else 1

            result = {
                "found": True,
                "days_ago": days_ago,
                "breakout_price": float(prev_52w_high),
                "breakout_high": float(today_high),
                "breakout_volume": float(breakout_tv),
                "avg_volume_20d": float(avg_tv),
                "rvol": float(rvol),
            }
            break

    return result


# ── 유형 분류 ──────────────────────────────────────────
def classify_type(df, nh):
    """
    신고가 유형 분류:
    START  = ① 추세 시작형 (수렴 후 첫 돌파, 가장 중요)
    MIDDLE = ② 추세 중간형 (이미 추세 중, 눌림 후 재돌파)
    PEAK   = ③ 고점형 (과열, 제거 대상)
    """
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values

    current = close[-1]

    # 52주 저점 대비 현재 위치
    low_52w = np.min(low[-HIGH_52W_DAYS:]) if len(low) >= HIGH_52W_DAYS else np.min(low)
    position_pct = (current - low_52w) / low_52w * 100 if low_52w > 0 else 999

    # 최근 RSI
    rsi = df["rsi"].iloc[-1] if "rsi" in df.columns else 50
    rsi_5d_avg = df["rsi"].iloc[-5:].mean() if "rsi" in df.columns else 50

    # 최근 5일 상승률
    ret_5d = ((current / close[-6]) - 1) * 100 if len(close) >= 6 else 0

    # ADX
    adx = df["adx"].iloc[-1] if "adx" in df.columns else 15

    # 윗꼬리 비율 (최근 3일 평균)
    recent_range = high[-3:] - low[-3:]
    recent_upper_wick = high[-3:] - close[-3:]
    wick_ratio = np.mean(recent_upper_wick / recent_range) if np.all(recent_range > 0) else 0

    # 과열 판정: 어떤 하나라도 해당하면 PEAK
    is_overheated = (
        rsi_5d_avg > 80 or
        (rsi > 78 and ret_5d > 12) or
        (ret_5d > 20) or
        (wick_ratio > 0.45 and rsi > 72)
    )

    if is_overheated:
        return "PEAK", position_pct

    # 추세 시작형: 저점 대비 30-100%, ADX 아직 낮음 (막 추세 시작)
    if 20 <= position_pct <= 120 and adx < 35:
        return "START", position_pct

    # 추세 중간형: ADX 높고 이미 상승 진행 중
    if adx >= 20 and position_pct > 30:
        return "MIDDLE", position_pct

    return "START", position_pct


# ── 수렴 감지 ──────────────────────────────────────────
def measure_convergence(df):
    """
    돌파 직전 수렴(수축) 강도 측정.
    BB width 기준 최근 수축률 반환 (낮을수록 강한 수렴)
    """
    if "bb_width" not in df.columns or len(df) < 60:
        return 1.0, False

    bb_width = df["bb_width"].dropna()
    if len(bb_width) < 30:
        return 1.0, False

    # 최근 10일 평균 BB폭 vs 60일 최대 BB폭
    recent_width = bb_width.iloc[-10:].mean()
    max_width_60d = bb_width.iloc[-60:].max()

    if max_width_60d <= 0:
        return 1.0, False

    ratio = recent_width / max_width_60d  # 낮을수록 수렴 강함
    has_convergence = ratio < 0.60

    return float(ratio), has_convergence


# ── 유지력 체크 ────────────────────────────────────────
def check_holding_power(df, nh):
    """
    신고가 후 N일 동안 버팀 여부 체크.
    반환: {
        "days_held": int,
        "price_held": bool,      # 신고가 돌파가 이탈 없음
        "volume_dried": bool,    # 거래량 감소 패턴
        "above_ma5": bool,       # 5일선 위 유지
        "holding_score": float,  # 0-1
    }
    """
    days_ago = nh.get("days_ago", 0)

    if days_ago < 1:
        # 오늘이 신고가 → 아직 검증 불가
        return {
            "days_held": 0,
            "price_held": True,
            "volume_dried": False,
            "above_ma5": df["Close"].iloc[-1] > df["ma5"].iloc[-1],
            "holding_score": 0.3,
        }

    close = df["Close"].values
    volume = df["Volume"].values
    ma5 = df["ma5"].values

    breakout_price = nh["breakout_price"]
    breakout_vol = nh["breakout_volume"]

    # 신고가 이후 날들의 가격/거래량 체크
    post_close = close[-(days_ago):]
    post_vol = volume[-(days_ago):]

    # 가격 유지: 신고가 기준가 * 0.97 이상 유지
    price_held = np.all(post_close >= breakout_price * 0.97) if len(post_close) > 0 else False

    # 거래량 감소: 신고가 당일 대비 이후 거래량 감소 (눌림에서 거래량 줄어야 좋음)
    if len(post_vol) >= 2 and breakout_vol > 0:
        avg_post_vol = np.mean(post_vol) * close[-1]  # 거래대금으로 변환
        volume_dried = avg_post_vol < breakout_vol * 0.7
    else:
        volume_dried = False

    # 5일선 위 유지
    above_ma5 = close[-1] > ma5[-1] if not np.isnan(ma5[-1]) else False

    # 종합 유지력 점수
    score = 0.0
    if price_held:
        score += 0.5
    if volume_dried:
        score += 0.3
    if above_ma5:
        score += 0.2

    return {
        "days_held": min(days_ago, 10),
        "price_held": bool(price_held),
        "volume_dried": bool(volume_dried),
        "above_ma5": bool(above_ma5),
        "holding_score": float(score),
    }


# ── MA 정배열 체크 ─────────────────────────────────────
def check_ma_alignment(df):
    """MA 정배열 확인 (5>20>60>120)"""
    try:
        ma5   = df["ma5"].iloc[-1]
        ma20  = df["ma20"].iloc[-1]
        ma60  = df["ma60"].iloc[-1]
        ma120 = df["ma120"].iloc[-1]
        price = df["Close"].iloc[-1]

        if any(pd.isna([ma5, ma20, ma60, ma120])):
            # MA120 없으면 3단계로만 체크
            if not pd.isna(ma5) and not pd.isna(ma20) and not pd.isna(ma60):
                if price > ma5 > ma20 > ma60:
                    return "FULL_3", 3
                elif price > ma20 > ma60:
                    return "PARTIAL", 2
            return "NONE", 0

        if price > ma5 > ma20 > ma60 > ma120:
            return "FULL_4", 4
        elif price > ma5 > ma20 > ma60:
            return "FULL_3", 3
        elif price > ma20 > ma60:
            return "PARTIAL", 2
        elif price > ma60:
            return "WEAK", 1
        else:
            return "NONE", 0
    except:
        return "NONE", 0


# ── 종합 점수 계산 ─────────────────────────────────────
def score_stock(df, nh, hold, conv_ratio, ma_type, stock_type, position_pct):
    """
    최종 점수 계산 (100점 만점)

    항목:
    - 위치 점수:    15점  (52주 저점 대비 초중반 위치)
    - 수렴 점수:    25점  (돌파 직전 BB 수축)
    - 돌파 거래대금: 20점  (RVOL)
    - 유지력:       25점  (3일 버팀 체크)
    - MA 정배열:    10점
    - 과열 감점:   -25점
    - ADX 보너스:  +5점
    """
    score = 0
    detail = {}

    close = df["Close"].values
    rsi = df["rsi"].iloc[-1] if "rsi" in df.columns else 50
    adx = df["adx"].iloc[-1] if "adx" in df.columns else 15
    macd_hist = df["macd_hist"].iloc[-1] if "macd_hist" in df.columns else 0
    macd_hist_prev = df["macd_hist"].iloc[-2] if "macd_hist" in df.columns and len(df) > 2 else 0
    ret_5d = ((close[-1] / close[-6]) - 1) * 100 if len(close) >= 6 else 0
    ret_20d = ((close[-1] / close[-21]) - 1) * 100 if len(close) >= 21 else 0

    # 1. 위치 점수 (15점): 52주 저점 대비 초중반이 가장 좋음
    if 20 <= position_pct <= 80:
        pos_score = 15
    elif 80 < position_pct <= 150:
        pos_score = 10
    elif 150 < position_pct <= 250:
        pos_score = 5
    elif position_pct < 20:
        pos_score = 8   # 막 저점 탈출, 아직 이름
    else:
        pos_score = 0   # 이미 너무 많이 옴
    score += pos_score
    detail["position"] = pos_score

    # 2. 수렴 점수 (25점): BB 수축률 기반
    if conv_ratio < 0.35:
        conv_score = 25
    elif conv_ratio < 0.50:
        conv_score = 18
    elif conv_ratio < 0.65:
        conv_score = 10
    elif conv_ratio < 0.80:
        conv_score = 4
    else:
        conv_score = 0
    score += conv_score
    detail["convergence"] = conv_score

    # 3. 돌파 거래대금 (20점): RVOL
    rvol = nh.get("rvol", 1.0)
    if rvol >= 3.0:
        vol_score = 20
    elif rvol >= 2.0:
        vol_score = 15
    elif rvol >= 1.5:
        vol_score = 10
    elif rvol >= 1.2:
        vol_score = 5
    else:
        vol_score = 0
    score += vol_score
    detail["breakout_volume"] = vol_score

    # 4. 유지력 점수 (25점)
    hold_score_raw = hold.get("holding_score", 0)
    days_held = hold.get("days_held", 0)

    # 3일 이상 버텼으면 최대 점수
    if days_held >= 3:
        hold_pts = int(hold_score_raw * 25)
    elif days_held == 2:
        hold_pts = int(hold_score_raw * 18)
    elif days_held == 1:
        hold_pts = int(hold_score_raw * 10)
    else:
        hold_pts = 5   # 오늘이 신고가 (검증 전)
    score += hold_pts
    detail["holding"] = hold_pts

    # 5. MA 정배열 (10점)
    ma_scores = {"FULL_4": 10, "FULL_3": 8, "PARTIAL": 4, "WEAK": 1, "NONE": 0}
    ma_score = ma_scores.get(ma_type, 0)
    score += ma_score
    detail["ma_alignment"] = ma_score

    # 6. 과열 감점
    penalty = 0
    high = df["High"].values
    low_arr = df["Low"].values
    recent_range = high[-3:] - low_arr[-3:]
    recent_wick = high[-3:] - close[-3:]
    wick_ratio = np.mean(recent_wick / recent_range) if np.all(recent_range > 0) else 0

    if rsi > 82:
        penalty += 20
    elif rsi > 78:
        penalty += 12
    elif rsi > 75:
        penalty += 6

    if ret_5d > 20:
        penalty += 15
    elif ret_5d > 15:
        penalty += 8
    elif ret_5d > 12:
        penalty += 4

    if wick_ratio > 0.45:
        penalty += 8
    elif wick_ratio > 0.35:
        penalty += 4

    score -= penalty
    detail["penalty"] = -penalty

    # 7. ADX 보너스 (+5점)
    adx_bonus = 0
    if adx > 30:
        adx_bonus = 5
    elif adx > 25:
        adx_bonus = 3
    elif adx > 20:
        adx_bonus = 1
    score += adx_bonus
    detail["adx_bonus"] = adx_bonus

    # 8. MACD 전환 보너스 (+3점)
    macd_bonus = 0
    if macd_hist > 0 and macd_hist > macd_hist_prev:
        macd_bonus = 3
    elif macd_hist > 0:
        macd_bonus = 1
    score += macd_bonus
    detail["macd_bonus"] = macd_bonus

    return max(0, min(100, score)), detail


# ── 등급 계산 ──────────────────────────────────────────
def get_grade(score, stock_type):
    if stock_type == "PEAK":
        return "PEAK", "⛔ 고점주의"
    if score >= 80:
        return "A", "🔥 최강"
    elif score >= 70:
        return "B", "⭐ 강력"
    elif score >= 55:
        return "C", "✓ 유효"
    else:
        return "D", "△ 약함"


# ── 단일 종목 분석 ─────────────────────────────────────
def analyze_stock(ticker_code, name, is_korean=True):
    """단일 종목 신고가 지속 분석"""
    try:
        df = fetch_daily_data(ticker_code, is_korean=is_korean)
        if df is None or len(df) < 100:
            return None

        df = calculate_indicators(df)
        if df is None:
            return None

        # 신고가 탐지
        nh = detect_new_high(df)
        if not nh["found"]:
            return None

        # 유형 분류
        stock_type, position_pct = classify_type(df, nh)
        if stock_type == "PEAK":
            return None   # 고점형 제외

        # 세부 분석
        hold = check_holding_power(df, nh)
        conv_ratio, has_conv = measure_convergence(df)
        ma_type, ma_level = check_ma_alignment(df)

        # 점수 계산
        score, detail = score_stock(df, nh, hold, conv_ratio, ma_type, stock_type, position_pct)

        # 너무 낮은 점수 제외
        if score < 30:
            return None

        grade, grade_label = get_grade(score, stock_type)
        if grade == "D":
            return None

        # 현재 시세 정보
        current_price = float(df["Close"].iloc[-1])
        rsi = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 0
        adx = float(df["adx"].iloc[-1]) if "adx" in df.columns else 0

        # 52주 범위
        high_52w = float(df["High"].iloc[-HIGH_52W_DAYS:].max()) if len(df) >= HIGH_52W_DAYS else float(df["High"].max())
        low_52w = float(df["Low"].iloc[-HIGH_52W_DAYS:].min()) if len(df) >= HIGH_52W_DAYS else float(df["Low"].min())

        # 신고가 위치 (52주 범위 내 %)
        range_52w = high_52w - low_52w
        position_in_range = ((current_price - low_52w) / range_52w * 100) if range_52w > 0 else 50

        type_labels = {
            "START": {"ko": "추세 시작형", "emoji": "🚀"},
            "MIDDLE": {"ko": "추세 중간형", "emoji": "📈"},
        }
        t_info = type_labels.get(stock_type, {"ko": stock_type, "emoji": "📊"})

        return {
            "ticker": ticker_code,
            "name": name,
            "score": round(score, 1),
            "grade": grade,
            "grade_label": grade_label,
            "type": stock_type,
            "type_ko": t_info["ko"],
            "type_emoji": t_info["emoji"],
            "current_price": current_price,
            "rsi": round(rsi, 1),
            "adx": round(adx, 1),
            "rvol": round(nh["rvol"], 2),
            "days_ago": nh["days_ago"],
            "breakout_price": round(nh["breakout_price"], 0),
            "position_in_range": round(position_in_range, 1),
            "position_pct": round(position_pct, 1),
            "has_convergence": bool(has_conv),
            "conv_ratio": round(conv_ratio, 2),
            "ma_type": ma_type,
            "ma_level": ma_level,
            "price_held": hold["price_held"],
            "volume_dried": hold["volume_dried"],
            "above_ma5": hold["above_ma5"],
            "days_held": hold["days_held"],
            "high_52w": round(high_52w, 0),
            "low_52w": round(low_52w, 0),
            "score_detail": detail,
        }

    except Exception as e:
        return None


# ── 전체 스캔 ──────────────────────────────────────────
def screen_stocks(market="ALL", top_n=150, progress_cb=None):
    """전체 종목 스캔"""
    tickers = get_fallback_tickers(top_n)
    results = []
    total = len(tickers)

    for i, t in enumerate(tickers):
        ticker = t["ticker"]
        name = t["name"]

        if progress_cb:
            progress_cb(i, total, name, len(results))

        result = analyze_stock(ticker, name, is_korean=True)
        if result:
            results.append(result)

        time.sleep(0.08)

    # 점수 내림차순, 추세시작형 우선
    results.sort(key=lambda x: (
        0 if x["type"] == "START" else 1,
        -x["score"]
    ))

    return results


if __name__ == "__main__":
    print("=== 신고가 지속 상승 스크리너 ===")
    results = screen_stocks(top_n=50)
    print(f"\n총 {len(results)}개 종목 발견\n")
    for r in results[:10]:
        print(f"[{r['grade_label']}] {r['name']}({r['ticker']}) "
              f"점수:{r['score']} 유형:{r['type_emoji']}{r['type_ko']} "
              f"신고가:{r['days_ago']}일전 RVOL:{r['rvol']}x "
              f"수렴:{r['has_convergence']} 유지:{r['price_held']}")
