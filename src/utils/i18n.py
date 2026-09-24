"""
v5.8 - Arabic phrasebook for the Telegram presentation layer.

GOAL: every message the user receives on Telegram is in Arabic, WITHOUT
touching strategy/risk logic (which stays English for tests + logs).

Design (three stages, in order):
  1. EXACT   - full-string dictionary for fixed phrases (strategy reasons,
               labels, close reasons).
  2. RULES   - ordered (regex -> template) list for parameterised f-strings;
               applied sequentially with re.sub so COMPOSITE strings built
               with " + ..." concatenation translate piece by piece.
               Capture groups keep the numbers intact.
  3. TOKENS  - always-applied safe substitutions: candlestick pattern names,
               raw direction words (bullish/bearish/neutral) and the
               "->" arrow cosmetic. These tokens are unambiguous.

Anything unmatched passes through unchanged (graceful degradation) - a
future English string can never garble a message, it just stays English
until a rule for it is added.

Public API:
  tr(text)           - translate any dynamic fragment
  ar_direction(d)    - bullish/bearish/neutral -> صاعد/هابط/محايد
  ar_mode(mode)      - PAPER/LIVE -> تجريبي/حقيقي (with Latin tag kept)
  ar_entry_type(t)   - market/limit -> سوق/أمر محدد
  ar_ichimoku(field, value) - ichimoku raw values -> Arabic
"""
import re
from typing import Dict, List, Tuple

# ------------------------------------------------------------------
# 1) EXACT full-string map
# ------------------------------------------------------------------
_EXACT: Dict[str, str] = {
    # --- close / manage reasons (risk manager) ---
    "Stop Loss Hit": "ضرب وقف الخسارة",
    "Stop Loss Hit (BE)": "ضرب وقف الخسارة (بعد التعادل)",
    "Take Profit Hit": "تحقق هدف جني الأرباح",
    "Take Profit 2 Hit": "تحقق الهدف الثاني TP2",
    "TP1 Partial": "جني جزئي عند TP1",
    # --- trailing / locks (risk manager) ---
    "Ichimoku regime flipped bearish": "انقلاب نظام إيشيموكي إلى هابط",
    "Ichimoku regime flipped bullish": "انقلاب نظام إيشيموكي إلى صاعد",
    # --- triple confluence trend ---
    "EMA50 above EMA200 (uptrend confirmed)": "EMA50 فوق EMA200 (اتجاه صاعد مؤكد)",
    "OBV rising (accumulation)": "OBV صاعد (تراكم شرائي)",
    # --- volatility breakout ---
    "Squeeze + price above middle BB (bullish setup)":
        "انضغاط بولينجر + سعر فوق الباند الأوسط (إعداد صاعد)",
    "Squeeze + price below middle BB (bearish setup)":
        "انضغاط بولينجر + سعر تحت الباند الأوسط (إعداد هابط)",
    "Price above EMA 50 (trend-aligned long)": "السعر فوق EMA50 (متوافق مع الاتجاه)",
    "Caution: breakout against EMA 50 trend": "تحذير: اختراق عكس اتجاه EMA50",
    # --- bb mean reversion ---
    "Stoch cross up (outside deep oversold)": "تقاطع ستوكاستك صاعد (خارج التشبع العميق)",
    "Stoch turning up in oversold zone": "ستوكاستك يتجه صعوداً داخل التشبع البيعي",
    "Bullish close (no falling knife)": "إغلاق صاعد (بدون سكين ساقطة)",
    # --- macd breakout ---
    "FRESH MACD cross above signal line": "تقاطع MACD صاعد طازج فوق خط الإشارة",
    "Zero-line cross (below zero - earliest entry)":
        "تقاطع عند خط الصفر (تحت الصفر - أقرب دخول)",
    "Cross just above zero line": "تقاطع فوق خط الصفر مباشرة",
    "Histogram flipped positive": "الهيستوجرام انقلب موجباً",
    "Histogram positive and rising": "الهيستوجرام موجب ومتزايد",
    "Histogram positive": "الهيستوجرام موجب",
    "EMA50 sloping up": "EMA50 بميل صاعد",
    # --- trend pullback ---
    "Strong uptrend (EMA 9 > 21 > 50, price above EMA 50)":
        "اتجاه صاعد قوي (EMA9 > 21 > 50 والسعر فوق EMA50)",
    "Uptrend (EMA 9 > 21)": "اتجاه صاعد (EMA9 > 21)",
    "High volume but bearish candle": "حجم مرتفع لكن شمعة هابطة",
    "MACD histogram rising": "هيستوجرام MACD متزايد",
    # --- liquidity sweep ---
    "Volume climax (capitulation)": "ذروة حجم (استسلام)",
    "Wyckoff Spring detected": "زنبرك وايكوف مكتشف",
    # --- fibonacci labels ---
    "Market - price in Fibonacci golden pocket":
        "سوق - السعر داخل الجيب الذهبي لفيبوناتشي",
    "Limit - Fibonacci 0.618 golden pocket":
        "أمر محدد - فيبوناتشي 0.618 (الجيب الذهبي)",
    "Market - reversal from swing low (discount zone)":
        "سوق - انعكاس من قاع الموجة (منطقة خصم)",
    "Market - reversal from swing high (premium zone)":
        "سوق - انعكاس من قمة الموجة (منطقة علاوة)",
    "Min R/R target": "هدف الحد الأدنى للعائد/المخاطرة",
    "Fib extension (projected)": "امتداد فيبوناتشي (متوقع)",
    "Market - structural reversal entry": "سوق - دخول انعكاس هيكلي",
    "Market - structural breakout entry": "سوق - دخول اختراق هيكلي",
    "Market - neutral zone": "سوق - منطقة محايدة",
    "Nearest support": "أقرب دعم",
    "Nearest resistance": "أقرب مقاومة",
    "Projected extension": "امتداد متوقع",
    "Neutral target": "هدف محايد",
    "Extended target": "هدف ممتد",
    "Elliott wave 5 projection": "إسقاط الموجة 5 لإليوت",
    # --- confluence ---
    "TP2 promoted to Elliott wave 5 projection":
        "تمت ترقية TP2 إلى إسقاط الموجة 5 لإليوت",
    # --- elliott implications ---
    "Impulse complete - correction expected, avoid fresh continuation entries":
        "الموجة الدافعة مكتملة - يُتوقع تصحيح، تجنب دخولات استمرار جديدة",
    "Wave 4 pullback - wave 5 ahead, good R/R zone":
        "تصحيح الموجة 4 - الموجة 5 قادمة، منطقة عائد/مخاطرة جيدة",
    "Wave 3 in progress - strongest phase of the cycle":
        "الموجة 3 جارية - أقوى مرحلة في الدورة",
    "ABC bounce still in progress - wait for completion":
        "ارتداد ABC ما زال جارياً - انتظر الاكتمال",
    # --- pending entries ---
    "price above entry zone - waiting for pullback":
        "السعر أعلى منطقة الدخول - بانتظار الارتداد",
    # --- market/context ---
    "No specific signals": "لا توجد إشارات محددة",
}

# ------------------------------------------------------------------
# 2) Ordered regex rules for parameterised strings
#    (more specific patterns FIRST)
# ------------------------------------------------------------------
_RULES: List[Tuple[str, str]] = [
    # --- time stops ---
    (r"Time stop: stale trade \((.+)\)", r"وقف زمني: صفقة راكدة (\1)"),
    (r"Time stop: max hold \((.+)\)", r"وقف زمني: أقصى مدة حيازة (\1)"),
    # --- chandelier ---
    (r"Chandelier trail ([\d.]+)xATR \(peak ([^)]+)\)",
     r"وقف ثريا \1×ATR (القمة \2)"),
    (r"Chandelier trail ([\d.]+)xATR \(trough ([^)]+)\)",
     r"وقف ثريا \1×ATR (القاع \2)"),
    # --- ladder + locks ---
    (r"Trailing stop \(profit \+?([^)]+)\)", r"وقف متتبع (الربح \1)"),
    (r"Lock \+2% profit \(current \+?([^)]+)\)", r"تثبيت ربح +2% (الحالي \1)"),
    (r"Lock \+1% profit \(current \+?([^)]+)\)", r"تثبيت ربح +1% (الحالي \1)"),
    (r"Break-even \(profit \+?([^)]+)\)", r"نقطة التعادل (الربح \1)"),
    (r"\+ Extended TP \(bullish continuation\)", r"+ تمديد الهدف (استمرار صاعد)"),
    (r"\+ Extended TP \(bearish continuation\)", r"+ تمديد الهدف (استمرار هابط)"),
    (r"\+ Locked ([\d.]+)% profit", r"+ تثبيت \1% من الربح"),
    # --- structural exits ---
    (r"Structural exit: (.+)", r"خروج هيكلي: \1"),
    (r"Structural: (.+)", r"هيكلي: \1"),
    (r"Opposite bullish signal \(conf ([\d.]+)% >= ([\d.]+)%\) - closing now",
     r"إشارة صاعدة معاكسة (ثقة \1% ≥ \2%) - إغلاق فوري"),
    (r"Opposite bearish signal \(conf ([\d.]+)% >= ([\d.]+)%\) - closing now",
     r"إشارة هابطة معاكسة (ثقة \1% ≥ \2%) - إغلاق فوري"),
    (r"Opposite bullish pressure defence \(([^)]+)\)",
     r"دفاع من ضغط صاعد معاكس (\1)"),
    (r"Opposite bearish pressure defence \(([^)]+)\)",
     r"دفاع من ضغط هابط معاكس (\1)"),
    (r"Kijun defence \(([^)]+)\)", r"دفاع عند خط كيجون (\1)"),
    (r"Tenkan cross-down defence \(([^)]+)\)", r"دفاع تقاطع تينكان هابطاً (\1)"),
    (r"Tenkan cross-up defence \(([^)]+)\)", r"دفاع تقاطع تينكان صاعداً (\1)"),
    # --- triple confluence trend ---
    (r"Price above EMA200 \(([^)]+)\)", r"السعر فوق EMA200 (\1)"),
    (r"Price below EMA200 \(([^)]+)\)", r"السعر تحت EMA200 (\1)"),
    (r"FRESH golden cross \(EMA50 x EMA200, (\d+) bars ago\)",
     r"تقاطع ذهبي طازج (EMA50 × EMA200 قبل \1 شمعة)"),
    (r"RSI crossed above 50 \(([^)]+)\)", r"RSI اخترق 50 صعوداً (\1)"),
    (r"RSI holding above 50 \(([^)]+)\)", r"RSI يحافظ فوق 50 (\1)"),
    (r"Strong volume \(([^)]+)\)", r"حجم قوي (\1)"),
    (r"Volume above average \(([^)]+)\)", r"حجم فوق المتوسط (\1)"),
    (r"Partial trend signal: (.+)", r"إشارة اتجاه جزئية: \1"),
    # --- volatility breakout ---
    (r"Active BB squeeze \(ratio ([^)]+)\)", r"انضغاط بولينجر نشط (النسبة \1)"),
    (r"Recent BB squeeze \(current ratio ([^)]+)\)",
     r"انضغاط بولينجر حديث (النسبة الحالية \1)"),
    (r"Volume buildup before breakout \(([^)]+)\)",
     r"تراكم حجم قبل الاختراق (\1)"),
    (r"Volume spike on breakout \(([^)]+)\)", r"قفزة حجم عند الاختراق (\1)"),
    (r"BUCKSHOT: close ([\d.]+) > upper BB ([\d.]+)",
     r"اختراق صاعد (BUCKSHOT): إغلاق \1 فوق الباند العلوي \2"),
    (r"BEARISH breakout: close ([\d.]+) < lower BB ([\d.]+)",
     r"اختراق هابط: إغلاق \1 تحت الباند السفلي \2"),
    (r"Strong \+ rising ADX \(([^,)]+), was ([^)]+)\)",
     r"ADX قوي ومتزايد (\1، كان \2)"),
    (r"Rising ADX \(([^)]+)\)", r"ADX متزايد (\1)"),
    (r"ADX above 20 \(([^)]+)\)", r"ADX فوق 20 (\1)"),
    (r"RSI bullish zone \(([^)]+)\)", r"RSI في المنطقة الصاعدة (\1)"),
    (r"RSI bearish zone \(([^)]+)\)", r"RSI في المنطقة الهابطة (\1)"),
    # --- bb mean reversion ---
    (r"Lower BB broken \(percent B = ([^)]+)\)",
     r"كسر الباند السفلي (Percent B = \1)"),
    (r"Lower BB touched \(percent B = ([^)]+)\)",
     r"لمس الباند السفلي (Percent B = \1)"),
    (r"Deeply oversold \(Stoch ([^,)]+), RSI ([^)]+)\)",
     r"تشبع بيعي عميق (ستوكاستك \1، RSI \2)"),
    (r"Stochastic oversold \(([^)]+)\)", r"ستوكاستك في تشبع بيعي (\1)"),
    (r"RSI deeply oversold \(([^)]+)\)", r"RSI في تشبع بيعي عميق (\1)"),
    (r"RSI oversold \(([^)]+)\)", r"RSI في تشبع بيعي (\1)"),
    (r"Fresh Stoch cross up in oversold zone \(([^)]+)\)",
     r"تقاطع ستوكاستك صاعد طازج داخل التشبع (\1)"),
    (r"Capitulation volume \(([^)]+)\)", r"حجم استسلام (\1)"),
    (r"Volume present \(([^)]+)\)", r"حجم مؤكد (\1)"),
    (r"Partial mean-reversion: (.+)", r"إشارة ارتداد جزئية: \1"),
    # --- macd breakout ---
    (r"Price above EMA50 \(([^)]+)\)", r"السعر فوق EMA50 (\1)"),
    (r"Recent MACD cross \(<= ([^)]+)\)", r"تقاطع MACD حديث (≤ \1)"),
    (r"Breakout volume \(([^)]+)\)", r"حجم اختراق (\1)"),
    (r"Volume confirmed \(([^)]+)\)", r"حجم مؤكد (\1)"),
    # --- trend pullback ---
    (r"Very strong trend \(ADX=([^)]+)\)", r"اتجاه قوي جداً (ADX=\1)"),
    (r"Strong trend \(ADX=([^)]+)\)", r"اتجاه قوي (ADX=\1)"),
    (r"Moderate trend \(ADX=([^)]+)\)", r"اتجاه متوسط (ADX=\1)"),
    (r"Weak trend \(ADX=([^)]+)\)", r"اتجاه ضعيف (ADX=\1)"),
    (r"Pullback to EMA 21 \(([^)]+)\) \+ recovered",
     r"ارتداد إلى EMA21 (\1) ثم تعافى"),
    (r"Pullback to EMA 9 \(([^)]+)\) \+ recovered",
     r"ارتداد إلى EMA9 (\1) ثم تعافى"),
    (r"Bullish reversal candle: (.+)", r"شمعة انعكاس صاعدة: \1"),
    (r"Healthy volume \(ratio ([^)]+)\)", r"حجم صحي (\1)"),
    (r"High volume bullish candle \(([^)]+)\)",
     r"شمعة صاعدة بحجم مرتفع (\1)"),
    (r"Low volume \(([^)]+)\)", r"حجم منخفض (\1)"),
    (r"RSI healthy \(([^)]+)\)", r"RSI في نطاق صحي (\1)"),
    (r"Partial signal: (.+)", r"إشارة جزئية: \1"),
    # --- liquidity sweep ---
    (r"Reversal candle: (.+)", r"شمعة انعكاس: \1"),
    (r"High volume \(([^)]+)\)", r"حجم مرتفع (\1)"),
    (r"Bullish Liquidity Sweep: swept low ([\d.]+), recovered to ([\d.]+) \(\+([^)]+)\)",
     r"كنس سيولة صاعد: كسر القاع \1 ثم استعاد \2 (+\3)"),
    (r"Bearish Liquidity Sweep: swept high ([\d.]+), reversed to ([\d.]+) \((-[^)]+)\)",
     r"كنس سيولة هابط: كسر القمة \1 ثم عاد إلى \2 (\3)"),
    (r"Wyckoff Spring: false breakdown below ([\d.]+)",
     r"زنبرك وايكوف: كسر وهمي تحت \1"),
    (r"Bullish Order Block at ([\d.]+)-([\d.]+)",
     r"كتلة أوامر صاعدة عند \1-\2"),
    (r"Price below lower BB \(Percent B=([^)]+)\)",
     r"السعر تحت الباند السفلي (Percent B=\1)"),
    (r"Price near lower BB \(Percent B=([^)]+)\)",
     r"السعر قرب الباند السفلي (Percent B=\1)"),
    # --- fibonacci target labels ---
    (r"Limit - Fib (.+) x support confluence",
     r"أمر محدد - فيب \1 مع توافق دعم"),
    (r"Fib ([\d.]+) extension", r"امتداد فيب \1"),
    (r"Fib ([\d.]+) retracement", r"ارتداد فيب \1"),
    (r"Fib 1\.0 \(swing high\)", r"فيب 1.0 (قمة الموجة)"),
    (r"Resistance (\d+)", r"مقاومة \1"),
    (r"Support (\d+)", r"دعم \1"),
    (r" \+ S/R confluence", r" + توافق دعم/مقاومة"),
    (r" \+ support confluence", r" + توافق دعم"),
    (r" \+ resistance confluence", r" + توافق مقاومة"),
    # --- confluence adjustments ---
    (r"Ichimoku regime aligned \(([^)]+)\) x([\d.]+)",
     r"نظام إيشيموكي موافق (\1) ×\2"),
    (r"Price inside Ichimoku cloud \(chop\) x([\d.]+)",
     r"السعر داخل سحابة إيشيموكي (تذبذب) ×\1"),
    (r"Ichimoku regime opposes signal \(([^)]+) vs ([^)]+)\)",
     r"نظام إيشيموكي يعارض الإشارة (\1 مقابل \2)"),
    (r"VETO: (.+)", r"نقض إيشيموكي: \1"),
    (r"Ichimoku regime opposes \(([^)]+)\) x([\d.]+)",
     r"نظام إيشيموكي يعارض (\1) ×\2"),
    (r"Elliott wave 3 in progress \+([\d.]+) \(strongest phase\)",
     r"موجة إليوت 3 جارية +\1 (أقوى مرحلة)"),
    (r"Elliott wave 4 pullback \+([\d.]+) \(wave 5 ahead\)",
     r"تصحيح موجة إليوت 4 +\1 (الموجة 5 قادمة)"),
    (r"Elliott wave 5 mature (-?[\d.]+) \(correction risk\)",
     r"الموجة 5 ناضجة \1 (خطر تصحيح)"),
    (r"Elliott impulse complete (-?[\d.]+) \(correction expected\)",
     r"الموجة الدافعة مكتملة \1 (تصحيح متوقع)"),
    (r"ABC correction complete \+([\d.]+) \(new cycle starting\)",
     r"تصحيح ABC مكتمل +\1 (بدء دورة جديدة)"),
    (r"ABC correction in progress (-?[\d.]+) \(wait\)",
     r"تصحيح ABC جارٍ \1 (انتظر)"),
    (r"ABC bounce complete \+([\d.]+) \(down cycle resuming\)",
     r"ارتداد ABC مكتمل +\1 (استئناف الدورة الهابطة)"),
    (r"ABC bounce complete (-?[\d.]+) \(down impulse next\)",
     r"ارتداد ABC مكتمل \1 (دافع هابط قادم)"),
    (r"ABC bounce in progress (-?[\d.]+) \(wait\)",
     r"ارتداد ABC جارٍ \1 (انتظر)"),
    (r"Corrective bounce complete (-?[\d.]+) \(down impulse next\)",
     r"ارتداد تصحيحي مكتمل \1 (دافع هابط قادم)"),
    (r"Counter-cycle signal (-?[\d.]+) \(([^)]+)\)",
     r"إشارة عكس الدورة \1 (\2)"),
    (r"A\+ SETUP \+([\d.]+) \(regime \+ cycle \+ zones all aligned\)",
     r"إعداد A+ ممتاز +\1 (النظام + الدورة + المناطق كلها متوافقة)"),
    (r"Harmony ([\d.]+) below A\+ bar ([\d.]+) \(setup is 2-layer, not 3-layer\)",
     r"الانسجام \1 تحت عتبة A+ البالغة \2 (الإعداد بطبقتين لا ثلاث)"),
]

_COMPILED: List[Tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), t) for p, t in _RULES
]

# ------------------------------------------------------------------
# 3) Always-applied token substitutions (safe, unambiguous)
# ------------------------------------------------------------------
_TOKENS: List[Tuple[re.Pattern, str]] = [
    # candlestick pattern names
    (re.compile(r"\bBullish Engulfing\b"), "الاحتواء الصاعد"),
    (re.compile(r"\bBearish Engulfing\b"), "الاحتواء الهابط"),
    (re.compile(r"\bPiercing Line\b"), "خط الاختراق (بيرسينج)"),
    (re.compile(r"\bMorning Star\b"), "نجمة الصباح"),
    (re.compile(r"\bTweezer Bottom\b"), "قاع الملقط"),
    (re.compile(r"\bInverted Hammer\b"), "المطرقة المقلوبة"),
    (re.compile(r"\bHammer\b"), "المطرقة"),
    (re.compile(r"\bShooting Star\b"), "الشهاب الهابط"),
]

_DIRECTION_WORDS: Dict[str, str] = {
    "bullish": "صاعد", "bearish": "هابط", "neutral": "محايد",
}


def tr(text) -> str:
    """Translate a dynamic fragment to Arabic (graceful pass-through)."""
    if not text or not isinstance(text, str):
        return text
    # 1) exact match
    if text in _EXACT:
        return _EXACT[text]
    out = text
    # 2) ordered regex rules (sequential -> composites translate piecewise)
    for pat, tpl in _COMPILED:
        out = pat.sub(tpl, out)
    # 3) token substitutions ALWAYS at the end: candlestick pattern names
    #    (may sit inside rule output like "شمعة انعكاس: Hammer") and raw
    #    direction words captured inside rule output ("regime aligned
    #    (bullish)"). Unambiguous Latin tokens - safe everywhere.
    for pat, tpl in _TOKENS:
        out = pat.sub(tpl, out)
    for en, ar in _DIRECTION_WORDS.items():
        out = re.sub(rf"\b{en}\b", ar, out, flags=re.IGNORECASE)
    # cosmetics: numeric arrows inside captured fragments
    out = out.replace(" -> ", " → ")
    return out


def ar_direction(direction: str) -> str:
    """bullish/bearish/neutral -> Arabic (falls back to the raw value)."""
    return _DIRECTION_WORDS.get((direction or "").lower(),
                                (direction or "محايد").title())


def ar_mode(mode: str) -> str:
    """PAPER/LIVE run-mode -> Arabic with the Latin tag kept for clarity."""
    m = (mode or "").upper()
    if m == "PAPER":
        return "تجريبي (PAPER)"
    if m == "LIVE":
        return "حقيقي (LIVE)"
    return mode or "غير معروف"


def ar_entry_type(entry_type: str) -> str:
    """market/limit -> Arabic."""
    t = (entry_type or "").lower()
    if t == "limit":
        return "أمر محدد"
    if t == "market":
        return "سوق"
    return entry_type or "سوق"


_ICHIMOKU_MAPS = {
    "regime": {"bullish": "صاعد", "bearish": "هابط", "neutral": "محايد"},
    "tk_state": {"bullish": "صاعد", "bearish": "هابط", "neutral": "محايد"},
    "cloud_color": {"green": "خضراء", "red": "حمراء",
                    "bullish": "صاعدة", "bearish": "هابطة"},
    "price_vs_cloud": {"above": "فوق السحابة", "below": "تحت السحابة",
                       "inside": "داخل السحابة"},
    "price_vs_kijun": {"above": "فوق كيجون", "below": "تحت كيجون"},
}


def ar_ichimoku(field: str, value) -> str:
    """Map raw ichimoku field values to Arabic (falls back to raw)."""
    if value is None:
        return "N/A"
    v = str(value).lower()
    return _ICHIMOKU_MAPS.get(field, {}).get(v, str(value))
