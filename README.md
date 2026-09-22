# 🤖 Crypto Signal Bot — بوت توصيات تداول العملات المشفرة على Binance Spot

> بوت متطور لاقتراح توصيات التداول على منصة **Binance Spot** يقوم بتحليل العملات المشفرة باستخدام مجموعة متقدمة من الاستراتيجيات (تحليل فني، حجم، نماذج شموع، زخم، سيولة، تعلم آلي) ويصدر توصيات مع نسبة صعود متوقعة ودرجة ثقة ومستويات وقف الخسارة وجني الأرباح.

## ✨ المميزات الرئيسية

- ✅ **تحليل متعدد الاستراتيجيات** — 6 استراتيجيات تعمل بالتوازي
- ✅ **تحليل متعدد الإطارات الزمنية** — 15m + 1h + 4h
- ✅ **درجة ثقة موحدة** من 0-100 لكل توصية
- ✅ **حساب تلقائي لـ** Stop Loss / Take Profit / Risk-Reward Ratio
- ✅ **تعلم آلي** باستخدام Random Forest للتنبؤ بالاتجاه
- ✅ **تحليل دفتر الأوامر** (Order Book) لكشف اختلال السيولة
- ✅ **Backtesting** لاختبار الاستراتيجيات على بيانات تاريخية
- ✅ **Paper Trading** لتنفيذ وهمي للتوصيات بدون مخاطرة
- ✅ **إشعارات Telegram** فورية لكل توصية
- ✅ **لوحة ويب حية** (Node.js + WebSocket) لعرض التوصيات مباشرة
- ✅ **مجدول دوري** لتشغيل التحليل تلقائياً كل ساعة (قابل للتخصيص)
- ✅ **يعمل بدون مفاتيح API** باستخدام البيانات العامة لـ Binance
- ✅ **جاهز للنشر على Render** عبر FastAPI + `render.yaml` + Dockerfile

---

## 🚀 النشر السريع على Render

تم تجهيز المشروع للنشر على Render عبر ملف `render.yaml`. كل ما عليك:

1. اذهب إلى <https://render.com> → **New** → **Blueprint**
2. اختر مستودع `salah55t/crypto-signal-bot`
3. اضغط **Apply** — سيتم البناء والنشر تلقائياً
4. اضبط متغيرات البيئة الحساسة (Binance API, Telegram) من لوحة Render

📖 **دليل النشر الكامل خطوة بخطوة**: راجع [`docs/RENDER_DEPLOYMENT.md`](docs/RENDER_DEPLOYMENT.md)

---

## 📂 هيكل المشروع

```
crypto-signal-bot/
├── config/
│   ├── settings.py          # إعدادات تُحمّل من .env
│   └── coins.yaml           # قائمة العملات المخصصة للمراقبة
├── src/
│   ├── core/
│   │   ├── binance_client.py    # عميل Binance REST API
│   │   ├── data_fetcher.py      # جلب OHLCV و order book
│   │   └── scheduler.py         # جدولة التحليل الدوري
│   ├── indicators/
│   │   ├── technical.py     # RSI, MACD, BB, EMA, ADX, Stochastic...
│   │   ├── volume.py        # OBV, VWAP, Volume Profile, CVD
│   │   ├── patterns.py      # كاشف نماذج الشموع اليابانية
│   │   └── liquidity.py     # تحليل دفتر الأوامر + Support/Resistance
│   ├── strategies/
│   │   ├── base.py                       # فئة أساسية لكل الاستراتيجيات
│   │   ├── technical_strategy.py         # استراتيجية التحليل الفني
│   │   ├── volume_strategy.py            # استراتيجية تحليل الحجم
│   │   ├── pattern_strategy.py           # استراتيجية نماذج الشموع
│   │   ├── momentum_strategy.py          # استراتيجية الزخم والاتجاه
│   │   ├── liquidity_strategy.py         # استراتيجية تحليل السيولة
│   │   └── ml_strategy.py                # استراتيجية تعلم آلي (Random Forest)
│   ├── analysis/
│   │   ├── analyzer.py      # المنسق الرئيسي للتحليل
│   │   └── scorer.py        # دمج كل الإشارات في درجة موحدة
│   ├── risk/
│   │   └── manager.py      # إدارة المخاطر وحجم المركز
│   ├── notifications/
│   │   ├── telegram_bot.py  # إشعارات Telegram
│   │   └── file_logger.py   # حفظ التوصيات في CSV/JSON
│   ├── backtesting/
│   │   └── backtester.py   # محرك الفحص الخلفي
│   └── utils/
│       ├── logger.py       # نظام التسجيل
│       └── helpers.py       # دوال مساعدة
├── web/dashboard/
│   ├── src/server.js        # خادم Node.js (Express + WebSocket)
│   └── public/              # ملفات الواجهة (HTML/CSS/JS)
├── scripts/
│   ├── run_bot.py           # تشغيل البوت الرئيسي
│   ├── run_backtest.py      # تشغيل الفحص الخلفي
│   ├── train_ml_model.py    # تدريب نموذج التعلم الآلي
│   ├── run_dashboard.py    # تشغيل لوحة الويب
│   └── test_analysis.py     # اختبار تحليل لعملة واحدة
├── data/                   # ملفات البيانات (تُنشأ تلقائياً)
│   ├── logs/              # السجلات
│   ├── historical/        # البيانات التاريخية
│   ├── models/            # نماذج ML
│   ├── recommendations.json   # أحدث التوصيات
│   ├── open_positions.json   # الصفقات الورقية المفتوحة
│   └── daily_stats.json       # إحصائيات يومية
├── .env.example            # نموذج ملف الإعدادات
├── requirements.txt        # متطلبات Python
└── README.md
```

---

## 🚀 التثبيت والبدء السريع

### المتطلبات الأساسية

- **Python 3.10+** 
- **Node.js 18+** (للوحة الويب فقط)
- **حساب Binance** (اختياري - للبيانات الحية أو تنفيذ الصفقات)

### 1) استنساخ المستودع

```bash
git clone https://github.com/YOUR_USERNAME/crypto-signal-bot.git
cd crypto-signal-bot
```

### 2) تثبيت الاعتماديات Python

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# أو: venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3) إعداد ملف الإعدادات

```bash
cp .env.example .env
# عدّل ملف .env وقم بتعبئة القيم اللازمة
```

### 4) تخصيص قائمة العملات (اختياري)

عدّل `config/coins.yaml` لإضافة أو إزالة العملات التي تريد مراقبتها.

### 5) تشغيل البوت (تشغيل واحد للتجربة)

```bash
python scripts/run_bot.py --once
```

### 6) تشغيل البوت بشكل مستمر (مع جدولة تلقائية)

```bash
python scripts/run_bot.py
```

سيقوم البوت بالتشغيل الأول فوراً، ثم يتكرر حسب الجدولة المحددة في `.env`
(الافتراضي: كل ساعة `0 * * * *`).

### 7) (اختياري) تدريب نموذج التعلم الآلي

```bash
python scripts/train_ml_model.py --symbol BTCUSDT --interval 1h --limit 1000
```

### 8) (اختياري) تشغيل الفحص الخلفي

```bash
python scripts/run_backtest.py --symbol BTCUSDT --interval 1h --limit 1000
```

### 9) (اختياري) تشغيل لوحة الويب الحية

```bash
python scripts/run_dashboard.py
# أو يدوياً:
cd web/dashboard
npm install
npm start
```

افتح المتصفح على: <http://localhost:8080>

---

## 🔑 إعداد مفاتيح Binance API

البوت يعمل بشكل افتراضي **بدون مفاتيح API** (يستخدم بيانات السوق العامة).
لتفعيل تنفيذ الصفقات الحقيقية أو جلب بيانات حسابك:

1. سجّل الدخول إلى <https://www.binance.com>
2. اذهب إلى **API Management**: <https://www.binance.com/en/my/settings/api-management>
3. اضغط **Create API** واختر **System Generated**
4. سمّ المفتاح باسم واضح (مثل "Crypto Signal Bot")
5. **مهم جداً**: 
   - فعّل صلاحية **Read** فقط (إذا كنت تريد البوت أن ينفذ صفقات حقيقية، فعّل **Spot Trading**)
   - **عطّل** صلاحية **Withdrawals** دائماً
6. انسخ **API Key** و **Secret Key** والصقها في ملف `.env`:
   ```
   BINANCE_API_KEY=your_api_key_here
   BINANCE_API_SECRET=your_secret_here
   ```
7. أضف عنوان IP الخاص بك إلى قائمة IP المسموح بها في إعدادات Binance API

> ⚠️ **تنبيه أمني**: لا تشارك أبداً مفاتيح API مع أي شخص، ولا ترفع ملف `.env` إلى GitHub
> (تم إضافته إلى `.gitignore`).

---

## 📨 إعداد إشعارات Telegram (اختياري)

1. افتح Telegram وابحث عن `@BotFather`
2. أرسل `/newbot` واتبع التعليمات لإنشاء بوت جديد
3. انسخ **Bot Token** والصقه في `.env`:
   ```
   TELEGRAM_BOT_TOKEN=1234567890:ABC...your_token...
   ```
4. للحصول على **Chat ID** الخاص بك:
   - ابحث عن `@userinfobot` في Telegram وأرسل له أي رسالة
   - سيقوم بإرسال لك `Id: 123456789`
   - الصق الـ ID في `.env`:
     ```
     TELEGRAM_CHAT_ID=123456789
     ```
5. فعّل الإشعارات في `.env`:
   ```
   TELEGRAM_ENABLED=true
   ```

---

## 📊 شرح الاستراتيجيات

### 1️⃣ التحليل الفني (`TechnicalStrategy`)
يجمع بين أهم المؤشرات الفنية:
- **RSI** (14): كشف مناطق الشراء/البيع المفرط
- **MACD** (12, 26, 9): كشف تقاطعات الصعود/الهبوط
- **Bollinger Bands** (20, 2): كشف ذروات السعر
- **EMA** (9, 21, 50, 200): تحديد الاتجاه العام
- **ADX** (14): قوة الاتجاه
- **Stochastic** (14, 3): كشف التقاطعات في مناطق الإفراط
- **ATR** (14): حساب التقلبات لتحديد SL/TP

### 2️⃣ تحليل الحجم (`VolumeStrategy`)
- **Volume Spike**: كشف الارتفاعات المفاجئة في الحجم
- **Cumulative Volume Delta (CVD)**: تقدير صافي ضغط البيع/الشراء
- **Volume Profile**: توزيع الحجم على مستويات السعر
- **POC** (Point of Control): مستوى السعر الأعلى حجماً
- **Volume Trend**: ميل الحجم صعوداً/هبوطاً

### 3️⃣ نماذج الشموع اليابانية (`PatternStrategy`)
يكتشف 11 نمط شموع:
- **Bullish**: Hammer, Bullish Engulfing, Morning Star, Piercing Line, Three White Soldiers, Bullish Harami
- **Bearish**: Shooting Star, Bearish Engulfing, Evening Star, Dark Cloud Cover, Three Black Crows, Bearish Harami
- **Neutral**: Doji

### 4️⃣ الزخم والاتجاه (`MomentumStrategy`)
- **ROC** (12): معدل التغير
- **CCI** (20): مؤشر قناة السلع
- **Williams %R** (14)
- **Multi-Timeframe Analysis**: تأكيد الإشارة عبر 15m + 1h + 4h
- **Divergence Detection**: كشف التباعد بين السعر و RSI

### 5️⃣ تحليل السيولة (`LiquidityStrategy`)
- **Order Book Imbalance**: اختلال دفتر الأوامر
- **Buy/Sell Walls**: كشف جدران الطلب الكبيرة
- **Support/Resistance**: كشف المستويات من القمم/القيعان السابقة
- **Spread Analysis**: تحليل الفرق بين Bid و Ask

### 6️⃣ تعلم آلي (`MLStrategy`)
- **Random Forest Classifier** للتنبؤ بالاتجاه (Up/Down/Neutral)
- ميزات مستخرجة من 22 مؤشر فني
- يتم تدريبه على بيانات تاريخية باستخدام `scripts/train_ml_model.py`
- يعطي احتمالية لكل اتجاه

---

## ⚙️ الإعدادات القابلة للتخصيص

كل الإعدادات في ملف `.env`:

| المتغير | الوصف | الافتراضي |
|--------|-------|----------|
| `RUN_MODE` | `paper` (تداول وهمي) أو `live` (حقيقي) | `paper` |
| `MIN_CONFIDENCE` | أدنى درجة ثقة لإصدار التوصية | `70` |
| `MIN_EXPECTED_RISE` | أدنى نسبة صعود متوقعة | `3.0` |
| `MAX_RECOMMENDATIONS` | أقصى عدد توصيات لكل دورة | `10` |
| `RISK_PER_TRADE` | نسبة المخاطرة من رأس المال لكل صفقة | `1.0` |
| `MAX_OPEN_POSITIONS` | أقصى عدد صفقات مفتوحة | `5` |
| `DAILY_MAX_LOSS` | أقصى خسارة يومية (% من رأس المال) | `5.0` |
| `MIN_RR_RATIO` | أدنى نسبة R/R المطلوبة | `2.0` |
| `TIMEFRAMES` | الإطارات الزمنية للتحليل | `15m,1h,4h` |
| `SCHEDULE_CRON` | جدولة التشغيل (cron) | `0 * * * *` |

---

## 📈 مثال على التوصية المُصدَرة

```
🟢 BTCUSDT — BULLISH
━━━━━━━━━━━━━━━
💰 السعر الحالي:     $42,150.00
📈 الصعود المتوقع:  +5.32%
🎯 الثقة:           82.5%
🛑 وقف الخسارة:    $40,800.00 (-3.20%)
✅ جني الأرباح:    $44,400.00 (+5.32%)
⚖️ نسبة R/R:       1.66:1
━━━━━━━━━━━━━━━
أبرز الإشارات:
• RSI oversold (28.4)
• MACD bullish crossover
• EMA 9 > 21 > 50 (strong uptrend)
• Volume spike 2.1x with bullish candle
• Multi-timeframe trend ALL bullish (3/3)
• Order book imbalance +0.32 (heavy buy walls)
```

---

## 🧪 الفحص الخلفي (Backtesting)

```bash
python scripts/run_backtest.py --symbol ETHUSDT --interval 4h --limit 500
```

سيظهر لك تقرير يشمل:
- Total Return (%)
- Win Rate (%)
- Profit Factor
- Max Drawdown (%)
- عدد الصفقات الرابحة/الخاسرة

يتم حفظ التقرير الكامل في `data/backtest_report.json`.

---

## 🌐 لوحة الويب الحية

عند تشغيل `python scripts/run_dashboard.py`:
- **REST API** على المنفذ 8080:
  - `GET /api/health` — فحص الصحة
  - `GET /api/recommendations` — أحدث التوصيات
  - `GET /api/positions` — الصفقات الورقية المفتوحة
  - `GET /api/stats` — إحصائيات الأداء
- **WebSocket** على `/ws`:
  - تبث التوصيات الجديدة فور ظهورها (يتراقب ملف `data/recommendations.json`)

الواجهة تتحدث تلقائياً كلما صدرت توصية جديدة — لا حاجة لإعادة تحميل الصفحة.

---

## ⚠️ تنبيه قانوني ومسؤولية

هذا البوت **لأغراض تعليمية وبحثية فقط** ولا يُعد:
- نصيحة استثمارية أو مالية
- ضماناً للأرباح أو توقعاً أكيداً لحركة الأسعار
- بديلاً عن بحوثك الخاصة وقراراتك المستقلة

أسواق العملات المشفرة شديدة التقلب وتنطوي على مخاطر كبيرة. يمكنك أن تخسر كامل رأس مالك. تداول فقط بما يمكنك تحمل خسارته، واستخدم دائماً وقف الخسارة.

---

## 🛠️ استكشاف الأخطاء

| المشكلة | الحل |
|--------|------|
| `Cannot reach Binance API` | تحقق من اتصال الإنترنت، قد تحتاج VPN إن كانت Binance محجوبة في بلدك |
| `Insufficient data` | زد `CANDLE_LIMIT` في `.env` (الافتراضي 200) |
| `ML model not trained` | شغّل `python scripts/train_ml_model.py` |
| `Telegram: chat not found` | أرسل رسالة للبوت أولاً في Telegram، وتأكد من `TELEGRAM_CHAT_ID` |
| `Rate limit exceeded` | قلل `MAX_WORKERS` أو `TIMEFRAMES` أو عدّل قائمة العملات |
| Dashboard فارغة | تأكد من تشغيل `run_bot.py` لإنشاء `data/recommendations.json` |

---

## 📜 الترخيص

هذا المشروع مرخص تحت **MIT License** — يمكنك استخدامه وتعديله بحرية.

---

## 🤝 المساهمة

المساهمات مرحب بها! افتح Issue أو Pull Request للمساعدة في تحسين البوت.

---

## 🙏 شكر خاص

- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [pandas-ta](https://github.com/twopirllc/pandas-ta) — مكتبة المؤشرات الفنية
- [scikit-learn](https://scikit-learn.org/) — مكتبة التعلم الآلي
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
