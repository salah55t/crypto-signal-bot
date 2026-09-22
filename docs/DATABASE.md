# 📊 دليل قاعدة بيانات الإحصائيات والمراقبة

> النظام الآن يستخدم SQLite محلياً لتسجيل كل نشاطات البوت — التحليلات، التوصيات، الصفقات، أداء الاستراتيجيات.

---

## 🗄️ المخطط (Schema)

### الجداول الستة:

| الجدول | الوصف |
|--------|------|
| `runs` | كل دورة تحليل (timestamp, duration, signals count) |
| `recommendations` | كل توصية صدرت (مع confidence, SL/TP, R/R) |
| `strategy_signals` | إشارة لكل استراتيجية لكل توصية (للتحليل التفصيلي) |
| `positions` | كل الصفقات (مفتوحة + مغلقة) مع P&L |
| `daily_stats` | إحصائيات يومية (trades, wins, losses, pnl) |
| `bottom_candidates` | سجل عملات القاع المكتشفة |

### المخطط التفصيلي:

```sql
CREATE TABLE runs (
    id, timestamp, duration_seconds, symbols_analyzed,
    signals_passed, recommendations_count, bottom_candidates_count,
    mode, error
);

CREATE TABLE recommendations (
    id, run_id, timestamp, symbol, direction,
    confidence, weighted_score, current_price,
    expected_rise_pct, stop_loss, take_profit,
    risk_reward_ratio, atr_pct, timeframe,
    boosted_from_bottom, paper, opened_position
);

CREATE TABLE strategy_signals (
    id, recommendation_id, strategy_name,
    direction, score, confidence, reasons
);

CREATE TABLE positions (
    id, symbol, direction, entry_price, exit_price,
    stop_loss, take_profit, size, notional_usd,
    entry_time, exit_time, pnl, pnl_pct, confidence,
    paper, buy_order_id, oco_order_id,
    close_reason, risk_updates_count, risk_updates
);

CREATE TABLE daily_stats (
    date, trades_opened, wins, losses, pnl,
    starting_capital, ending_capital
);

CREATE TABLE bottom_candidates (
    id, timestamp, symbol, score, current_price,
    recent_low, distance_from_low_pct, rsi, atr_pct,
    signals, patterns_detected
);
```

---

## 🔌 الـ Endpoints الجديدة

كل endpoints موجودة تحت `/api/stats/`:

### 1) `GET /api/stats/summary`
إحصائيات مجمعة:
- إجمالي دورات التحليل
- إجمالي التوصيات
- عدد الصفقات المغلقة
- معدل الفوز (Win Rate)
- إجمالي P&L
- متوسط P&L%
- آخر دورة تحليل

### 2) `GET /api/stats/strategies?days=30`
أداء كل استراتيجية في آخر N يوم:
- عدد الإشارات الكلي
- إشارات صاعدة (bullish)
- إشارات هابطة (bearish)
- متوسط النتيجة (avg_score)
- متوسط الثقة (avg_confidence)

### 3) `GET /api/stats/runs?limit=50`
تاريخ آخر 50 دورة تحليل:
- timestamp
- duration_seconds
- symbols_analyzed
- signals_passed
- recommendations_count

### 4) `GET /api/stats/recommendations?limit=100&symbol=BTC`
تاريخ التوصيات مع فلترة برمز العملة:
- كل تفاصيل التوصية (entry, SL, TP, confidence, etc.)

### 5) `GET /api/stats/positions?closed_only=true&limit=50`
تاريخ الصفقات (مفتوحة/مغلقة):
- entry/exit prices
- P&L
- close_reason (Stop Loss Hit / Take Profit Hit)

### 6) `GET /api/stats/daily?limit=30`
إحصائيات يومية لآخر 30 يوم:
- trades_opened
- wins / losses
- pnl (يومي)
- starting/ending capital

### 7) `GET /api/stats/bottoms?limit=50&min_score=50`
سجل عملات القاع (bounce candidates):
- score, recent_low, distance_from_low_pct
- rsi, atr_pct
- signals, patterns_detected

---

## 🖥️ تبويب "📈 الإحصائيات" في لوحة الويب

عند فتح تبويب "📈 الإحصائيات"، سترى:

### 6 بطاقات إحصائية
| البطاقة | الوصف |
|---------|------|
| إجمالي دورات التحليل | كم مرة اشتغل البوت |
| إجمالي التوصيات | كم توصية تم إصدارها |
| صفقات مغلقة | عدد الصفقات المُغلقة |
| معدل الفوز | % (win_rate) — أخضر ≥50% / أحمر <50% |
| إجمالي P&L | $ — أخضر موجب / أحمر سالب |
| متوسط P&L% | % — متوسط الربح/الخسارة لكل صفقة |

### جدول "🏆 أداء الاستراتيجيات"
لكل استراتيجية (من أصل 14):
- اسم الاستراتيجية
- إشارات صاعدة 🟢 / إشارات هابطة 🔴
- إجمالي الإشارات
- متوسط النتيجة
- نسبة الإشارات الصاعدة
- متوسط الثقة

### جدول "📅 آخر 10 دورات تحليل"
لكل دورة:
- التاريخ والوقت
- الوضع (paper/live)
- عدد الإشارات المقبولة
- عدد التوصيات
- عدد العملات المحللة
- زمن التحليل (seconds)
- عدد عملات القاع المكتشفة

---

## 💾 التخزين على Render

### Free Tier (مجاني)
- ❌ قاعدة البيانات **تُمحى** عند إعادة التشغيل أو النوم
- الحل: إذا كنت تريد الحفاظ على الإحصائيات، ارفع لـ Starter

### Starter Plan ($7/شهر) — موصى به للإنتاج
- ✅ Persistent Disk 1GB (مجاني مع Starter)
- قاعدة البيانات تُحفظ عبر إعادة التشغيل
- لا يوجد sleep بعد 15 دقيقة

### Persistent Disk في render.yaml
```yaml
services:
  - type: web
    name: crypto-signal-bot
    plan: starter  # required for disks
    disk:
      name: bot-data
      mountPath: /app/data
      sizeGB: 1
```

### موقع ملف قاعدة البيانات
- المسار الافتراضي: `/app/data/bot_stats.db`
- عبر متغير بيئة: `DB_PATH=/path/to/db.sqlite`

---

## 🔬 استعلامات SQL مفيدة

يمكنك تشغيل هذه الاستعلامات مباشرة على SQLite عبر Render Shell:

### 1) إحصائيات سريعة:
```sql
SELECT
  (SELECT COUNT(*) FROM runs) as total_runs,
  (SELECT COUNT(*) FROM recommendations) as total_recs,
  (SELECT COUNT(*) FROM positions WHERE exit_time IS NOT NULL) as closed_positions,
  (SELECT SUM(pnl) FROM positions WHERE exit_time IS NOT NULL) as total_pnl;
```

### 2) أفضل 10 عملات بأعلى P&L:
```sql
SELECT symbol, COUNT(*) as trades, SUM(pnl) as total_pnl, AVG(pnl_pct) as avg_pct
FROM positions WHERE exit_time IS NOT NULL
GROUP BY symbol
ORDER BY total_pnl DESC
LIMIT 10;
```

### 3) أداء كل استراتيجية:
```sql
SELECT strategy_name,
  COUNT(*) as signals,
  SUM(CASE WHEN direction='bullish' THEN 1 ELSE 0 END) as bullish,
  AVG(score) as avg_score
FROM strategy_signals
GROUP BY strategy_name
ORDER BY signals DESC;
```

### 4) معدل الفوز لكل يوم:
```sql
SELECT date,
  wins, losses,
  ROUND(wins * 100.0 / (wins + losses), 1) as win_rate_pct,
  ROUND(pnl, 2) as daily_pnl
FROM daily_stats
WHERE wins + losses > 0
ORDER BY date DESC;
```

---

## 🛠️ حل المشاكل الشائعة

### المشكلة: "Database disk image is malformed"
- عادة بعد إعادة تشغيل غير متوقعة
- الحل: احذف الملف `data/bot_stats.db` وسيُعاد إنشاؤه تلقائياً

### المشكلة: "database is locked"
- يحدث عند التشغيل المتوازي بكثرة
- الحل: تأكد أن `PRAGMA journal_mode=WAL` (مفعّل افتراضياً)

### المشكلة: الإحصائيات فارغة على Render
- على Free Tier، البيانات تُمحى عند النوم
- الحل: استخدم Starter plan + Persistent Disk

---

## 📦 النسخ الاحتياطي

### لتحميل قاعدة البيانات من Render:
```bash
# على Render → خدمتك → Settings → Serial Command
sqlite3 /app/data/bot_stats.db ".dump" > backup.sql
```

### للاستعادة:
```bash
sqlite3 /app/data/bot_stats.db < backup.sql
```

---

## 🔗 روابط مفيدة

- [SQLite Documentation](https://sqlite.org/docs.html)
- [Render Disks](https://render.com/docs/disks)
- [SQLAlchemy Cheatsheet](https://www.sqlite.org/quickstart.html)

---

قاعدة البيانات جاهزة للاستخدام! بعد النشر على Render (مع Starter plan + Persistent Disk)، ستبدأ كل تحليل بجمع الإحصائيات تلقائياً.
