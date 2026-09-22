# 🚀 دليل النشر على Render

> هذا الدليل يشرح كيفية نشر بوت **Crypto Signal Bot** على [Render](https://render.com) — منصة استضافة سحابية مجانية (مع خطة مدفوعة اختيارية).

---

## 📋 نظرة عامة على البنية المُنشرَة

يستخدم النشر **خدمة ويب واحدة (Web Service)** تعمل بـ FastAPI وتقوم بـ:

| الوظيفة | كيف تعمل |
|--------|----------|
| 🌐 لوحة الويب الحية (HTML/CSS/JS) | خدمة FastAPI ملفات ثابتة |
| 🔌 WebSocket للتحديثات الحية | نفس خادم FastAPI |
| 📊 REST APIs (`/api/...`) | نفس خادم FastAPI |
| ⏰ تشغيل البوت الدوري | APScheduler داخل العملية |
| 💾 تخزين التوصيات | نظام ملفات محلي (`data/`) |

**المميزات**:
- خدمة واحدة فقط (بسيطة الإدارة)
- لا يحتاج Postgres أو Redis
- يعمل على Render Free Tier (مجاني)

---

## 🆓 حدود Render Free Tier

| العنصر | الحد |
|--------|------|
| ساعات التشغيل الشهرية | 750 ساعة (يكفي لخدمة واحدة 24/7) |
| Sleep بعد عدم النشاط | 15 دقيقة (يتوقف الخادم) |
| الذاكرة | 512MB RAM |
| مساحة القرص | 1GB |
| النطاق الترددي | 100GB/شهر |
| وقت البناء | 500 دقيقة/شهر |

> ⚠️ **ملاحظة**: عند توقف الخادم بعد 15 دقيقة من عدم النشاط، المجدول الداخلي (APScheduler) سيتوقف أيضاً.
> للتشغيل المستمر 24/7 بدون توقف، اترقي لخطة **Starter** ($7/شهر).

---

## 📦 خطوات النشر (الأبسط — عبر GitHub)

### الخطوة 1: تأكد من أن المشروع على GitHub

المشروع يجب أن يكون على GitHub. اتبع دليل README الرئيسي لرفعه.

### الخطوة 2: اذهب إلى Render

1. سجل دخول إلى <https://render.com> (يمكنك الدخول بحساب GitHub)
2. اضغط **New +** ثم **Blueprint**
3. ابحث عن مستودع `salah55t/crypto-signal-bot` واختره

### الخطوة 3: تفعيل Blueprint

Render سيكتشف ملف `render.yaml` تلقائياً ويعرض إعدادات النشر:

- **Service Name**: `crypto-signal-bot` (افتراضي)
- **Branch**: `main`
- **Region**: اختر الأقرب لك (`oregon` للولايات المتحدة، `frankfurt` لأوروبا، `singapore` لآسيا)
- **Instance Type**: `Free` (مجاني) أو `Starter` ($7/شهر)

اضغط **Apply** للبدء.

### الخطوة 4: انتظر البناء

سيقوم Render بـ:
1. نسخ المستودع
2. تثبيت `requirements.txt` (يأخذ 3-5 دقائق)
3. تشغيل `uvicorn src.web.app:app --host 0.0.0.0 --port $PORT`

عند الانتهاء، ستحصل على رابط مثل:
```
https://crypto-signal-bot-xxxx.onrender.com
```

### الخطوة 5: تحقق من عمل الخدمة

افتح في المتصفح:
- <https://crypto-signal-bot-xxxx.onrender.com/> → لوحة الويب
- <https://crypto-signal-bot-xxxx.onrender.com/api/health> → يجب أن يعيد JSON بحالة "ok"

---

## ⚙️ إعداد متغيرات البيئة (Environment Variables)

الملف `render.yaml` يحتوي على كل الإعدادات الافتراضية، لكن تحتاج لضبط القيم الحساسة يدوياً:

1. في لوحة Render، اذهب إلى خدمتك → **Environment**
2. أضف/عدّل المتغيرات التالية:

### 🔐 متغيرات حساسة (sync: false في render.yaml — تُدخل يدوياً)

| المتغير | القيمة | الوصف |
|--------|-------|------|
| `BINANCE_API_KEY` | مفتاحك من Binance | **اختياري** — يعمل بدونها بالبيانات العامة |
| `BINANCE_API_SECRET` | سرّك من Binance | **اختياري** |
| `TELEGRAM_BOT_TOKEN` | توكن بوت Telegram | **اختياري** للإشعارات |
| `TELEGRAM_CHAT_ID` | ID محادثتك | **اختياري** |

3. بعد التعديل، اضغط **Save Changes** — سيُعاد النشر تلقائياً.

---

## ⏯️ تفعيل التشغيل الفوري عند الإقلاع (اختياري)

افتراضياً، البوت ينتظر الجدولة (كل ساعة) لبدء التحليل. لتفعيل تحليل فوري عند الإقلاع:

في **Environment** على Render، أضف:
```
RUN_ON_STARTUP = true
```

> ⚠️ ملاحظة: هذا سيؤدي إلى استهلاك دورة بيانات Binance API عند كل إقلاع.

---

## 📅 ضبط جدولة التحليل

الجدولة الافتراضية: كل ساعة (`0 * * * *`). للتغيير:

عدّل `SCHEDULE_CRON` في Environment على Render:

| الجدولة المطلوبة | القيمة |
|------------------|--------|
| كل ساعة | `0 * * * *` (افتراضي) |
| كل 30 دقيقة | `*/30 * * * *` |
| كل 4 ساعات | `0 */4 * * *` |
| كل يوم الساعة 00:00 UTC | `0 0 * * *` |
| كل يوم الساعة 9 صباحاً بتوقيت UTC | `0 9 * * *` |

> استخدم <https://crontab.guru> لبناء cron expression.

---

## 📊 مراقبة الخدمة

### عرض السجلات
في لوحة Render → خدمتك → **Logs** (يمين الشاشة).

ستظهر سجلات مثل:
```
[INFO] Scheduler started - cron: '0 * * * *'
[INFO] Starting market analysis for 41 symbols
[INFO] Analysis complete - 5/41 signals passed filter
[INFO] Sent 5 recommendations to Telegram
```

### مراقبة الصحة
- Render يفحص `/api/health` كل 60 ثانية
- إذا فشل الفحص 3 مرات متتالية، يتم إعادة تشغيل الخدمة

---

## 🔄 التحديثات التلقائية (Auto-Deploy)

افتراضياً، Render يعيد النشر تلقائياً عند كل `git push` إلى فرع `main`.

لإيقاف التحديث التلقائي:
1. خدمتك → **Settings**
2. ابحث عن **Auto-Deploy**
3. عطّلها

---

## 💡 نصائح لتحسين الأداء على Render

### 1) ترقية إلى Starter ($7/شهر) — موصى به للإنتاج
- لا توقّف بعد 15 دقيقة من عدم النشاط (always-on)
- ذاكرة 512MB+ أفضل
- أولوية في البناء
- مثالي لتشغيل البوت بشكل مستمر 24/7

### 2) تخفيض عدد العملات إذا واجهت أخطاء Rate Limit
في `config/coins.yaml` قلل القائمة إلى 20 عملة أو أقل.

### 3) تفعيل Persistent Disk (مدفوع)
للحفاظ على البيانات بين عمليات النشر:
```yaml
services:
  - type: web
    name: crypto-signal-bot
    disk:
      name: bot-data
      mountPath: /app/data
      sizeGB: 1  # $0.25/month
```

### 4) استخدام Binance Testnet للتجربة
في Environment:
```
BINANCE_TESTNET = true
BINANCE_BASE_URL = https://testnet.binance.vision
```
ثم أنشئ مفاتيح API على <https://testnet.binance.vision>

---

## 🛠️ حل المشكلات الشائعة

### المشكلة: `Build failed: Could not install requirements`
**السبب**: انتهت مدة بناء Render المجانية أو مشكلة شبكة.
**الحل**:
- أعد البناء: خدمتك → **Manual Deploy** → **Clear build cache & deploy**
- أو حدّث `requirements.txt` بأقل اعتماديات

### المشكلة: `Worker exceeded memory limit of 512MB`
**السبب**: تحليل 41 عملة بالتوازي مع ML يستهلك ذاكرة كبيرة.
**الحل**:
- قلل عدد العملات في `config/coins.yaml`
- عطّل ML بإزالة `MLStrategy()` من `src/analysis/scorer.py`
- أو رقّ إلى خطة Starter

### المشكلة: `Binance API rate limit exceeded`
**السبب**: عدد طلبات كبير خلال وقت قصير.
**الحل**:
- قلل `MAX_WORKERS` في `src/analysis/analyzer.py` من 5 إلى 2
- قلل عدد العملات

### المشكلة: `Service went to sleep` (Free Tier)
**السبب**: لا يوجد نشاط HTTP لمدة 15 دقيقة.
**الحل**:
- استخدم خدمة خارجية مثل <https://uptimerobot.com> لطلب `/api/health` كل 10 دقائق
- أو رقّ إلى Starter plan

### المشكلة: `Dashboard empty — لا تظهر توصيات`
**السبب**: التحليل لم يجد إشارات قوية (confidence < 70%).
**الحل**:
- اضبط `MIN_CONFIDENCE=50` و `MIN_EXPECTED_RISE=1.0` في Environment
- أو انتظر الساعة القادمة لتحليل جديد

---

## 📞 بديل: النشر عبر Dockerfile بدلاً من Native Python

إذا واجهت مشاكل مع Native Python runtime على Render:

1. في لوحة Render → **New +** → **Web Service**
2. اختر المستودع
3. في **Runtime**، اختر **Docker** بدلاً من Python
4. Render سيكتشف `Dockerfile` تلقائياً
5. اضغط **Create Web Service**

---

## 🎯 ملخص سريع

```bash
# 1. شامل المستودع على GitHub (موجود)
git clone https://github.com/salah55t/crypto-signal-bot.git

# 2. ادخل Render واختر Blueprint
# 3. اختر المستودع
# 4. اترك الإعدادات الافتراضية + اضبط متغيرات البيئة الحساسة
# 5. اضغط Apply
# 6. انتظر 5 دقائق
# 7. افتح الرابط المُعطى من Render
```

---

## 🆘 طلب المساعدة

إذا واجهت مشاكل:
1. راجع **Logs** في لوحة Render
2. تحقق من `/api/health` endpoint
3. ابحث عن الخطأ في [Render Docs](https://render.com/docs)
4. افتح Issue في المستودع: <https://github.com/salah55t/crypto-signal-bot/issues>
