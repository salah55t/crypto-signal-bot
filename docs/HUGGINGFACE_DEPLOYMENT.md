# 🚀 دليل نشر مدرب RL على Hugging Face Spaces

> يمكنك تدريب نماذج RL (DQN, PPO, A2C) مباشرة على Hugging Face Spaces مجاناً، بدون تثبيت أي شيء على جهازك.

---

## ✅ الإجابة على أسئلتك

### 1) هل يمكن تدريب النماذج على Hugging Face؟
**نعم، يمكن ذلك!** Hugging Face Spaces توفر:

| الميزة | متوفرة | ملاحظة |
|--------|--------|------|
| CPU مجاني | ✅ | 2 vCPU, 16GB RAM |
| GPU مجاني (ZeroGPU) | ✅ | NVIDIA RTX Pro (3.5 دقيقة/يوم) |
| GPU مدفوع | ✅ | $4.99/ساعة (A10G, A100) |
| مساحة تخزين | ✅ | 50GB ephemeral (مجاني) |
| Persistent storage | ✅ | $5/شهر لـ 20GB |
| Secrets | ✅ | متغيرات بيئة مشفّرة |
| Docker support | ✅ | CUDA + GPU |

### 2) هل Binance محجورة على Hugging Face؟

**Hugging Face Spaces تعمل على AWS US East (N. Virginia)** — وهي في الولايات المتحدة.

| Endpoint Binance | يعمل من Hugging Face؟ | السبب |
|------------------|----------------------|------|
| `api.binance.com` | ❌ محجور (HTTP 451) | ممنوع في الولايات المتحدة |
| `data-api.binance.vision` | ✅ **يعمل** | Endpoint للبيانات العامة فقط، بدون قيود جغرافية |
| `testnet.binance.vision` | ✅ يعمل | للتجربة فقط |

> **ملاحظة مهمة**: نحن نستخدم `data-api.binance.vision` في البوت بالفعل (في `config/settings.py`)، لذلك يمكن للمدرب على HF Spaces جلب بيانات Binance بدون مشاكل.

---

## 📦 ما تم إنشاؤه لك

أنشأت لك تطبيق **Gradio** كامل في مجلد `hf-spaces-trainer/`:

```
hf-spaces-trainer/
├── app.py              # تطبيق Gradio (واجهة تدريب)
├── requirements.txt    # اعتماديات HF Spaces
└── README.md           # ملف HF Spaces metadata
```

### المميزات:
- ✅ **واجهة ويب بسيطة** — اختر العملة، الفريم، التimesteps، اضغط تدريب
- ✅ **Progress tracking** — شريط تقدم أثناء التدريب
- ✅ **Download button** — حمّل الملف .zip بعد التدريب
- ✅ **تقرير أداء** — Return, Win Rate, Max Drawdown
- ✅ **لا يحتاج مفتاح Binance API** — يستخدم endpoint البيانات العامة
- ✅ **يعمل من أي منطقة جغرافية** (US, EU, Asia)

---

## 🚀 خطوات النشر على Hugging Face Spaces

### الخطوة 1: أنشئ حساب على Hugging Face
- اذهب إلى: <https://huggingface.co/join>
- سجّل بحساب GitHub أو بريد إلكتروني

### الخطوة 2: أنشئ Space جديد
1. اذهب إلى: <https://huggingface.co/new-space>
2. املأ البيانات:
   - **Space name**: `crypto-rl-trainer`
   - **License**: MIT
   - **SDK**: **Gradio**
   - **SDK version**: 4.44.1
   - **Space Hardware**: **CPU basic (free)** أو **ZeroGPU (free, with limits)**
   - **Visibility**: Public أو Private
3. اضغط **Create Space**

### الخطوة 3: ارفع الملفات

#### الطريقة 1: عبر واجهة الويب (الأسهل)
1. ادخل إلى Space الذي أنشأته
2. اضغط **Files** في الأعلى
3. اضغط **+ Add file** → **Upload file**
4. ارفع 3 ملفات من مجلد `hf-spaces-trainer/`:
   - `app.py`
   - `requirements.txt`
   - `README.md`

#### الطريقة 2: عبر Git (للمتقدمين)
```bash
# استنسخ Space الذي أنشأته
git clone https://huggingface.co/USERNAME/crypto-rl-trainer
cd crypto-rl-trainer

# انسخ الملفات من المشروع
cp /path/to/crypto-signal-bot/hf-spaces-trainer/* .

# Commit & Push
git add .
git commit -m "🚀 Initial Space - Crypto RL Trainer"
git push
```

### الخطوة 4: انتظر التثبيت (3-5 دقائق)

سيقوم Hugging Face تلقائياً بـ:
1. قراءة `requirements.txt`
2. تثبيت `gradio`, `torch`, `stable-baselines3`, `gymnasium`
3. تشغيل `app.py`

عند الانتهاء، ستظهر رسالة في أعلى Space:
```
Running on public URL: https://USERNAME-crypto-rl-trainer.hf.space
```

### الخطوة 5: استخدم التطبيق

1. افتح رابط الـ Space في المتصفح
2. اختر:
   - **Symbol**: `BTCUSDT`
   - **Timeframe**: `1h`
   - **Bars**: `2000`
   - **Algorithm**: `PPO` (موصى به للبدء)
   - **Timesteps**: `50000` (5-10 دقائق على CPU)
3. اضغط **🚀 Start Training**
4. انتظر حتى يكتمل التدريب
5. حمّل الملف `ppo_crypto.zip` بالضغط على رابط التحميل
6. كرر لكل من DQN و A2C

### الخطوة 6: ضع النماذج في البوت

1. انسخ الملفات التي حمّلتها إلى مجلد `data/models/` في البوت:
   ```bash
   cp ~/Downloads/dqn_crypto.zip /path/to/crypto-signal-bot/data/models/
   cp ~/Downloads/ppo_crypto.zip /path/to/crypto-signal-bot/data/models/
   cp ~/Downloads/a2c_crypto.zip /path/to/crypto-signal-bot/data/models/
   ```

2. فعّل RL في `.env`:
   ```bash
   ENABLE_RL_STRATEGY=true
   ```

3. أعد تشغيل البوت — سيجد النماذج تلقائياً ويستخدمها

---

## ⚡ حدود Hugging Face Spaces

### Free CPU Tier
- **المواصفات**: 2 vCPU, 16GB RAM, 50GB ephemeral storage
- **مدة التدريب التقريبية**:
  - 50K خطوة على PPO: ~5-10 دقائق
  - 100K خطوة على PPO: ~15-25 دقيقة
  - 100K خطوة على DQN: ~20-30 دقيقة
- **Sleep بعد 48 ساعة** من عدم النشاط
- **3.5 دقيقة GPU يومياً** (ZeroGPU)

### ZeroGPU (مجاني، محدود)
- NVIDIA RTX Pro 6000
- 3.5 دقيقة/يوم للحساب المجاني
- **مثالي للاختبار السريع**

### Paid GPU ($4.99-$15/ساعة)
- A10G, A100, L4X4
- أسرع 5-10 مرات من CPU
- للتدريب الإنتاجي

### Persistent Storage ($5/شهر)
- 20GB+ مساحة دائمة
- النماذج تبقى محفوظة بعد restart
- موصى به إذا ستعمل تدريب متكرر

---

## 🎯 نصائح للنجاح

### 1) ابدأ بـ PPO فقط (لا تحتاج الثلاثة معاً)
PPO هو الأكثر استقراراً والأكثر استخداماً. ابدأ به فقط للتجربة، ثم أضف DQN و A2C لاحقاً.

### 2) استخدم 2000-3000 شمعة (وليس 5000)
- 2000 شمعة = ~83 يوم على فريم 1h = ~3 أشهر بيانات
- كافية لتدريب جيد بدون overfitting

### 3) جرّب فريم 1h أولاً
- فريم 15m فيه ضوضاء كثيرة
- فريم 4h يحتاج بيانات أكثر
- 1h هو التزامن الذهبي للتعلم

### 4) لا تدرب على عملة غير سيولة
- تجنّب العملات بحجم تداول أقل من $50M يومياً
- أفضل خيارات: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT

### 5) أعِد التدريب كل 2-4 أسبوع
- السوق يتغير باستمرار
- نماذج قديمة قد تقل دقتها مع الوقت

---

## 🛠️ حل المشاكل الشائعة

### المشكلة: "Error loading models"
**السبب**: بعض النماذج لم تُحمّل بعد (SB3 يستغرق وقتاً).
**الحل**: انتظر دقيقة وأعد المحاولة.

### المشكلة: "Out of Memory (OOM)"
**السبب**: البيانات أو النموذج كبير جداً.
**الحل**:
- قلل `Number of bars` إلى 1000
- قلل `Training timesteps` إلى 30000
- استخدم ZeroGPU

### المشكلة: "Training is very slow"
**السبب**: على Free CPU Tier.
**الحل**:
- استخدم ZeroGPU للحصول على GPU مجاني (3.5 دقيقة/يوم)
- أو استخدم GPU مدفوع ($4.99/ساعة)

### المشكلة: "Data fetch failed"
**السبب**: مشكلة شبكة أو رمز غير صحيح.
**الحل**:
- تأكد من أن الرمز ينتهي بـ `USDT` (مثل `BTCUSDT`)
- جرّب رمزاً آخر

### المشكلة: "Model downloaded but bot doesn't load it"
**السبب**: الملف ليس في المكان الصحيح.
**الحل**:
- ضع الملف في `/path/to/crypto-signal-bot/data/models/`
- الاسم يجب أن يكون: `dqn_crypto.zip`, `ppo_crypto.zip`, `a2c_crypto.zip`

---

## 📞 روابط مفيدة

| الرابط | الوصف |
|--------|------|
| <https://huggingface.co/new-space> | إنشاء Space جديد |
| <https://huggingface.co/settings/storage> | إدارة Persistent Storage |
| <https://huggingface.co/docs/hub/spaces> | توثيق HF Spaces |
| <https://github.com/salah55t/crypto-signal-bot> | المستودع الرئيسي للبوت |
| <https://data.binance.vision> | بيانات Binance العامة |

---

## 🎉 الخلاصة

✅ **يمكنك تدريب نماذج RL على Hugging Face Spaces مجاناً**
✅ **Binance ليست محجورة تماماً** — فقط `api.binance.com` محجور، لكن `data-api.binance.vision` يعمل
✅ **التدريب على PPO 50K خطوة يستغرق 5-10 دقائق على CPU مجاني**
✅ **النماذج المدرّبة تعمل في البوت مباشرة** بعد وضعها في `data/models/`

ابدأ الآن:
1. اذهب إلى <https://huggingface.co/new-space>
2. أنشئ Space باسم `crypto-rl-trainer`
3. ارفع الملفات من `hf-spaces-trainer/`
4. ابدأ التدريب! 🚀
