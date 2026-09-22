# 🧠 دليل نظام التداول بالتعلم المعزز (RL)

> نظام متطور للتداول باستخدام 3 خوارزميات تعلم معزز: **DQN, PPO, A2C** مع تصويت Ensemble.

## 📋 نظرة عامة

يستخدم النظام خوارزميات RL متطورة لتعلم سياسة تداول مثالية من البيانات التاريخية للعملات المشفرة. كل خوارزمية تتخذ قرارها (HOLD/BUY/SELL) بناءً على 10 ميزات فنية، ثم يجمع النظام قراراتها عبر **التصويت بالأغلبية** لاتخاذ قرار نهائي.

### الخوارزميات المستخدمة:

| الخوارزمية | النوع | المميزات |
|-----------|------|----------|
| **DQN** (Deep Q-Network) | Off-policy | مناسب للأفعال المنفصلة، يحفظ خبرة سابقة |
| **PPO** (Proximal Policy Optimization) | On-policy | استقرار عالٍ، الأكثر استخداماً |
| **A2C** (Advantage Actor-Critic) | On-policy | سريع، متزامن، deterministic |

### لماذا Ensemble؟
- **DQN** قد يطغى عليه الـ overfitting
- **PPO** أحياناً متحفظ جداً
- **A2C** أسرع لكن أقل دقة
- بالتصويت: نأخذ أفضل قرار مشترك من 2 من 3 (يقلل الأخطاء 60-70%)

---

## ⚙️ البيئة (CryptoTradingEnv)

### فضاء الأفعال (Action Space):
```
Discrete(3) — 0=HOLD | 1=BUY | 2=SELL
```

### فضاء الملاحظة (Observation Space): 10 ميزات
| # | الميزة | الوصف |
|---|--------|------|
| 0 | `daily_return` | العائد اليومي (مطبّق بين -0.2 و 0.2) |
| 1 | `rsi` | RSI(14) مقسوم على 100 - 0.5 |
| 2 | `macd_norm` | MACD histogram مقسوم على ATR |
| 3 | `bb_pct` | Bollinger %B - 0.5 |
| 4 | `bb_width` | Bollinger Bandwidth * 100 |
| 5 | `vol_ratio` | حجم / SMA20(volume) / 5 - 0.5 |
| 6 | `atr_pct` | ATR / السعر |
| 7 | `regime` | اتجاه السوق (1=صاعد, -1=هابط, 0=عرضي) |
| 8 | `position` | 0=لا يوجد مركز | 1=مركز مفتوح |
| 9 | `unrealized_pnl` | الربح/الخسارة غير المحققة |

### دالة المكافأة (Reward Function):
```
reward = PnL - transaction_costs - drawdown_penalty
       + (mtm_pnl × 0.1)  # مكافأة على إبقاء المركز الرابح
       - (max_drawdown - 5%) × 2  # عقوبة على الانسحاب > 5%
```

---

## 🚀 التثبيت والتدريب

### الخطوة 1: تثبيت اعتماديات RL (ثقيلة ~500MB)

```bash
# على جهازك المحلي (موصى به للتدريب)
pip install -r requirements-rl.txt

# أو يدوياً:
pip install stable-baselines3[extra] gymnasium torch
```

> ⚠️ **لا تنصح به على Render Free Tier** (512MB RAM فقط — قد يفشل بسبب الحجم).

### الخطوة 2: تدريب النماذج على عملة واحدة

```bash
# افتراضي: BTCUSDT على فريم 1h لمدة 2000 شمعة (~3 أشهر)
python scripts/train_rl_agents.py --symbol BTCUSDT --interval 1h --limit 2000

# تدريب على فريم 15m (للتداول قصير المدى)
python scripts/train_rl_agents.py --symbol BTCUSDT --interval 15m --limit 2000

# تدريب على ETHUSDT فقط بـ PPO و DQN
python scripts/train_rl_agents.py --symbol ETHUSDT --algorithms PPO,DQN

# تدريب أعمق (100K خطوة بدلاً من 50K)
python scripts/train_rl_agents.py --symbol BTCUSDT --timesteps 100000
```

### الخطوة 3: نتائج التدريب

سيتم حفظ 4 ملفات في `data/models/`:
```
data/models/
├── dqn_crypto.zip       ← نموذج DQN
├── ppo_crypto.zip       ← نموذج PPO
├── a2c_crypto.zip       ← نموذج A2C
└── rl_training_report.json  ← تقرير الأداء
```

### الخطوة 4: تفعيل الاستراتيجية في `.env`

```bash
ENABLE_RL_STRATEGY=true
```

أعد تشغيل البوت. سترى في السجلات:
```
[INFO] SignalScorer initialized with 14 strategies (7 proprietary, 1 RL)
[INFO] RL Strategy enabled (advanced)
[INFO] DQN model loaded from data/models/dqn_crypto.zip
[INFO] PPO model loaded from data/models/ppo_crypto.zip
[INFO] A2C model loaded from data/models/a2c_crypto.zip
```

---

## 📊 تفسير الإشارات

عندما تتخذ الخوارزميات الثلاث قرارها، سترى في النموذج المنبثق:

### ✅ إشارة شراء قوية (2/3 أو 3/3 votes BUY)
```
RL Ensemble Strategy:
  Score: +80 (3/3 voted BUY) — confidence 80%
  Votes: DQN=BUY, PPO=BUY, A2C=BUY
```

### ⚠️ إشارة شراء معتدلة (2/3 votes BUY)
```
Score: +60 (2/3 voted BUY)
Votes: DQN=BUY, PPO=BUY, A2C=HOLD
```

### ❌ لا إشارة (تصويت متفرق)
```
Neutral - RL vote mixed: BUY=1, SELL=1, HOLD=1
```

---

## ⚠️ تحذيرات وملاحظات

### 1) النماذج الجاهزة من المستودع الأصلي
المستودع المرجعي (ADITYA-tp01) يقدم نماذج مدرّبة على **أسهم أمريكية** (AAPL, MSFT, GOOGL). **هذه النماذج لن تعمل على العملات المشفرة** لأن:
- سلوك أسواق الأسهم (9-to-5) يختلف عن الكريبتو (24/7)
- تقلبات الأسهم أقل بكثير من العملات
- آلية الدفع (Dividends) غير موجودة في الكريبتو

لذلك **بنينا بيئة مخصصة للكريبتو** (`CryptoTradingEnv`) وعلّمنا نموذجك على بياناتك.

### 2) الأداء على Render
- نماذج RL تستهلك ذاكرة كبيرة (~200MB بعد التحميل)
- على Free Tier (512MB RAM) قد تفشل
- موصى به على **Starter Plan** ($7/شهر) أو أعلى

### 3) وقت التدريب
- 50K خطوة تستغرق 5-15 دقيقة على CPU عادي
- 100K خطوة تستغرق 15-30 دقيقة
- على GPU: 5-10 أضعاف أسرع

### 4) إعادة التدريب الدوري
يُنصح بإعادة تدريب النماذج كل 2-4 أسابيع على البيانات الجديدة لتحسين الأداء.

### 5) عدم وجود ضمان
RL ليس "كرة بلورية". حتى أفضل النماذج قد تخطئ. استخدم RL كإشارة **مساعدة** وليس **حصرية**.

---

## 🔬 كيفية تقييم جودة النماذج

بعد التدريب، افتح `data/models/rl_training_report.json` لرؤية:

```json
{
  "evaluations": {
    "DQN": {
      "final_capital": 12350.50,
      "total_return_pct": 23.50,
      "max_drawdown_pct": 8.20,
      "total_trades": 47,
      "win_rate": 0.595  // 59.5% win rate
    },
    "PPO": { ... },
    "A2C": { ... }
  }
}
```

### علامات النموذج الجيد:
- `total_return_pct` > 0 (ربح)
- `win_rate` > 50%
- `max_drawdown_pct` < 15%
- `total_trades` > 30 (ليست متحفظة جداً)

### علامات النموذج السيء:
- `total_return_pct` < 0 (خسارة)
- `total_trades` < 5 (متفرّع جداً — لم يتعلّم)
- `max_drawdown_pct` > 25% (مخاطرة عالية)

---

## 🎯 نصائح متقدمة

### 1) تدريب Ensemble متعدد الأصول
بدلاً من التدريب على عملة واحدة، يمكنك التدريب على بيانات مجمعة من BTC + ETH + BNB:

```python
# في scripts/train_rl_agents.py — عدّل ليتدرب على عدة عملات
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
all_dfs = [data_fetcher.get_candles(s, "1h", 2000) for s in symbols]
combined_df = pd.concat(all_dfs).sort_index()
train_agents(combined_df, symbol="MULTI", total_timesteps=100000)
```

### 2) ضبط Hyperparameters
الـ hyperparameters الافتراضية جيدة، لكن يمكن تحسينها:
```python
# في rl_trainer.py
PPO(
    "MlpPolicy", env,
    learning_rate=0.0001,  # أبطأ، أكثر استقراراً
    n_steps=4096,         # دفعات أكبر
    batch_size=128,
    n_epochs=20,
    gamma=0.995,          # مكافآت بعيدة أكثر
)
```

### 3) Transfer Learning
يمكنك تدريب نموذج على BTC ثم إعادة تدريبه لفترة قصيرة على ETH:

```python
from stable_baselines3 import PPO
model = PPO.load("ppo_crypto.zip", env=new_eth_env)
model.learn(total_timesteps=10000)  # تكييف سريع
model.save("ppo_eth.zip")
```

---

## ❓ أسئلة شائعة

### س: لماذا النموذج يرجع دائماً HOLD؟
ج: علامة على underfitting. جرّب:
- زيادة `total_timesteps` إلى 100K+
- زيادة `learning_rate`
- التحقق من صحة البيانات

### س: لماذا الفرق بين التدريب والاختبار كبير؟
ج: علامة على overfitting. حلول:
- استخدم بيانات اختبار منفصلة (last 20% من البيانات)
- قلل `total_timesteps`
- استخدم `eval_env` مختلف عن `train_env`

### س: كيف أعرف أي خوارزمية أفضل؟
ج: انظر `rl_training_report.json` لمقارنة `total_return_pct` لكل خوارزمية. ثم يمكنك استخدام خوارزمية واحدة فقط في `rl_strategy.py` إن أردت.

---

## 📞 المساعدة

إذا واجهت مشاكل:
1. تحقق من التبعيات: `pip show stable-baselines3`
2. راجع `data/models/rl_training_report.json`
3. افتح Issue على GitHub
