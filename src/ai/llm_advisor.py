"""
v5.9 AI Advisor - Arabic technical explanations via CodeCraft API.

The user obtained an API key from https://codecraftapi.com (OpenAI-
compatible LLM relay, key prefix cc_). This module uses it for ONE job:
explaining each recommendation in 2-3 Arabic sentences ("لماذا هذه
الإشارة؟") so the Telegram card and the dashboard carry a plain-language
read of the technical picture, not just raw indicator names.

Design rules (important):
- OPTIONAL LAYER: no key / disabled / any network error -> the bot sends
  exactly what it sent before (comment simply absent). Never raises.
- READ-ONLY over the rec dict: adds "ai_comment" (sanitized for Telegram
  Markdown) and "ai_model" only.
- CACHED: data/ai_cache.json keyed by (symbol, direction, conf-bucket) with
  a TTL aligned with the Telegram re-notification cooldown (4h default), so
  a stable setup is not re-explained on every cycle.
- PARALLEL: top-N recommendations are enriched concurrently with a per-call
  timeout, bounded by a total budget so the cycle is never blocked long.
- SECRET-SAFE: the API key never leaves this module except inside the
  Authorization header; nothing logs it.
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import requests

from config.settings import settings
from src.utils.logger import log
from src.utils.helpers import load_json, save_json, now_utc

AI_CACHE_FILE = Path("data/ai_cache.json")

# Telegram uses ParseMode=Markdown (legacy). Any of these characters in the
# model output can break parsing and fail the WHOLE message send, so the
# comment is stripped to plain text: Arabic + Latin + digits + punctuation.
_MD_CHARS = re.compile(r"[*_`\[\]()~>#+=|{}]")


def sanitize_comment(text: str, max_len: int = 500) -> str:
    """Make LLM output safe for Telegram legacy Markdown + dashboard HTML."""
    if not text:
        return ""
    # strip markdown emphasis/code/links chars entirely
    text = _MD_CHARS.sub("", text)
    # collapse whitespace / newlines -> single line-ish paragraph
    text = re.sub(r"\s+", " ", text).strip()
    # hard length cap (model occasionally over-talks despite the prompt)
    if len(text) > max_len:
        text = text[: max_len - 1].rsplit(" ", 1)[0] + "…"
    return text


class LLMAdvisor:
    """Thin OpenAI-compatible chat client + Arabic recommendation explainer."""

    def __init__(self):
        self.base_url = settings.CODECRAFT_BASE_URL.rstrip("/")
        self.key = settings.CODECRAFT_API_KEY
        self.model = settings.AI_ADVISOR_MODEL
        self.timeout = settings.AI_ADVISOR_TIMEOUT
        self.enabled = (
            settings.AI_ADVISOR_ENABLED
            and bool(self.key)
        )
        # in-memory cache mirror (persisted to disk on writes)
        self._cache: Dict[str, Dict] = load_json(AI_CACHE_FILE, default={})
        if self.enabled:
            log.info(
                f"[green]AI advisor enabled[/] (codecraftapi.com, "
                f"model={self.model})"
            )
        else:
            log.info("[yellow]AI advisor disabled (no CODECRAFT_API_KEY)[/]")

    # ---------------- low-level chat ----------------

    def _chat(self, system: str, user: str, max_tokens: int = 300) -> Optional[str]:
        """One chat completion. Returns content or None on ANY failure."""
        if not self.enabled:
            return None
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            # the relay sits behind Cloudflare; a browser-ish UA avoids
            # the bot-challenge page on datacenter IPs
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.4,
        }
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            if r.status_code != 200:
                log.warning(f"AI advisor HTTP {r.status_code}: {r.text[:200]}")
                return None
            data = r.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            return content or None
        except Exception as e:
            log.warning(f"AI advisor call failed: {e}")
            return None

    # ---------------- prompt building ----------------

    @staticmethod
    def _system_prompt() -> str:
        return (
            "أنت محلل فني متمرس داخل بوت توصيات لعملات رقمية. "
            "مهمتك شرح إشارة التداول المعطاة بالعربية الفصحى المبسطة، "
            "في جملتين إلى ثلاث جمل كحد أقصى. اشرح منطق الإشارة من المؤشرات "
            "المعطاة فقط (الاتجاه، الزخم، التقلب، حجم التداول)، ولا تخترع "
            "بيانات غير موجودة، ولا تقدم نصائح مالية، ولا تذكر أن هذا تحليل آلي. "
            "اكتب نصاً عادياً بدون أي تنسيق أو رموز خاصة."
        )

    @staticmethod
    def _user_prompt(rec: Dict) -> str:
        d = rec.get("direction", "neutral")
        direction = "شرائي (صاعد)" if d == "bullish" else "هبائي (هابط)" if d == "bearish" else "محايد"
        symbol = rec.get("symbol", "?")
        price = rec.get("current_price", 0)
        sl = rec.get("stop_loss", 0)
        tp = rec.get("take_profit", 0)
        tp2 = rec.get("take_profit_2") or 0
        rr = rec.get("risk_reward_ratio", 0)
        conf = rec.get("confidence", 0)
        atr = rec.get("atr_pct", 0)
        harmony = rec.get("harmony")
        boosted = rec.get("boosted_from_bottom")

        lines = [
            f"العملة: {symbol}",
            f"الإشارة: {direction} بثقة {conf:.0f}%",
            f"السعر الحالي: {price}",
            f"وقف الخسارة: {sl} | الهدف الأول: {tp}" + (f" | الهدف الثاني: {tp2}" if tp2 else ""),
            f"العائد مقابل المخاطرة: {rr:.2f}:1 | التقلب ATR: {atr:.2f}%",
        ]
        if harmony is not None:
            lines.append(f"انسجام الطبقات: {float(harmony) * 100:.0f}%")
        if boosted:
            lines.append("هذه إشارة ارتداد من قاع السوق (اقتراح قاع).")

        # strategy signals - names + top reason each (already English; the
        # model is multilingual and can weave them into Arabic fluently)
        sigs = rec.get("signals") or []
        if sigs:
            lines.append("الاستراتيجيات الدافعة:")
            for s in sigs[:4]:
                name = (s.get("strategy") or "strategy").replace("Strategy", "")
                reasons = s.get("reasons") or []
                top = reasons[0] if reasons else ""
                lines.append(f"- {name}: {top}")
        # confluence context (short)
        icho = rec.get("ichimoku") or {}
        if icho:
            lines.append(
                f"إيشيموكي: النظام {icho.get('regime', '؟')}، "
                f"السعر {icho.get('price_vs_cloud', '؟')} السحابة"
            )
        ell = rec.get("elliott") or {}
        if ell and ell.get("pattern", "unclear") != "unclear":
            lines.append(
                f"إليوت: نمط {ell.get('pattern', '؟')}، "
                f"الموجة {ell.get('current_wave', '؟')}"
            )
        lines.append(
            "اشرح في جملتين إلى ثلاث جمل عربية: لماذا تجتمع هذه العناصر "
            "لدعم هذه الإشارة، وما الخطر الرئيسي الذي يراقبه المتداول "
            "(مثل كسر وقف الخسارة أو تلاشي الحجم)."
        )
        return "\n".join(lines)

    # ---------------- cache ----------------

    @staticmethod
    def _cache_key(rec: Dict) -> str:
        conf_bucket = int(float(rec.get("confidence", 0)) // 5)  # 5% buckets
        return f"{rec.get('symbol', '?')}|{rec.get('direction', 'neutral')}|{conf_bucket}"

    def _cache_get(self, key: str) -> Optional[str]:
        """Return cached comment if younger than TTL."""
        entry = self._cache.get(key)
        if not entry:
            return None
        ts_raw = entry.get("ts", "")
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(ts_raw)
            age_h = (now_utc() - ts).total_seconds() / 3600
            if age_h >= settings.AI_ADVISOR_CACHE_HOURS:
                return None
            return entry.get("comment") or None
        except (ValueError, TypeError):
            return None

    def _cache_put(self, key: str, comment: str):
        self._cache[key] = {"ts": now_utc().isoformat(), "comment": comment}
        # keep the file bounded: latest 300 entries by timestamp
        if len(self._cache) > 300:
            try:
                items = sorted(
                    self._cache.items(),
                    key=lambda kv: kv[1].get("ts", ""), reverse=True)
                self._cache = dict(items[:300])
            except Exception:
                pass
        try:
            save_json(self._cache, AI_CACHE_FILE)
        except Exception as e:
            log.debug(f"AI cache save skipped: {e}")

    # ---------------- public API ----------------

    def explain_recommendation(self, rec: Dict) -> Optional[str]:
        """Generate (or fetch cached) Arabic comment for one recommendation."""
        if not self.enabled:
            return None
        key = self._cache_key(rec)
        cached = self._cache_get(key)
        if cached:
            return cached
        raw = self._chat(self._system_prompt(), self._user_prompt(rec))
        if not raw:
            return None
        comment = sanitize_comment(raw)
        if comment:
            self._cache_put(key, comment)
        return comment or None

    def enrich_recommendations(
        self, recommendations: List[Dict]
    ) -> int:
        """Add 'ai_comment' (+ 'ai_model') to up to AI_ADVISOR_MAX_RECS recs.

        Parallel across recs, bounded by per-call timeout; any failure leaves
        that rec untouched. Returns how many recs got a comment.
        """
        if not self.enabled or not recommendations:
            return 0
        targets = recommendations[: max(1, settings.AI_ADVISOR_MAX_RECS)]

        def _one(rec: Dict):
            c = self.explain_recommendation(rec)
            if c:
                rec["ai_comment"] = c
                rec["ai_model"] = self.model
            return bool(c)

        done = 0
        try:
            with ThreadPoolExecutor(max_workers=min(3, len(targets))) as ex:
                futures = {ex.submit(_one, r): r for r in targets}
                budget = time.time() + self.timeout * 2 + 5  # total budget cap
                for fut in as_completed(futures, timeout=self.timeout * 2 + 5):
                    try:
                        done += 1 if fut.result() else 0
                    except Exception:
                        pass
                    if time.time() > budget:
                        break
        except Exception as e:
            log.warning(f"AI advisor enrich interrupted: {e}")
        if done:
            log.info(f"[green]AI advisor:[/] {done}/{len(targets)} "
                     f"recommendation(s) got an Arabic AI comment")
        return done

    # ---------------- v5.11: trade autopsy lesson ----------------

    @staticmethod
    def _postmortem_system_prompt() -> str:
        return (
            "أنت مدرب تداول متمرس يراجع صفقات بوت آلي بعد إغلاقها. "
            "اقرأ حقول الصفقة (سبب الدخول، الربح الأقصى الذي مرّ به السعر، "
            "التراجع الأعمق، سبب الخروج، المدة، عدد التعديلات) واكتب درساً "
            "واحداً عملياً بالعربية في جملة إلى جملتين كحد أقصى. "
            "علّق على إدارة الصفقة فقط (الهدف، الوقف، التوقيت) ولا تتنبأ "
            "بالمستقبل، ولا تقدم نصائح مالية، ولا تذكر أنك ذكاء اصطناعي. "
            "اكتب نصاً عادياً بدون أي تنسيق أو رموز خاصة."
        )

    @staticmethod
    def _postmortem_user_prompt(t: Dict) -> str:
        """Facts of the closed trade (whole-trade view incl. partials)."""
        total_pct = t.get("total_pnl_pct")
        if total_pct is None:
            total_pct = t.get("pnl_pct", 0) or 0
        facts = [
            f"العملة: {t.get('symbol', '?')}",
            f"النتيجة الإجمالية: {float(total_pct or 0):+.2f}%",
        ]
        if t.get("mfe_pct"):
            facts.append(f"أعلى ربح مرّ بالسعر: +{float(t['mfe_pct']):.2f}%")
        if t.get("mae_pct"):
            facts.append(f"أعمق تراجع: -{float(t['mae_pct']):.2f}%")
        if t.get("capture_efficiency") is not None:
            facts.append(
                f"نسبة التقاط القمة: {float(t['capture_efficiency']):.0f}%")
        if t.get("duration_hours") is not None:
            facts.append(f"المدة: {float(t['duration_hours']):.1f} ساعة")
        if t.get("partials_count"):
            facts.append(f"جني جزئي TP1: {int(t['partials_count'])} مرة")
        if t.get("risk_updates_count"):
            facts.append(f"تعديلات وقف/هدف: {int(t['risk_updates_count'])}")
        reason = t.get("reason") or t.get("close_reason") or ""
        if reason:
            facts.append(f"سبب الخروج: {reason}")
        return "\n".join(facts)

    def trade_postmortem(self, closed_trade: Dict) -> Optional[str]:
        """v5.11 creative layer: one Arabic lesson sentence for a CLOSED trade.

        Best-effort and non-blocking by design (called from a throwaway
        thread in cycle.py): returns None on ANY failure. No caching -
        every closed trade is unique.
        """
        if not self.enabled or not closed_trade:
            return None
        raw = self._chat(
            self._postmortem_system_prompt(),
            self._postmortem_user_prompt(closed_trade),
            max_tokens=160,
        )
        if not raw:
            return None
        return sanitize_comment(raw, max_len=280) or None


# Singleton
ai_advisor = LLMAdvisor()
