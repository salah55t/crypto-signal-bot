"""
Telegram Notification Bot
Sends formatted recommendations to a Telegram chat.

v2: per-symbol re-notification cooldown — the same symbol is not re-sent
within TELEGRAM_COOLDOWN_HOURS (default 4h) unless it comes with an open
position alert. Cooldown state persists in data/telegram_state.json.
"""
import requests
from pathlib import Path
from typing import List, Dict
from config.settings import settings
from src.utils.logger import log
from src.utils.helpers import fmt_price, fmt_pct, load_json, save_json, now_utc

TG_STATE_FILE = Path("data/telegram_state.json")


class TelegramNotifier:
    """Sends formatted trading signals to Telegram."""

    def __init__(self):
        self.token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID
        self.enabled = (
            settings.TELEGRAM_ENABLED
            and self.token
            and self.chat_id
        )
        self._sent_at: Dict[str, str] = load_json(TG_STATE_FILE, default={})
        if self.enabled:
            log.info("[green]Telegram notifier enabled[/]")
        else:
            log.info("[yellow]Telegram notifier disabled (no token/chat id)[/]")

    def _in_cooldown(self, symbol: str) -> bool:
        """True if this symbol was notified within the cooldown window."""
        cooldown_h = settings.TELEGRAM_COOLDOWN_HOURS
        if cooldown_h <= 0:
            return False
        last = self._sent_at.get(symbol)
        if not last:
            return False
        try:
            from datetime import datetime
            last_dt = datetime.fromisoformat(last)
            hours_since = (now_utc() - last_dt).total_seconds() / 3600
            return hours_since < cooldown_h
        except (ValueError, TypeError):
            return False

    def _mark_sent(self, symbol: str):
        """Record notification time for a symbol (persisted)."""
        self._sent_at[symbol] = now_utc().isoformat()
        # Keep the state file small: keep only the latest 200 entries
        if len(self._sent_at) > 200:
            sorted_items = sorted(
                self._sent_at.items(), key=lambda kv: kv[1], reverse=True
            )
            self._sent_at = dict(sorted_items[:200])
        save_json(self._sent_at, TG_STATE_FILE)

    def send(self, message: str) -> bool:
        """Send a plain text message to Telegram."""
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code != 200:
                log.error(f"Telegram send failed: {r.text}")
                return False
            return True
        except Exception as e:
            log.error(f"Telegram error: {e}")
            return False

    def send_recommendations(self, recommendations: List[Dict]) -> int:
        """Send a formatted list of recommendations (top 5 only).
        Symbols already notified within the cooldown window are skipped."""
        if not self.enabled or not recommendations:
            return 0
        # Limit to top 5 (MAX_RECOMMENDATIONS already does this, but enforce)
        top_5 = recommendations[:5]

        # Cooldown filter
        fresh = [r for r in top_5 if not self._in_cooldown(r.get("symbol", ""))]
        skipped = len(top_5) - len(fresh)
        if skipped:
            log.info(f"[yellow]Telegram cooldown:[/] skipped {skipped} recently-notified symbol(s)")
        if not fresh:
            return 0

        sent = 0
        # Header — emphasize these are TRACKED positions
        header = (
            f"*🎯 TOP {len(fresh)} SIGNALS — AUTO-TRACKED POSITIONS*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"⏰ _{fresh[0].get('analyzed_at', 'N/A')[:19]}_\n"
            f"💼 Mode: `{settings.RUN_MODE.upper()}`\n"
            f"💵 Trade size: `${settings.TRADE_AMOUNT_USD}` per position\n"
            f"📊 Bot will OPEN + TRACK all {len(fresh)} positions\n"
            f"🔄 Dynamic SL/TP updates via continuous analysis\n"
            f"🛑 Auto-close on SL/TP hit (checked every cycle)\n"
            f"━━━━━━━━━━━━━━━\n"
        )
        self.send(header)
        # Send each recommendation
        for i, rec in enumerate(fresh, 1):
            msg = self._format_recommendation(rec, position_num=i, total=len(fresh))
            if self.send(msg):
                sent += 1
                self._mark_sent(rec.get("symbol", ""))
        # Footer
        footer = (
            f"\n📈 *Total: {len(fresh)} positions opened & tracked*\n"
            f"⏭️ Next update in the next analysis cycle\n"
            f"🔔 You'll receive alerts on SL/TP hits + risk updates\n\n"
            f"⚠️ _Educational use only. Trade responsibly._"
        )
        self.send(footer)
        log.info(f"[green]Sent {sent} top recommendations to Telegram[/]")
        return sent

    def _format_recommendation(self, rec: Dict, position_num: int = 0,
                                 total: int = 0) -> str:
        direction = rec.get("direction", "neutral").upper()
        emoji = "🟢" if direction == "BULLISH" else "🔴" if direction == "BEARISH" else "⚪"
        symbol = rec.get("symbol", "")
        price = rec.get("current_price", 0)
        conf = rec.get("confidence", 0)
        expected = rec.get("expected_rise_pct", 0)
        sl = rec.get("stop_loss", 0)
        tp = rec.get("take_profit", 0)
        rr = rec.get("risk_reward_ratio", 0)
        score = rec.get("weighted_score", 0)
        atr_pct = rec.get("atr_pct", 0)

        # Top reasons (combine strategy reasons)
        reasons = []
        for sig in rec.get("signals", []):
            for r in sig.get("reasons", [])[:2]:  # top 2 per strategy
                reasons.append(r)
        reasons_text = "\n".join(f"• {r}" for r in reasons[:8]) or "No specific signals"

        return (
            f"{emoji} *{symbol}* — `{direction}`"
            + (f" (Position {position_num}/{total})" if position_num else "")
            + f"\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 *Price:* `{fmt_price(price)}`\n"
            f"📈 *Expected Rise:* `{fmt_pct(expected)}`\n"
            f"🎯 *Confidence:* `{conf:.1f}%` (score: {score:+.1f})\n"
            f"🛑 *Stop Loss:* `{fmt_price(sl)}` ({fmt_pct((sl-price)/price*100)})\n"
            f"✅ *Take Profit:* `{fmt_price(tp)}` ({fmt_pct((tp-price)/price*100)})\n"
            f"⚖️ *R/R Ratio:* `{rr:.2f}:1`\n"
            f"📊 *ATR:* `{atr_pct:.2f}%`\n"
            f"━━━━━━━━━━━━━━━\n"
            f"*Signals:*\n{reasons_text}"
        )

    def send_alert(self, title: str, message: str) -> bool:
        """Send a single alert message."""
        return self.send(f"⚠️ *{title}*\n\n{message}")


# Singleton
telegram_notifier = TelegramNotifier()
