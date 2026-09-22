"""
Telegram Notification Bot
Sends formatted recommendations to a Telegram chat.
"""
import requests
from typing import List, Dict
from config.settings import settings
from src.utils.logger import log
from src.utils.helpers import fmt_price, fmt_pct


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
        if self.enabled:
            log.info("[green]Telegram notifier enabled[/]")
        else:
            log.info("[yellow]Telegram notifier disabled (no token/chat id)[/]")

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
        """Send a formatted list of recommendations (top 5 only)."""
        if not self.enabled or not recommendations:
            return 0
        # Limit to top 5 (MAX_RECOMMENDATIONS already does this, but enforce)
        top_5 = recommendations[:5]
        sent = 0
        # Header — emphasize these are TRACKED positions
        header = (
            f"*🎯 TOP {len(top_5)} SIGNALS — AUTO-TRACKED POSITIONS*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"⏰ _{recommendations[0].get('analyzed_at', 'N/A')[:19]}_\n"
            f"💼 Mode: `{settings.RUN_MODE.upper()}`\n"
            f"💵 Trade size: `${settings.TRADE_AMOUNT_USD}` per position\n"
            f"📊 Bot will OPEN + TRACK all {len(top_5)} positions\n"
            f"🔄 Dynamic SL/TP updates via continuous analysis\n"
            f"🛑 Auto-close on SL/TP hit (checked every 1 min)\n"
            f"━━━━━━━━━━━━━━━\n"
        )
        self.send(header)
        # Send each recommendation
        for i, rec in enumerate(top_5, 1):
            msg = self._format_recommendation(rec, position_num=i, total=len(top_5))
            if self.send(msg):
                sent += 1
        # Footer
        footer = (
            f"\n📈 *Total: {len(top_5)} positions opened & tracked*\n"
            f"⏭️ Next update in ~5 minutes (next analysis cycle)\n"
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
