# ============================================
# Crypto Signal Bot - Configuration Module
# ============================================
import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Load .env
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """Central configuration loaded from environment variables."""

    # --- Binance API ---
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    # Public market data endpoint - works in regions where api.binance.com is blocked (451 error)
    # Options:
    #   - https://data-api.binance.vision  (RECOMMENDED - public data, no geo-block, no API key needed)
    #   - https://api.binance.com          (default - blocked in US/UK)
    #   - https://api1.binance.com         (alt endpoint)
    #   - https://api2.binance.com         (alt endpoint)
    #   - https://api3.binance.com         (alt endpoint)
    #   - https://api4.binance.com         (alt endpoint)
    #   - https://testnet.binance.vision   (Binance Testnet)
    BINANCE_BASE_URL: str = os.getenv("BINANCE_BASE_URL", "https://data-api.binance.vision")
    # Separate URL for signed endpoints (place orders, account info) - must be api.binance.com
    # If you live in US/UK and need live trading, you'll need to use Binance Testnet or a VPN
    BINANCE_SIGNED_URL: str = os.getenv("BINANCE_SIGNED_URL", "https://api.binance.com")
    BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    USE_PUBLIC_ONLY: bool = not BINANCE_API_KEY or not BINANCE_API_SECRET

    # --- Telegram ---
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    TELEGRAM_ENABLED: bool = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"

    # --- Dashboard ---
    DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8080"))
    DASHBOARD_HOST: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")

    # --- Bot ---
    RUN_MODE: str = os.getenv("RUN_MODE", "paper")  # paper | live
    # v2 confidence scale (strength x confluence): neutral market = 0%,
    # one strong strategy ~= 64%, two agreeing ~= 76%. 60 = quality floor.
    # v4.1: raised 60 -> 68 — live monitoring (2026-09-23: 45% WR on 20 trades)
    # showed the 60-68 band churns marginal entries; 68+ keeps high-confluence setups only.
    MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "68"))
    MIN_EXPECTED_RISE: float = float(os.getenv("MIN_EXPECTED_RISE", "1.0"))
    # Top 5 recommendations only (was 15) - sent to Telegram + opened as positions
    MAX_RECOMMENDATIONS: int = int(os.getenv("MAX_RECOMMENDATIONS", "5"))
    # Fixed trade amount in USD per position (default: $10 — configurable)
    TRADE_AMOUNT_USD: float = float(os.getenv("TRADE_AMOUNT_USD", "10"))
    # Binance spot trading fee: 0.1% per side (0.075% if paying with BNB)
    TRADING_FEE_PCT: float = float(os.getenv("TRADING_FEE_PCT", "0.1"))
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", "1.0"))
    # Maximum 5 open positions (matches MAX_RECOMMENDATIONS - all top 5 become positions)
    MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "5"))
    DAILY_MAX_LOSS: float = float(os.getenv("DAILY_MAX_LOSS", "5.0"))
    # Minimum risk/reward for any recommendation (SL/TP are built to satisfy this)
    # v4.1: grid-tested conf{60,65,68} x RR{1.5,1.6,1.8} on BTC+ETH 1000x1h:
    # RR 1.5 wins on average (ETH degrades at higher RR), conf 68 wins overall
    # (avg PF 3.70, avg expectancy +1.35%/trade vs v4's 2.02 / +0.90%).
    MIN_RR_RATIO: float = float(os.getenv("MIN_RR_RATIO", "1.5"))

    # --- v4.1 Loss-avoidance guards (based on live paper monitoring) ---
    # Daily cap on NEW positions opened (was unbounded: 20+/day churned fees).
    MAX_TRADES_PER_DAY: int = int(os.getenv("MAX_TRADES_PER_DAY", "12"))
    # After N consecutive losing closes, pause new entries for a few hours (anti-tilt).
    LOSS_STREAK_LIMIT: int = int(os.getenv("LOSS_STREAK_LIMIT", "3"))
    LOSS_STREAK_PAUSE_HOURS: float = float(os.getenv("LOSS_STREAK_PAUSE_HOURS", "4"))
    # After a symbol hits Stop Loss, block re-opening it for N hours
    # (prevents re-entering the same chop that caused the loss).
    REENTRY_COOLDOWN_HOURS: float = float(os.getenv("REENTRY_COOLDOWN_HOURS", "2"))

    # --- v4 Integrated Confluence (Ichimoku + Elliott) ---
    # When True, a signal whose direction opposes the Ichimoku cloud regime
    # is VETOED (excluded from Telegram + positions). Set false to only
    # discount instead of veto.
    CONFLUENCE_VETO_ENABLED: bool = os.getenv("CONFLUENCE_VETO_ENABLED", "true").lower() == "true"

    # --- Position hygiene ---
    # Skip a recommendation if the same symbol already has an open position
    # (prevents duplicate entries on consecutive cycles)
    SKIP_DUPLICATE_SYMBOLS: bool = os.getenv("SKIP_DUPLICATE_SYMBOLS", "true").lower() == "true"
    # Telegram re-notification cooldown per symbol (hours) - prevents spam
    TELEGRAM_COOLDOWN_HOURS: float = float(os.getenv("TELEGRAM_COOLDOWN_HOURS", "4"))

    # --- Analysis ---
    # Timeframes for analysis. For scalping: "15m" (single) or "15m,1h" (multi-TF confirmation)
    TIMEFRAMES: list = [tf.strip() for tf in os.getenv("TIMEFRAMES", "15m").split(",")]
    # Run every 10 minutes by default (was: hourly)
    # Examples: "*/10 * * * *" = every 10 min | "0 * * * *" = hourly | "*/30 * * * *" = every 30 min
    SCHEDULE_CRON: str = os.getenv("SCHEDULE_CRON", "*/10 * * * *")
    ORDER_BOOK_DEPTH: int = int(os.getenv("ORDER_BOOK_DEPTH", "20"))
    CANDLE_LIMIT: int = int(os.getenv("CANDLE_LIMIT", "200"))

    # --- Symbol Selection ---
    # When True: fetch ALL USDT pairs listed on Binance (~400 symbols) and use them
    # When False: use the custom list from config/coins.yaml
    USE_ALL_USDT_PAIRS: bool = os.getenv("USE_ALL_USDT_PAIRS", "false").lower() == "true"
    # Filter: only keep pairs with 24h quote volume >= this (in USDT)
    # Default: $5M (filters out low-liquidity pairs)
    MIN_VOLUME_USDT: float = float(os.getenv("MIN_VOLUME_USDT", "5000000"))
    # Cap maximum number of symbols to analyze per cycle (prevents OOM on Render Free Tier)
    # Binance has ~400 USDT pairs; on Free Tier (512MB RAM), cap at 100-150
    MAX_SYMBOLS: int = int(os.getenv("MAX_SYMBOLS", "150"))
    # Parallel workers for fetching data (Binance IP limit: 6000 weight/min)
    # Each klines call = weight 2, each order book = weight 5
    # 10 workers × ~6 calls/sec = 60 calls/sec = OK for Binance
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "8"))
    # Skip order book fetch to speed up analysis (still uses klines + ticker)
    # Set to true for faster analysis of many symbols (loses liquidity strategy signal)
    SKIP_ORDER_BOOK: bool = os.getenv("SKIP_ORDER_BOOK", "false").lower() == "true"

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "data/logs/bot.log")

    # --- GitHub ---
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GITHUB_REPO_NAME: str = os.getenv("GITHUB_REPO_NAME", "crypto-signal-bot")
    GITHUB_REPO_DESCRIPTION: str = os.getenv(
        "GITHUB_REPO_DESCRIPTION",
        "Cryptocurrency Spot Trading Signal Bot for Binance with Multi-Strategy Analysis"
    )

    # --- Capital (for paper trading) ---
    INITIAL_CAPITAL: float = float(os.getenv("INITIAL_CAPITAL", "10000"))

    @classmethod
    def load_coins(cls) -> list:
        """Load custom coin list from config/coins.yaml"""
        coins_file = PROJECT_ROOT / "config" / "coins.yaml"
        if not coins_file.exists():
            return []
        with open(coins_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        coins = data.get("coins", [])
        # Append USDT suffix
        return [f"{c}USDT" for c in coins]

    @classmethod
    def load_excluded(cls) -> list:
        coins_file = PROJECT_ROOT / "config" / "coins.yaml"
        if not coins_file.exists():
            return []
        with open(coins_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("exclude", [])


# Singleton instance
settings = Settings()
