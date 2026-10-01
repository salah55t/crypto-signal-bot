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

    # --- v5 "Veteran Trader" entry harmony ( layered agreement gate ) ---
    # Harmony = how well ALL layers agree (regime + cycle + entry zone +
    # strategy confluence), 0..1. An experienced trader demands layered
    # agreement, not just a high single score.
    MIN_HARMONY: float = float(os.getenv("MIN_HARMONY", "0.45"))
    # Skip chaotic candles (ATR% of price above max) and dead markets (below min)
    # v5.18: 0.12 -> 0.30 — on the 4h primary TF a 0.12% floor admits
    # gold/pegged tokens (XAUTUSDT paid 0.2% fees for a 0.13% move).
    # Real tradeable alts sit far above 0.5% ATR on 4h.
    ATR_PCT_MAX: float = float(os.getenv("ATR_PCT_MAX", "3.5"))
    ATR_PCT_MIN: float = float(os.getenv("ATR_PCT_MIN", "0.30"))
    EXCLUDE_VOLATILITY_EXTREME: bool = os.getenv(
        "EXCLUDE_VOLATILITY_EXTREME", "true").lower() == "true"
    # Market tide filter: block new longs when BTC regime is strongly bearish
    MARKET_FILTER_ENABLED: bool = os.getenv(
        "MARKET_FILTER_ENABLED", "true").lower() == "true"
    MARKET_FILTER_SYMBOL: str = os.getenv("MARKET_FILTER_SYMBOL", "BTCUSDT")
    MARKET_FILTER_CACHE_MIN: int = int(os.getenv("MARKET_FILTER_CACHE_MIN", "30"))

    # --- v5 Pending limit entries (buy the pocket, don't chase) ---
    PENDING_ENTRIES_ENABLED: bool = os.getenv(
        "PENDING_ENTRIES_ENABLED", "true").lower() == "true"
    MAX_PENDING_ENTRIES: int = int(os.getenv("MAX_PENDING_ENTRIES", "6"))
    PENDING_TTL_HOURS: float = float(os.getenv("PENDING_TTL_HOURS", "4"))
    # If price is more than 0.35 ATR above the golden-pocket zone -> arm a
    # pending limit entry instead of chasing with a market buy.
    PENDING_CHASE_ATR: float = float(os.getenv("PENDING_CHASE_ATR", "0.35"))
    # Cancel a pending entry when price collapses this far BELOW the zone
    PENDING_INVALID_ATR: float = float(os.getenv("PENDING_INVALID_ATR", "0.5"))
    # Momentum bypass: A+ setups / very high confidence enter at market even
    # above the zone (veterans chase ONLY the strongest momentum).
    PENDING_MOMENTUM_CONF: float = float(os.getenv("PENDING_MOMENTUM_CONF", "82"))
    # v5.17: hard sanity cap on the momentum bypass - even an A+ setup pays
    # at most this many ATRs above the zone; beyond it every entry waits
    # for the pullback. Without the cap the bypass opened market positions
    # 13-20% above their planned limit zone (confidence is regime-inflated).
    PENDING_MOMENTUM_MAX_ATR: float = float(
        os.getenv("PENDING_MOMENTUM_MAX_ATR", "1.0"))
    # v5.19: reach cap for pending entries. Production showed pendings armed
    # with the golden pocket 4.9-5.0 ATR below price (AAVE/NVDABUSDT
    # 2026-09-29) and a 4h TTL - a 5-ATR pullback inside 4 hours is a crash,
    # not an entry, so those orders were guaranteed to expire unfilled while
    # the strategy looked "weak". Zones farther than this many ATRs above
    # the price drop the rec entirely; closer-but-far zones get a
    # distance-scaled TTL (1x..4x PENDING_TTL_HOURS).
    PENDING_REACH_MAX_ATR: float = float(os.getenv("PENDING_REACH_MAX_ATR", "3.0"))

    # --- v5 Market-aware position management (exits like a pro) ---
    # Take 50% off at TP1, move SL to break-even+fees, trail the rest to TP2.
    PARTIAL_TP_ENABLED: bool = os.getenv("PARTIAL_TP_ENABLED", "true").lower() == "true"
    PARTIAL_TP_FRACTION: float = float(os.getenv("PARTIAL_TP_FRACTION", "0.5"))
    # SL placed at entry*(1 + buffer) after TP1 so fees never turn it into a loss
    TP1_FEE_BUFFER_PCT: float = float(os.getenv("TP1_FEE_BUFFER_PCT", "0.25"))
    # ATR chandelier trailing: trail SL `mult` x ATR below the highest seen price
    CHANDELIER_ENABLED: bool = os.getenv("CHANDELIER_ENABLED", "true").lower() == "true"
    CHANDELIER_ATR_MULT: float = float(os.getenv("CHANDELIER_ATR_MULT", "2.5"))
    # Min unrealized profit (%) before the chandelier trail arms itself.
    # Default 1.0 keeps v5 behaviour; comprehensive backtest (2026-09-23)
    # showed SL_initial trades gave back avg +2.11% MFE before dying at -1%.
    CHANDELIER_ACTIVATE_PCT: float = float(os.getenv("CHANDELIER_ACTIVATE_PCT", "1.0"))
    # Structure exits: Ichimoku regime flip / opposite strong signal
    STRUCTURAL_EXITS_ENABLED: bool = os.getenv(
        "STRUCTURAL_EXITS_ENABLED", "true").lower() == "true"
    # v5.5: close a position IMMEDIATELY when fresh analysis signals the
    # opposite direction at/above this confidence. Was 75 -> too slow: a
    # bearish 60-74% read only "tightened" the SL 0.5% below price, which
    # LOCKS A LOSS when the trade is underwater, then the stop gets hit.
    # User rule: bearish analysis on a long = exit now, never wait for SL.
    OPPOSITE_SIGNAL_CONF: float = float(os.getenv("OPPOSITE_SIGNAL_CONF", "55"))
    # Milder opposite pressure (conf >= this, < exit threshold) -> tighten
    # SL to 0.5% below price as a defensive step instead of exiting
    SIGNAL_TIGHTEN_CONF: float = float(os.getenv("SIGNAL_TIGHTEN_CONF", "40"))
    # v5.5 continuation updates: when analysis still says "up" with decent
    # confidence AND the trade is already in profit, extend the target AND
    # raise the SL in the SAME update so every extension secures profit.
    # (Old behaviour extended TP alone - 3 updates could still end negative.)
    CONTINUATION_CONF: float = float(os.getenv("CONTINUATION_CONF", "65"))
    # Min unrealized profit (%) before a continuation TP-extension may fire
    CONTINUATION_MIN_PROFIT_PCT: float = float(
        os.getenv("CONTINUATION_MIN_PROFIT_PCT", "1.0"))

    # --- v5.11 Persistent Trade Ledger (entry prices live in the DB) ---
    # Every open/partial/level-update/close is mirrored into the positions
    # table + trade_events audit trail. On startup the JSON working set is
    # reconciled with the ledger in BOTH directions, so a wiped disk
    # (Render redeploy) can no longer erase entry prices or updated levels.
    TRADE_LEDGER_RESTORE: bool = os.getenv(
        "TRADE_LEDGER_RESTORE", "true").lower() == "true"
    # Fraction of the CURRENT profit to lock into the SL on continuation
    # (0.5 = trade up +3% -> SL to at least +1.5%; ladder may raise it more)
    PROFIT_LOCK_FRACTION: float = float(os.getenv("PROFIT_LOCK_FRACTION", "0.5"))
    # Time stop: a trade that goes nowhere is dead capital
    # v5.5: scaled up for the 4h timeframe (24h = just 6 bars on 4h)
    MAX_TRADE_HOURS: float = float(os.getenv("MAX_TRADE_HOURS", "72"))
    TIME_STOP_MIN_PNL_PCT: float = float(os.getenv("TIME_STOP_MIN_PNL_PCT", "0.5"))
    ABSOLUTE_MAX_TRADE_HOURS: float = float(os.getenv("ABSOLUTE_MAX_TRADE_HOURS", "120"))

    # --- v5.7 Signal Stack: user-specified strategy trio ---
    # The "Signal Stack Framework" golden rule: every strategy must combine
    # ONE indicator from each class - Direction / Momentum / Volume-Volatility
    # - never stack same-class indicators (RSI+Stoch+MACD together = one
    # redundant vote). Each new strategy follows that rule:
    #   1. TripleConfluenceTrend: EMA200+EMA50 (dir) + RSI (mom) + Vol/OBV (liq)
    #   2. BBMeanReversion:       BB bands (vol) + Stochastic 14,3,3 (mom) + candle/vol
    #   3. MACDBreakout:          EMA50 (dir) + MACD 12,26,9 (mom) + Vol (liq)
    STRATEGY_TRIPLE_TREND_ENABLED: bool = os.getenv(
        "STRATEGY_TRIPLE_TREND_ENABLED", "true").lower() == "true"
    STRATEGY_TRIPLE_TREND_WEIGHT: float = float(
        os.getenv("STRATEGY_TRIPLE_TREND_WEIGHT", "1.6"))
    STRATEGY_BB_MEAN_REV_ENABLED: bool = os.getenv(
        "STRATEGY_BB_MEAN_REV_ENABLED", "true").lower() == "true"
    STRATEGY_BB_MEAN_REV_WEIGHT: float = float(
        os.getenv("STRATEGY_BB_MEAN_REV_WEIGHT", "1.2"))
    STRATEGY_MACD_BREAKOUT_ENABLED: bool = os.getenv(
        "STRATEGY_MACD_BREAKOUT_ENABLED", "true").lower() == "true"
    STRATEGY_MACD_BREAKOUT_WEIGHT: float = float(
        os.getenv("STRATEGY_MACD_BREAKOUT_WEIGHT", "1.4"))
    # Confluence scale reference. The strength x confluence confidence model
    # was CALIBRATED on the original 3-strategy stack (total weight 5.8):
    # "one strong strategy ~= 64%, two agreeing ~= 76%". Adding strategies
    # must not dilute that scale (a lone signal would sink from 64% towards
    # 58% and MIN_CONFIDENCE=68 would silently demand 3-of-6 agreement).
    # confluence = agree_w / max(total_weight, REF) capped at 1.0 - the
    # original stack behaves bit-identically; extra strategies can only ADD
    # confluence when they actually agree, never punish a lone signal.
    STRATEGY_CONFLUENCE_REF_WEIGHT: float = float(
        os.getenv("STRATEGY_CONFLUENCE_REF_WEIGHT", "5.8"))

    # --- Rate limiting (Binance 6000 weight/min, shared Render IP) ---
    RATE_LIMIT_BUDGET_PER_MIN: int = int(os.getenv("RATE_LIMIT_BUDGET_PER_MIN", "4500"))
    # Order book snapshots are cached this many minutes (weight 5 each)
    ORDER_BOOK_TTL_MIN: int = int(os.getenv("ORDER_BOOK_TTL_MIN", "30"))

    # --- v5.3 WebSocket candle feed (zero REST weight for klines) ---
    # Binance WS streams bypass the REST request-weight budget entirely.
    # One combined connection keeps a live candle cache for the universe,
    # seeded by the first REST cycle and updated incrementally. Every consumer
    # (analyzer, bottom scanner, market map, backtests via REST) reads the
    # cache first and falls back to REST when it is disabled/stale.
    # v5.5: multi-interval - the feed subscribes to every interval in
    # WS_INTERVALS (strategy TFs + 1h) with per-(symbol,interval) caches.
    USE_WS_FEED: bool = os.getenv("USE_WS_FEED", "true").lower() == "true"
    # Market-data WS endpoint (data-stream.binance.vision = public data twin
    # of data-api.binance.vision; stream.binance.com also works)
    WS_ENDPOINT: str = os.getenv("WS_ENDPOINT", "wss://data-stream.binance.vision/stream")
    # Cache older than this many minutes is considered stale -> REST fallback
    # (15 min << 1 bar, so a closed 1h bar can never be silently missing)
    WS_FRESH_TTL_MIN: float = float(os.getenv("WS_FRESH_TTL_MIN", "15"))
    # Dynamic symbol list refresh: /ticker/24hr costs weight 80 - cache it
    # this many minutes instead of refetching every cycle (was every cycle)
    SYMBOL_REFRESH_MIN: int = int(os.getenv("SYMBOL_REFRESH_MIN", "60"))

    # --- v5.14 WS-first: live prices over WS + zero-REST degraded cycles ---
    # The REST ban is per-IP on the REST API only - WS streams keep flowing.
    # 1) miniTicker streams keep a live last-price for every universe symbol
    #    (position watch, dashboard P&L, pending fills = zero REST weight).
    WS_PRICE_TTL_S: float = float(os.getenv("WS_PRICE_TTL_S", "70"))
    # 2) When a 429/418 cooldown is active, the analysis cycle still runs
    #    entirely from the WS candle cache ("degraded" cycle, zero REST
    #    weight) if at least this fraction of the universe is fresh.
    WS_DEGRADED_COVERAGE: float = float(os.getenv("WS_DEGRADED_COVERAGE", "0.9"))
    # 3) Cold-start/missing-series seeding paces ONE REST fetch per delay
    #    seconds (300 weight spread over minutes instead of an 8-worker burst
    #    that collides with shared-IP pressure).
    WS_SEED_DELAY_S: float = float(os.getenv("WS_SEED_DELAY_S", "0.35"))

    # --- v5.15 ban-survivor: state that outlives process restarts ---------
    # The dynamic USDT universe is persisted to data/symbols_cache.json after
    # every successful fetch; a restart during a REST 418/429 ban boots from
    # this cache (accepted while younger than this many hours) instead of
    # firing an 80-weight /ticker/24hr into the banned IP.
    SYMBOLS_CACHE_MAX_AGE_H: float = float(os.getenv("SYMBOLS_CACHE_MAX_AGE_H", "168"))
    # Cooldown state (data/rate_state.json) is written for any ban >= this
    # many seconds - short shared-IP pressure blips are not worth inheriting
    # across restarts.
    RATE_STATE_MIN_S: float = float(os.getenv("RATE_STATE_MIN_S", "60"))

    # --- v5.16 session clock: the fixed daily rhythm of every desk --------
    # Diagnostic (90d, 171 live-logic trades): hours 11-14 UTC (London close
    # -> US pre-market) ran PF 0.05-0.79 over 54 trades; Asia hours 00-03
    # ran PF 2.6-11. Entries only - exits are never session-gated.
    SESSION_FILTER_ENABLED: bool = os.getenv(
        "SESSION_FILTER_ENABLED", "true").lower() == "true"
    # Hard entry blackout (UTC hours, comma list) - the chop window.
    SESSION_BLOCK_HOURS: str = os.getenv("SESSION_BLOCK_HOURS", "11,12,13,14")
    # Saturday is the thinnest day - block breakout/momentum NEW entries
    # unless the rec is A+ (Sunday keeps the v5.13 weight cuts only).
    SESSION_SATURDAY_BLOCK_BREAKOUTS: bool = os.getenv(
        "SESSION_SATURDAY_BLOCK_BREAKOUTS", "true").lower() == "true"
    # Mon 00:00-01:00 unwind: weekend wicks flush right after the weekly
    # open - reversal (knife-catching) entries blocked, trends allowed.
    SESSION_MONDAY_BLOCK_REVERSALS: bool = os.getenv(
        "SESSION_MONDAY_BLOCK_REVERSALS", "true").lower() == "true"

    # --- v5.16 pinned / quasi-stable coin protection -----------------------
    # Pegged or fiat-quoted pairs move a few basis points a day - fees and
    # spread alone guarantee a slow bleed. Hard-skipped before any analysis.
    PEGGED_SYMBOLS: list = [s.strip().upper() for s in os.getenv(
        "PEGGED_SYMBOLS",
        "USDCUSDT,FDUSDUSDT,TUSDUSDT,USDPUSDT,DAIUSDT,EURUSDT,EURIUSDT,"
        "AEURUSDT,USD1USDT,XUSDUSDT,USTCUSDT,FRAXUSDT,BUSDUSDT,PAXGUSDT,"
        "XUSDUSDT,USDEUSDT,USDGUSDT,BUSDUSDT"
    ).split(",") if s.strip()]
    # Statistical flatness: rolling range % of the last FLAT_LOOKBACK bars.
    # A "semi-stable" alt on 4h can pass a tiny ATR floor but never passes
    # this. Keyed by primary timeframe (fallback: FLAT_RANGE_PCT_DEFAULT).
    FLAT_LOOKBACK: int = int(os.getenv("FLAT_LOOKBACK", "48"))
    FLAT_RANGE_PCT_BY_TF: dict = {
        "15m": 0.8, "30m": 1.1, "1h": 1.6, "2h": 2.4, "4h": 3.8,
        "6h": 5.0, "8h": 6.0, "12h": 8.0, "1d": 12.0,
    }
    FLAT_RANGE_PCT_DEFAULT: float = float(os.getenv("FLAT_RANGE_PCT_DEFAULT", "1.6"))
    # Also require a minimum mean absolute per-bar return (%) - catches
    # pinned coins whose range comes from a single old spike.
    FLAT_RETURN_ABS_MIN: float = float(os.getenv("FLAT_RETURN_ABS_MIN", "0.05"))

    # --- Position hygiene ---
    # Skip a recommendation if the same symbol already has an open position
    # (prevents duplicate entries on consecutive cycles)
    SKIP_DUPLICATE_SYMBOLS: bool = os.getenv("SKIP_DUPLICATE_SYMBOLS", "true").lower() == "true"
    # Telegram re-notification cooldown per symbol (hours) - prevents spam
    TELEGRAM_COOLDOWN_HOURS: float = float(os.getenv("TELEGRAM_COOLDOWN_HOURS", "4"))

    # --- Analysis ---
    # Timeframes for analysis. For scalping: "15m" (single) or "15m,1h" (multi-TF confirmation)
    # v5.1 (2026-09-23): default switched 15m -> 1h. Comprehensive backtest
    # (18 symbols x 3000 bars + 8 x 2000 @15m) showed 1h expectancy +0.815%/trade
    # (PF 1.83) vs 15m +0.127%/trade (PF 1.15) — 6.4x better per trade.
    # v5.5 (2026-09-24): user moved the primary timeframe to 4h (position
    # trading; fewer noise-driven updates, cleaner trend following). Market
    # map / market-cycle correlation stays on 1h regardless.
    TIMEFRAMES: list = [tf.strip() for tf in os.getenv("TIMEFRAMES", "4h").split(",")]
    # v5.5: intervals the WS kline feed subscribes to = strategy TFs + 1h
    # (1h always kept: market map correlation + leader cycle read it free).
    WS_INTERVALS: list = sorted({*TIMEFRAMES, "1h"})
    # Run every 10 minutes by default (was: hourly)
    # Examples: "*/10 * * * *" = every 10 min | "0 * * * *" = hourly | "*/30 * * * *" = every 30 min
    SCHEDULE_CRON: str = os.getenv("SCHEDULE_CRON", "*/10 * * * *")
    ORDER_BOOK_DEPTH: int = int(os.getenv("ORDER_BOOK_DEPTH", "20"))
    # v5.7: 200 -> 300. EMA200 needs warmup: 200 bars give exactly ONE valid
    # EMA200 value (min_periods=period) - statistically weak. 300 bars leave
    # 100 warmup bars. Binance klines weight is unchanged (101-500 -> weight 2).
    CANDLE_LIMIT: int = int(os.getenv("CANDLE_LIMIT", "300"))

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

    # --- v5.6 Bottom-fishing admission (why bottom coins never opened) ---
    # The boost channel was written when MIN_CONFIDENCE was 60. Two hidden
    # couplings then silently killed it:
    #   1) v4.1 raised MIN_CONFIDENCE 60 -> 68 while the boost formula
    #      (45 + score/3, cap 72) needs score >= 69 to clear 68 - so the
    #      documented "score >= 60" gate became a de-facto 69 gate.
    #   2) validate_recommendation's harmony gate rejected EVERY boosted rec
    #      (they carry no harmony key -> 0.0 < 0.45 = "Harmony too low"),
    #      so even a boosted recommendation could NEVER become a position.
    # v5.6 gives the channel its own explicit knobs and a bounce-derived
    # harmony instead of the hidden coupling to the strategy scale.
    BOTTOM_BOOST_ENABLED: bool = os.getenv(
        "BOTTOM_BOOST_ENABLED", "true").lower() == "true"
    # Min bounce score (0..100) for a bottom candidate to be boosted
    # (documented intent was 60; 62 + bullish-close + RR gates keep it safe)
    BOTTOM_STRONG_SCORE: float = float(os.getenv("BOTTOM_STRONG_SCORE", "62"))
    # Max boosted bottom entries per cycle (they compete for the top slots)
    BOTTOM_MAX_PER_CYCLE: int = int(os.getenv("BOTTOM_MAX_PER_CYCLE", "2"))
    # Confidence cap for boosted entries so genuine strategy signals rank first
    BOTTOM_CONF_CAP: float = float(os.getenv("BOTTOM_CONF_CAP", "72"))
    # v5.18: bottom-channel volatility floors (user rule: exclude semi-stable
    # coins whose price is near-fixed). Scanned on TIMEFRAMES[0] (4h).
    # XAUTUSDT (gold) runs ~0.2-0.4% ATR on 4h -> rejected; real alts > 1%.
    BOTTOM_MIN_ATR_PCT: float = float(os.getenv("BOTTOM_MIN_ATR_PCT", "0.60"))
    # v5.18: the 2.5x-ATR bottom TP must clear round-trip fees (0.2%) + edge.
    # 0.9% minimum keeps every bottom trade worth taking after costs.
    BOTTOM_MIN_TP_PCT: float = float(os.getenv("BOTTOM_MIN_TP_PCT", "0.90"))
    # v5.18: grace window (minutes) after entry during which the Ichimoku
    # regime / Kijun structural exits are suppressed (hard SL + opposite
    # signal >= 55 stay armed). Prevents the next-cycle instant kill that
    # closed every bottom-fishing trade ~9 minutes after entry.
    STRUCTURAL_EXIT_GRACE_MIN: float = float(
        os.getenv("STRUCTURAL_EXIT_GRACE_MIN", "30"))
    # v5.19: bottom-trade TP ladder. The scanner fishes a 4h BOUNCE, but the
    # old single TP at 2.5x ATR was a day-scale target: production showed
    # avg MFE 0.60% vs a ~10% TP (trades covered 0-19% of the way) while the
    # 30-min structural churn closed everything first. Now TP1 banks half
    # the bounce at BOTTOM_TP1_ATR_MULT (veteran partial flow: SL -> BE),
    # the runner aims for the classic 2.5x-ATR target (TP2).
    BOTTOM_TP1_ATR_MULT: float = float(os.getenv("BOTTOM_TP1_ATR_MULT", "1.2"))
    BOTTOM_TP2_ATR_MULT: float = float(os.getenv("BOTTOM_TP2_ATR_MULT", "2.5"))
    # v5.20 PRODUCTION FORENSICS (6 closed trades, all bottom_scanner_boost,
    # PF 0.24, losses 4x wins): TP1 = 1.2x ATR on a 4h ATR of 2.4-4.6% sits
    # 2.9-5.5% away while the observed bounces died at MFE 0.11-2.86%
    # (median 1.6%) -> TP1 filled 1/6 trades, and only on the lowest-ATR
    # coin. The cap makes the bank leg reachable: min(1.2x ATR, 1.5%).
    BOTTOM_TP1_CAP_PCT: float = float(os.getenv("BOTTOM_TP1_CAP_PCT", "1.5"))
    # v5.20: NO maximum ATR gate existed - ZAMAUSDT (4h ATR ~4.6%) took a
    # 5.5% stop that alone produced 79% of the net loss (MFE 0.11% = the
    # entry never worked). Bottoms are bounce trades: above ~3% 4h ATR the
    # noise swamps the signal and the ladder distances become fantasy.
    BOTTOM_MAX_ATR_PCT: float = float(os.getenv("BOTTOM_MAX_ATR_PCT", "3.0"))
    # v5.20: entry-clustering caps. 2026-09-29 put 5 bottom entries into the
    # market within 6 hours on a falling tape (per-cycle cap only counts
    # entries INSIDE one cycle). Cap concurrent bottom positions and enforce
    # a minimum spacing between consecutive bottom entries.
    BOTTOM_MAX_OPEN_CONCURRENT: int = int(
        os.getenv("BOTTOM_MAX_OPEN_CONCURRENT", "2"))
    BOTTOM_ENTRY_SPACING_MIN: float = float(
        os.getenv("BOTTOM_ENTRY_SPACING_MIN", "45"))
    # v5.20: bottom longs stand against the tape by design, but not into a
    # bearish 1h BTC regime (the 6-trade burst was 6/6 bullish bottoms into
    # a falling market). Reuses the market_tide.json cache/fetch.
    BOTTOM_BTC_TIDE_GATE: bool = os.getenv(
        "BOTTOM_BTC_TIDE_GATE", "true").lower() == "true"
    # v5.20: stagnation exit for bottom trades. MAX_TRADE_HOURS=72h is a
    # trend-trade horizon; a bounce that has not appeared within 6h (pnl
    # < 0.2% AND MFE < 0.6% - it never went anywhere) is dead capital:
    # INTCB bled -2.59% over 11.6h with MFE 0.22%, ZAMA -5.71% with MFE
    # 0.11%. Recycle the slot while the loss is still small.
    BOTTOM_STAGNATION_HOURS: float = float(
        os.getenv("BOTTOM_STAGNATION_HOURS", "6"))
    BOTTOM_STAGNATION_MAX_PNL_PCT: float = float(
        os.getenv("BOTTOM_STAGNATION_MAX_PNL_PCT", "0.20"))
    BOTTOM_STAGNATION_MAX_MFE_PCT: float = float(
        os.getenv("BOTTOM_STAGNATION_MAX_MFE_PCT", "0.60"))
    # v5.20: finer profit ladder (settings-driven, mirrored for shorts).
    # The old coarse ladder (+1% -> BE, +2% -> +1%) gave back 50-100% of
    # every bounce: ZEC peaked +1.95% and exited -0.20% (BE), CAKE peaked
    # +2.86% and exited +0.80%. Locking a third of the move at +1% turns
    # the ZEC case from a fee-loss into a small win while keeping the
    # higher rungs intact for real runners.
    LADDER_LOCK1_PCT: float = float(os.getenv("LADDER_LOCK1_PCT", "1.0"))
    LADDER_LOCK1_LEVEL_PCT: float = float(
        os.getenv("LADDER_LOCK1_LEVEL_PCT", "0.30"))
    LADDER_LOCK2_PCT: float = float(os.getenv("LADDER_LOCK2_PCT", "2.0"))
    LADDER_LOCK2_LEVEL_PCT: float = float(
        os.getenv("LADDER_LOCK2_LEVEL_PCT", "1.10"))
    LADDER_LOCK3_PCT: float = float(os.getenv("LADDER_LOCK3_PCT", "3.0"))
    LADDER_LOCK3_LEVEL_PCT: float = float(
        os.getenv("LADDER_LOCK3_LEVEL_PCT", "2.0"))
    LADDER_TRAIL_PCT: float = float(os.getenv("LADDER_TRAIL_PCT", "5.0"))

    # --- v5.22 Double-Indicator momentum channel (user strategy, spot-only,
    #     long-only) ---
    # The user's documented strategy, applied to the letter:
    #   Bollinger Bands (period 11, deviation 3) + SuperTrend (ATR 2,
    #   multiplier 2). BUY only when >= 3 consecutive green candles sit
    #   ABOVE the SuperTrend line AND very close to the UPPER Bollinger
    #   band. The mirrored SELL setup is deliberately NOT traded (spot has
    #   no shorts; the user asked for the upward side only).
    # Timeframe note: the source doc uses 15s/30s binary-options candles;
    # Binance spot klines only go down to 1m, which is also the doc's own
    # trade duration - so DOUBLE_IND_TIMEFRAME defaults to "1m".
    # Exit note: the doc has no exits (binary options expire). For spot we
    # keep the entry rules exact and add a fee-survival exit ladder: on 1m
    # candles a typical ATR target (0.05-0.2%) is smaller than the 0.2%
    # round-trip fee, so TP1/TP2/SL carry percentage FLOORS that keep every
    # trade worth taking after costs (the v5.18 "TP inside fees" lesson).
    DOUBLE_IND_ENABLED: bool = os.getenv(
        "DOUBLE_IND_ENABLED", "true").lower() == "true"
    DOUBLE_IND_TIMEFRAME: str = os.getenv("DOUBLE_IND_TIMEFRAME", "1m")
    # Scan only the most liquid head of the dynamic universe (sorted by 24h
    # quote volume): scalping belongs where the book is deep. 15 x weight-2
    # klines calls = 30 REST weight per cycle (the bottom scanner spends 384).
    DOUBLE_IND_TOP_N: int = int(os.getenv("DOUBLE_IND_TOP_N", "15"))
    DOUBLE_IND_KLINES_LIMIT: int = int(os.getenv("DOUBLE_IND_KLINES_LIMIT", "90"))
    # Exact indicator settings from the user's document.
    DOUBLE_IND_BB_PERIOD: int = int(os.getenv("DOUBLE_IND_BB_PERIOD", "11"))
    DOUBLE_IND_BB_DEV: float = float(os.getenv("DOUBLE_IND_BB_DEV", "3.0"))
    DOUBLE_IND_ST_PERIOD: int = int(os.getenv("DOUBLE_IND_ST_PERIOD", "2"))
    DOUBLE_IND_ST_MULT: float = float(os.getenv("DOUBLE_IND_ST_MULT", "2.0"))
    # Entry conditions ("100% or nothing" - the doc's golden rule).
    DOUBLE_IND_MIN_GREEN: int = int(os.getenv("DOUBLE_IND_MIN_GREEN", "3"))
    # "Very close to the upper band": last close in the top 10% of the
    # band range (percent_B >= 0.90) AND the 3-candle run averages >= 0.80.
    DOUBLE_IND_BB_PCTB_MIN: float = float(
        os.getenv("DOUBLE_IND_BB_PCTB_MIN", "0.90"))
    DOUBLE_IND_BB_PCTB_AVG_MIN: float = float(
        os.getenv("DOUBLE_IND_BB_PCTB_AVG_MIN", "0.80"))
    # Semi-stable / fee-food gates (1m scale): a pinned coin never passes
    # the ATR floor; a dead range never passes the rolling-range floor.
    DOUBLE_IND_MIN_ATR_PCT: float = float(
        os.getenv("DOUBLE_IND_MIN_ATR_PCT", "0.12"))
    DOUBLE_IND_MIN_RANGE_PCT: float = float(
        os.getenv("DOUBLE_IND_MIN_RANGE_PCT", "0.60"))
    # Fee-survival exit ladder (percentage floors + ATR multiples).
    # SL = max(1.5 x ATR, SL floor); TP1 = max(1.2 x ATR, TP1 floor);
    # TP2 = max(2.5 x ATR, TP2 floor); RR is computed on TP2.
    DOUBLE_IND_SL_MIN_PCT: float = float(
        os.getenv("DOUBLE_IND_SL_MIN_PCT", "0.55"))
    DOUBLE_IND_TP1_MIN_PCT: float = float(
        os.getenv("DOUBLE_IND_TP1_MIN_PCT", "0.50"))
    DOUBLE_IND_TP2_MIN_PCT: float = float(
        os.getenv("DOUBLE_IND_TP2_MIN_PCT", "1.00"))
    DOUBLE_IND_TP1_ATR_MULT: float = float(
        os.getenv("DOUBLE_IND_TP1_ATR_MULT", "1.2"))
    DOUBLE_IND_TP2_ATR_MULT: float = float(
        os.getenv("DOUBLE_IND_TP2_ATR_MULT", "2.5"))
    DOUBLE_IND_SL_ATR_MULT: float = float(
        os.getenv("DOUBLE_IND_SL_ATR_MULT", "1.5"))
    # Confidence mapping for the channel (admission still passes through
    # validate_recommendation's regime-adjusted MIN_CONFIDENCE gate).
    DOUBLE_IND_CONF_BASE: float = float(
        os.getenv("DOUBLE_IND_CONF_BASE", "76.0"))
    DOUBLE_IND_CONF_CAP: float = float(
        os.getenv("DOUBLE_IND_CONF_CAP", "80.0"))
    # Thin-volume burst penalty (below this ratio of the 20-bar average the
    # 3-candle run is suspect - fee-food filter, ranking only).
    DOUBLE_IND_MIN_VOL_RATIO: float = float(
        os.getenv("DOUBLE_IND_MIN_VOL_RATIO", "0.8"))
    # Channel caps (mirror of the v5.20 bottom-channel clustering guards).
    DOUBLE_IND_MAX_PER_CYCLE: int = int(
        os.getenv("DOUBLE_IND_MAX_PER_CYCLE", "1"))
    DOUBLE_IND_MAX_OPEN_CONCURRENT: int = int(
        os.getenv("DOUBLE_IND_MAX_OPEN_CONCURRENT", "1"))
    DOUBLE_IND_ENTRY_SPACING_MIN: float = float(
        os.getenv("DOUBLE_IND_ENTRY_SPACING_MIN", "20"))
    # Same burst persists across cycles - do not re-fire a symbol while its
    # 3-candle run is still the same momentum episode.
    DOUBLE_IND_SYMBOL_COOLDOWN_MIN: float = float(
        os.getenv("DOUBLE_IND_SYMBOL_COOLDOWN_MIN", "30"))
    DOUBLE_IND_BTC_TIDE_GATE: bool = os.getenv(
        "DOUBLE_IND_BTC_TIDE_GATE", "true").lower() == "true"

    # --- Market Map: leader/follower correlation classification (v5.2) ---
    # Many altcoins chart almost identically to a major (BTC/SOL/XRP...).
    # We classify each symbol to its highest-correlated leader and use the
    # leader's trend as a regime filter: follower of a bearish leader gets a
    # confidence penalty (can demote out), bullish leader gets a rank-only
    # boost (same v4 philosophy as other boosts - merit gates admission).
    MARKET_MAP_LEADERS: list = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    # Assign to a leader only if Pearson corr(returns) >= this; else independent
    MARKET_MAP_CORR_THRESHOLD: float = float(os.getenv("MARKET_MAP_CORR_THRESHOLD", "0.55"))
    # Correlation window: hours of 1h closes used for returns (168 = 7 days)
    MARKET_MAP_LOOKBACK_HOURS: int = int(os.getenv("MARKET_MAP_LOOKBACK_HOURS", "168"))
    # How often the map is recomputed (cached in data/market_map.json)
    MARKET_MAP_REFRESH_HOURS: float = float(os.getenv("MARKET_MAP_REFRESH_HOURS", "6"))
    # Regime action: bearish leader -> confidence -= corr * PENALTY (admission too)
    LEADER_BEARISH_CONF_PENALTY: float = float(os.getenv("LEADER_BEARISH_CONF_PENALTY", "20"))
    # Regime action: bullish leader -> confidence += corr * BOOST (rank-only)
    LEADER_BULLISH_CONF_BOOST: float = float(os.getenv("LEADER_BULLISH_CONF_BOOST", "8"))
    # v5.4: a build where MORE than this fraction of the universe failed to
    # fetch candles is REJECTED (not cached) - a leaders-only/holey map used
    # to be saved as "fresh" and the dashboard groups tab stayed empty for
    # the whole 6h TTL (root cause of the empty market-groups tab).
    MARKET_MAP_MAX_FAIL_PCT: float = float(os.getenv("MARKET_MAP_MAX_FAIL_PCT", "0.25"))

    # --- Market cycle via leader coins (v5.4) ---
    # "دورة تحليل السوق عبر العملات السيدة": a periodic deep read of the
    # leader coins (trend + RSI + momentum + SMA50 distance) that produces a
    # market-wide verdict and a human-readable classification file
    # (data/market_groups.txt) used for decision-making.
    MARKET_CYCLE_REFRESH_MIN: int = int(os.getenv("MARKET_CYCLE_REFRESH_MIN", "60"))
    MARKET_GROUPS_FILE: str = os.getenv("MARKET_GROUPS_FILE", "data/market_groups.txt")

    # --- v5.9 AI Advisor (CodeCraft API - OpenAI-compatible LLM relay) ---
    # The user obtained an API key from https://codecraftapi.com (cc_...).
    # We use it to generate a short ARABIC technical explanation of every
    # recommendation ("تحليل ذكي") that is appended to the Telegram card and
    # shown on the dashboard. Fully optional: if the key is missing or the
    # call fails, the bot sends everything as before (graceful degradation).
    CODECRAFT_BASE_URL: str = os.getenv("CODECRAFT_BASE_URL", "https://codecraftapi.com")
    CODECRAFT_API_KEY: str = os.getenv("CODECRAFT_API_KEY", "")
    AI_ADVISOR_ENABLED: bool = os.getenv("AI_ADVISOR_ENABLED", "true").lower() == "true"
    # Fast + cheap model is the right default for 2-3 sentence answers.
    # Catalog (v1/models) includes: gpt-5.5/5.6-*, claude-opus-*, gemini-3.*,
    # glm-5.*, deepseek-v4-*, qwen3.8-*, kimi-k3, grok-4.*, muse-spark-1.1.
    AI_ADVISOR_MODEL: str = os.getenv("AI_ADVISOR_MODEL", "gemini-3.6-flash")
    # Per-call timeout (seconds) and how many top recommendations get comments
    AI_ADVISOR_TIMEOUT: int = int(os.getenv("AI_ADVISOR_TIMEOUT", "25"))
    AI_ADVISOR_MAX_RECS: int = int(os.getenv("AI_ADVISOR_MAX_RECS", "5"))
    # Cache TTL (hours) for a generated comment - aligns with the Telegram
    # re-notification cooldown so the same setup is not re-explained.
    AI_ADVISOR_CACHE_HOURS: float = float(os.getenv("AI_ADVISOR_CACHE_HOURS", "4"))

    # --- v5.10 rate-limiter priority lane ---
    # The last N weight units of RATE_LIMIT_BUDGET_PER_MIN are reserved for
    # PRIORITY requests (position watch, dashboard P&L, pending fills) so a
    # bulk analysis burst can never starve open-position monitoring.
    RATE_PRIORITY_RESERVE: float = float(os.getenv("RATE_PRIORITY_RESERVE", "200"))

    # --- v5.13 Regime Router (right strategy for the market state) ---
    # Market state = leaders (BTC/ETH/SOL/XRP) + Fear & Greed + weekend +
    # BTC volatility. Routes strategy ranking, admission gates and sizing.
    REGIME_ENABLED: bool = os.getenv("REGIME_ENABLED", "true").lower() == "true"
    REGIME_REFRESH_MIN: int = int(os.getenv("REGIME_REFRESH_MIN", "15"))
    # Fear & Greed index (alternative.me, free, no key; updates daily)
    FNG_ENABLED: bool = os.getenv("FNG_ENABLED", "true").lower() == "true"
    FNG_CACHE_HOURS: float = float(os.getenv("FNG_CACHE_HOURS", "4"))
    # Weekend window: Fri 22:00 UTC -> Mon 00:00 UTC (thin liquidity)
    WEEKEND_FILTER: bool = os.getenv("WEEKEND_FILTER", "true").lower() == "true"

    # --- v5.21 Professional multi-timeframe (MTF) context ---
    # "Daily for context, 1h for timing, trade only with the macro tide."
    # When unavailable (bans, no data) every strategy fails OPEN - the
    # pre-v5.21 behaviour is bit-identical.
    MTF_ENABLED: bool = os.getenv("MTF_ENABLED", "true").lower() == "true"
    # 1d macro frame fetch (weight 2, TTL-cached; skipped during REST bans)
    MTF_FETCH_DAILY: bool = os.getenv("MTF_FETCH_DAILY", "true").lower() == "true"
    MTF_DAILY_TTL_MIN: int = int(os.getenv("MTF_DAILY_TTL_MIN", "60"))
    MTF_DAILY_FAIL_TTL_S: int = int(os.getenv("MTF_DAILY_FAIL_TTL_S", "300"))
    # 1h tactical frame (WS-cached, free) - timing confirmation
    MTF_LTF_ENABLED: bool = os.getenv("MTF_LTF_ENABLED", "true").lower() == "true"
    # Confidence bonus when >= 2 strategies agree AND the daily tide agrees
    MTF_CONF_BONUS: float = float(os.getenv("MTF_CONF_BONUS", "3.0"))
    # v5.21 volatility-breakout quality gates (professional fakeout filter):
    # a breakout candle needs real volume; a vertical candle is a chase.
    VOL_BREAKOUT_MIN_VOL_RATIO: float = float(
        os.getenv("VOL_BREAKOUT_MIN_VOL_RATIO", "1.2"))
    VOL_BREAKOUT_CHASE_ATR: float = float(
        os.getenv("VOL_BREAKOUT_CHASE_ATR", "2.5"))
    # v5.21 BB mean-reversion regime guard: fade band extremes only when
    # the tape is not a violent trend (mean reversion collapses in regime
    # breaks - the single most-cited professional caveat).
    BB_REGIME_ADX_MAX: float = float(os.getenv("BB_REGIME_ADX_MAX", "32.0"))

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
