"""Binance API client - supports both authenticated and public-only modes.

v5: every outbound request is gated by a weighted sliding-window rate limiter
(Binance allows 6000 weight/min per IP - and Render egress IPs are SHARED).
HTTP 429 responses feed `Retry-After` back into the limiter as a global
cooldown instead of triggering an instant retry storm.
"""
import hmac
import hashlib
import json
import time
import urllib.parse
import requests
from typing import Dict, Any, Optional, List
from config.settings import settings
from src.utils.logger import log
from src.core.rate_limiter import rate_limiter, RateLimitError

# Static request-weight map (Binance spot docs).
# Dynamic endpoints (depth/ticker) are computed from params when needed.
_ENDPOINT_WEIGHTS = {
    "/api/v3/ping": 1,
    "/api/v3/time": 1,
    "/api/v3/exchangeInfo": 20,
    "/api/v3/klines": 2,
    "/api/v3/avgPrice": 2,
    "/api/v3/trades": 25,
    "/api/v3/account": 20,
    "/api/v3/order": 1,          # POST place / DELETE cancel
    "/api/v3/order/oco": 1,
}


def _endpoint_weight(path: str, params: Dict[str, Any]) -> float:
    """Best-effort request weight for the endpoint + params."""
    if path == "/api/v3/klines":
        # weight scales with the limit: [1,100]->1, (100,500]->2,
        # (500,1000]->5, (1000,1500]->10 (Binance spot docs)
        try:
            lim = int(params.get("limit", 500))
        except (TypeError, ValueError):
            lim = 500
        if lim <= 100:
            return 1
        if lim <= 500:
            return 2
        if lim <= 1000:
            return 5
        return 10
    if path == "/api/v3/depth":
        try:
            lim = int(params.get("limit", 20))
        except (TypeError, ValueError):
            lim = 20
        return 5 if lim <= 100 else 10
    if path == "/api/v3/ticker/24hr":
        # full-market call is weight 80 (!) - single symbol is 2
        return 2 if params.get("symbol") else 80
    if path == "/api/v3/ticker/price":
        syms = params.get("symbols")
        if syms:
            try:
                n = len(json.loads(syms)) if isinstance(syms, str) else len(syms)
            except Exception:
                n = 4
            return 2 if n <= 100 else 4
        return 4  # full price list
    if path == "/api/v3/openOrders":
        return 6
    return _ENDPOINT_WEIGHTS.get(path, 2)


class BinanceClient:
    """
    Lightweight REST client for Binance Spot API.
    Uses public endpoints when no API keys are provided.
    """

    def __init__(self):
        self.base_url = settings.BINANCE_BASE_URL  # for public data (works everywhere)
        self.signed_url = settings.BINANCE_SIGNED_URL  # for private endpoints (place orders)
        self.api_key = settings.BINANCE_API_KEY
        self.api_secret = settings.BINANCE_API_SECRET
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})
        log.info(
            f"[cyan]BinanceClient[/] initialized - "
            f"Mode: {'AUTHENTICATED' if not settings.USE_PUBLIC_ONLY else 'PUBLIC-ONLY'} | "
            f"Public URL: {self.base_url} | "
            f"Signed URL: {self.signed_url}"
        )

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sign request params with HMAC-SHA256 (required for private endpoints)."""
        if not self.api_secret:
            raise PermissionError("API secret required for signed endpoints")
        params["timestamp"] = int(time.time() * 1000)
        query = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _get(self, path: str, params: Optional[Dict] = None,
             signed: bool = False) -> Dict[str, Any]:
        """Perform a GET request to Binance API (rate-limit aware, v5)."""
        # Use signed_url for signed requests, base_url for public
        base = self.signed_url if signed else self.base_url
        url = f"{base}{path}"
        params = dict(params or {})
        if signed:
            if settings.USE_PUBLIC_ONLY:
                raise PermissionError(
                    "Signed endpoint called but no API keys configured. "
                    "Set BINANCE_API_KEY and BINANCE_API_SECRET in .env"
                )
            params = self._sign(params)

        # v5: reserve request weight BEFORE firing (blocks when the shared
        # IP is near the ceiling; honours 429 cooldowns globally).
        weight = _endpoint_weight(path, params)
        if not rate_limiter.acquire(weight, timeout=90.0):
            raise RateLimitError(
                f"Rate budget exhausted ({weight}w needed); "
                "request aborted to avoid a 429 ban"
            )

        try:
            response = self.session.get(url, params=params, timeout=15)
            # Feed the server-reported shared-IP usage back into the limiter
            used = response.headers.get("X-MBX-USED-WEIGHT-1M")
            if used:
                rate_limiter.note_server_weight(used)
            if response.status_code == 429 or response.status_code == 418:
                retry_after = response.headers.get("Retry-After", "30")
                try:
                    retry_after = float(retry_after)
                except ValueError:
                    retry_after = 30.0
                if response.status_code == 418:
                    # v5.2: IP auto-ban - Retry-After can be minutes..hours and
                    # every request sent during the ban can EXTEND it. Back off
                    # hard (>= 15 min) instead of poking it every cycle.
                    cooldown = min(max(retry_after, 900.0), 3600.0)
                else:
                    cooldown = min(retry_after + 2, 120.0)
                rate_limiter.trigger_cooldown(cooldown)
                raise RateLimitError(
                    f"Binance {response.status_code}: {response.text[:120]}"
                )
            response.raise_for_status()
            return response.json()
        except RateLimitError:
            raise
        except requests.exceptions.HTTPError as e:
            log.error(f"Binance API HTTP error: {e} - URL: {url} - Response: {response.text[:200]}")
            raise
        except requests.exceptions.RequestException as e:
            log.error(f"Binance API request error: {e}")
            raise

    def get_tickers_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        v5: batched /api/v3/ticker/price for selected symbols.
        Weight 2 (<=100 symbols) instead of 80 for the full /ticker/24hr
        call - the previous code fetched ALL tickers just to read 5 prices.
        Returns {symbol: {"price": float, ...}}.
        """
        if not symbols:
            return {}
        if len(symbols) > 100:
            # endpoint caps at 100 symbols per call
            out: Dict[str, Dict] = {}
            for i in range(0, len(symbols), 100):
                out.update(self.get_tickers_batch(symbols[i:i + 100]))
            return out
        # Binance rejects spaces in the symbols array (code -1100):
        # json.dumps default separator is ", " -> ["A", "B"] is INVALID.
        # separators=(",", ":") produces ["A","B"] as the API requires.
        params = {"symbols": json.dumps(list(symbols), separators=(",", ":"))}
        rows = self._get("/api/v3/ticker/price", params)
        rows = rows if isinstance(rows, list) else [rows]
        return {r["symbol"]: r for r in rows if r.get("symbol")}

    # ============================================
    # PUBLIC ENDPOINTS (no API key needed)
    # ============================================

    def ping(self) -> bool:
        """Test connectivity."""
        try:
            r = self._get("/api/v3/ping")
            return r == {}
        except Exception:
            return False

    def get_server_time(self) -> int:
        """Get Binance server time (ms)."""
        return self._get("/api/v3/time").get("serverTime")

    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange info (all symbols, filters, etc.)."""
        return self._get("/api/v3/exchangeInfo")

    def get_all_tickers(self) -> List[Dict[str, Any]]:
        """Get ticker prices for all symbols."""
        return self._get("/api/v3/ticker/24hr")

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get 24h ticker stats for a symbol."""
        return self._get("/api/v3/ticker/24hr", {"symbol": symbol})

    def get_klines(self, symbol: str, interval: str = "1h",
                   limit: int = 200, start_time: Optional[int] = None,
                   end_time: Optional[int] = None) -> List[List]:
        """
        Get historical klines (candles).
        Response: [[openTime, open, high, low, close, volume, closeTime,
                    quoteAssetVolume, trades, takerBuyBase, takerBuyQuote, ignore], ...]
        """
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return self._get("/api/v3/klines", params)

    def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book depth."""
        return self._get("/api/v3/depth", {"symbol": symbol, "limit": limit})

    def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get recent trades."""
        return self._get("/api/v3/trades", {"symbol": symbol, "limit": limit})

    def get_avg_price(self, symbol: str) -> Dict[str, Any]:
        """Get current average price."""
        return self._get("/api/v3/avgPrice", {"symbol": symbol})

    # ============================================
    # PRIVATE ENDPOINTS (require API key + secret)
    # ============================================

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information (balances, fees, etc.)."""
        return self._get("/api/v3/account", signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get current open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._get("/api/v3/openOrders", params, signed=True)

    def _post(self, path: str, params: Optional[Dict] = None,
              signed: bool = True) -> Dict[str, Any]:
        """Perform a signed POST request (used for placing orders). v5: rate-limit aware."""
        import time as _t
        # Use signed_url (api.binance.com) for private endpoints
        url = f"{self.signed_url}{path}"
        params = dict(params or {})
        params["timestamp"] = int(_t.time() * 1000)
        if signed:
            if settings.USE_PUBLIC_ONLY:
                raise PermissionError(
                    "Signed endpoint called but no API keys configured."
                )
            params = self._sign(params)

        weight = _endpoint_weight(path, params)
        if not rate_limiter.acquire(weight, timeout=90.0):
            raise RateLimitError(
                f"Rate budget exhausted before POST {path}"
            )
        try:
            response = self.session.post(url, params=params, timeout=15)
            used = response.headers.get("X-MBX-USED-WEIGHT-1M")
            if used:
                rate_limiter.note_server_weight(used)
            if response.status_code in (429, 418):
                retry_after = response.headers.get("Retry-After", "30")
                try:
                    retry_after = float(retry_after)
                except ValueError:
                    retry_after = 30.0
                rate_limiter.trigger_cooldown(min(retry_after + 2, 120))
                raise RateLimitError(
                    f"Binance {response.status_code}: {response.text[:120]}"
                )
            response.raise_for_status()
            return response.json()
        except RateLimitError:
            raise
        except requests.exceptions.HTTPError as e:
            log.error(f"Binance POST HTTP error: {e} - Response: {response.text[:300]}")
            raise
        except requests.exceptions.RequestException as e:
            log.error(f"Binance POST request error: {e}")
            raise

    def place_market_buy(self, symbol: str, quote_quantity: float) -> Dict:
        """
        Place a MARKET BUY order using quoteOrderQty (e.g. buy $50 of BTC).
        Returns the order response from Binance.
        """
        if settings.USE_PUBLIC_ONLY:
            raise PermissionError("Cannot place real orders without API keys")
        params = {
            "symbol": symbol,
            "side": "BUY",
            "type": "MARKET",
            "quoteOrderQty": round(quote_quantity, 8),
        }
        log.info(f"[yellow]PLACING REAL MARKET BUY[/] {symbol} quote=${quote_quantity}")
        return self._post("/api/v3/order", params, signed=True)

    def place_limit_buy(self, symbol: str, quantity: float, price: float,
                       time_in_force: str = "GTC") -> Dict:
        """Place a LIMIT BUY order."""
        if settings.USE_PUBLIC_ONLY:
            raise PermissionError("Cannot place real orders without API keys")
        params = {
            "symbol": symbol,
            "side": "BUY",
            "type": "LIMIT",
            "timeInForce": time_in_force,
            "quantity": round(quantity, 8),
            "price": round(price, 8),
        }
        log.info(f"[yellow]PLACING REAL LIMIT BUY[/] {symbol} qty={quantity} @ {price}")
        return self._post("/api/v3/order", params, signed=True)

    def place_oco_sell(self, symbol: str, quantity: float,
                       take_profit_price: float, stop_loss_price: float,
                       stop_limit_price: float) -> Dict:
        """
        Place an OCO (One-Cancels-the-Other) SELL order —
        automatically places both TP and SL. When one triggers, the other is cancelled.
        """
        if settings.USE_PUBLIC_ONLY:
            raise PermissionError("Cannot place real orders without API keys")
        params = {
            "symbol": symbol,
            "side": "SELL",
            "quantity": round(quantity, 8),
            "price": round(take_profit_price, 8),
            "stopPrice": round(stop_loss_price, 8),
            "stopLimitPrice": round(stop_limit_price, 8),
            "stopLimitTimeInForce": "GTC",
            "listOrderSide": "SELL",
        }
        log.info(f"[yellow]PLACING REAL OCO SELL[/] {symbol} qty={quantity} "
                 f"TP={take_profit_price} SL={stop_loss_price}")
        return self._post("/api/v3/order/oco", params, signed=True)

    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel an open order."""
        if settings.USE_PUBLIC_ONLY:
            raise PermissionError("Cannot cancel orders without API keys")
        params = {"symbol": symbol, "orderId": order_id}
        log.info(f"[yellow]CANCELLING ORDER[/] {symbol} id={order_id}")
        return self._post("/api/v3/order", params, signed=True)

    def get_account_balances(self) -> Dict[str, float]:
        """Return non-zero balances {asset: amount}."""
        if settings.USE_PUBLIC_ONLY:
            return {}
        info = self.get_account_info()
        return {
            b["asset"]: float(b["free"])
            for b in info.get("balances", [])
            if float(b["free"]) > 0
        }

    def get_symbol_filters(self, symbol: str) -> Dict:
        """Get trading filters for a symbol (LOT_SIZE, PRICE_FILTER, etc.)."""
        info = self.get_exchange_info()
        for s in info.get("symbols", []):
            if s["symbol"] == symbol:
                filters = {f["filterType"]: f for f in s.get("filters", [])}
                return {
                    "base_asset": s.get("baseAsset"),
                    "quote_asset": s.get("quoteAsset"),
                    "lot_size_step": float(filters.get("LOT_SIZE", {}).get("stepSize", 0.00000001)),
                    "lot_size_min": float(filters.get("LOT_SIZE", {}).get("minQty", 0)),
                    "lot_size_max": float(filters.get("LOT_SIZE", {}).get("maxQty", 0)),
                    "min_notional": float(filters.get("MIN_NOTIONAL", {}).get("minNotional", 10)),
                    "tick_size": float(filters.get("PRICE_FILTER", {}).get("tickSize", 0.00000001)),
                }
        return {}

    def round_quantity_to_lot(self, symbol: str, quantity: float) -> float:
        """Round quantity to the symbol's LOT_SIZE step."""
        filters = self.get_symbol_filters(symbol)
        if not filters:
            return quantity
        step = filters["lot_size_step"]
        min_qty = filters["lot_size_min"]
        if quantity < min_qty:
            return 0
        # Round down to nearest step
        import math
        rounded = math.floor(quantity / step) * step
        return round(rounded, 8)

    # ============================================
    # CONVENIENCE METHODS
    # ============================================

    def get_top_usdt_symbols_by_volume(self, limit: int = 50) -> List[str]:
        """Get top USDT spot pairs by 24h quote volume."""
        tickers = self.get_all_tickers()
        usdt_pairs = [
            t for t in tickers
            if t.get("symbol", "").endswith("USDT")
            and float(t.get("quoteVolume", 0)) > 0
        ]
        usdt_pairs.sort(key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)
        return [t["symbol"] for t in usdt_pairs[:limit]]


# Singleton
binance_client = BinanceClient()
