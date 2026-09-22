"""ML package - lazy loading to handle optional RL dependencies."""
from src.strategies.ml_strategy import MLStrategy

def get_crypto_trading_env():
    """Lazy import for CryptoTradingEnv (requires gymnasium)."""
    from src.ml.rl_environment import CryptoTradingEnv
    return CryptoTradingEnv
