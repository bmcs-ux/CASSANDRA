from datetime import datetime, timezone
from typing import Any, Optional

try:
    from mt5linux import MetaTrader5 as _MT5Class
    _mt5 = _MT5Class()          # BUAT INSTANCE
    IS_MT5_LINUX = True
except ImportError:
    try:
        import MetaTrader5 as _mt5  # MODULE
        IS_MT5_LINUX = False
    except ImportError:
        import adapters.dummy_MetaTrader5 as _mt5 # FALLBACK
        IS_MT5_LINUX = False


class MT5Adapter:
    def __init__(self, logger=None, mt5_backend: Optional[Any] = None):
        self._mt5 = mt5_backend if mt5_backend is not None else _mt5
        self._initialized = False
        self._logged_in = False
        self._logger = logger if logger else print
        self._is_mt5linux = IS_MT5_LINUX and mt5_backend is None

        self._log(
            "Using injected MT5 backend." if mt5_backend is not None
            else "Using mt5linux bridge." if self._is_mt5linux
            else "Using native MetaTrader5." if not self._is_mt5linux and mt5_backend is None
            else "Using dummy MetaTrader5 (fallback)."
        )

    def _log(self, msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._logger(f"[{ts}] [MT5Adapter] {msg}")

    def _const(self, name: str, default: Any = None) -> Any:
        return getattr(self._mt5, name, default)

    # --- CORE ---
    def initialize(self):
        if not self._initialized:
            if not self._mt5.initialize():
                self._log(f"initialize() failed: {self._mt5.last_error()}")
                return False
            self._initialized = True
            self._log("MT5 initialized")
        return True

    def login(self, login, password, server):
        if not self.initialize():
            return False
        if not self._logged_in:
            if not self._mt5.login(login, password=password, server=server):
                self._log(f"login() failed: {self._mt5.last_error()}")
                return False
            self._logged_in = True
            self._log(f"Logged in: {login}")
        return True

    def shutdown(self):
        self._mt5.shutdown()
        self._initialized = False
        self._logged_in = False

    def last_error(self):
        return self._mt5.last_error()

    # --- MARKET DATA ---
    def symbol_info(self, symbol):
        return self._mt5.symbol_info(symbol)

    def symbol_select(self, symbol, enable=True):
        symbol_select = getattr(self._mt5, "symbol_select", None)
        if callable(symbol_select):
            return symbol_select(symbol, enable)
        return True

    def symbol_info_tick(self, symbol):
        return self._mt5.symbol_info_tick(symbol)

    def positions_get(self, **kwargs):
        return self._mt5.positions_get(**kwargs)

    # --- ORDER ---
    def order_send(self, request: dict):
        result = self._mt5.order_send(request)
        if result is None:
            self._log(f"order_send returned None: {self.last_error()}")
        return result

    # --- SL / TP MODIFY (CORRECT WAY) ---
    def modify_position(self, ticket, sl, tp):
        request = {
            "action": self.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": float(sl),
            "tp": float(tp),
        }
        return self.order_send(request)

    # --- RATES ---
    def eval(self, code: str):
        backend_eval = getattr(self._mt5, "eval", None)
        if callable(backend_eval):
            return backend_eval(code)

        conn = getattr(self._mt5, "_MetaTrader5__conn", None)
        if conn is not None and hasattr(conn, "eval"):
            return conn.eval(code)

        raise AttributeError("Underlying MT5 backend does not expose eval().")

    def account_info(self):
        account_info = getattr(self._mt5, "account_info", None)
        if callable(account_info):
            return account_info()
        return None

    def copy_rates_range(self, symbol, timeframe, date_from, date_to):
        if not isinstance(date_from, datetime):
            date_from = datetime.fromtimestamp(float(date_from), tz=timezone.utc)
        if not isinstance(date_to, datetime):
            date_to = datetime.fromtimestamp(float(date_to), tz=timezone.utc)
        if date_from.tzinfo is None:
            date_from = date_from.replace(tzinfo=timezone.utc)
        if date_to.tzinfo is None:
            date_to = date_to.replace(tzinfo=timezone.utc)

        return self._mt5.copy_rates_range(
            symbol,
            timeframe,
            date_from,
            date_to,
        )

    # --- CONSTANTS (PASS-THROUGH) ---
    @property
    def TRADE_ACTION_DEAL(self):
        return self._const("TRADE_ACTION_DEAL")

    @property
    def TRADE_ACTION_SLTP(self):
        return self._const("TRADE_ACTION_SLTP")

    @property
    def ORDER_TYPE_BUY(self):
        return self._const("ORDER_TYPE_BUY")

    @property
    def ORDER_TYPE_SELL(self):
        return self._const("ORDER_TYPE_SELL")

    @property
    def ORDER_TIME_GTC(self):
        return self._const("ORDER_TIME_GTC")

    @property
    def ORDER_FILLING_FOK(self):
        return self._const("ORDER_FILLING_FOK", self._const("ORDER_FILLING_FOC"))

    @property
    def ORDER_FILLING_RETURN(self):
        return self._const("ORDER_FILLING_RETURN")

    @property
    def TRADE_RETCODE_DONE(self):
        return self._const("TRADE_RETCODE_DONE")

    @property
    def POSITION_TYPE_BUY(self):
        return self._const("POSITION_TYPE_BUY", self._const("ORDER_TYPE_BUY"))

    @property
    def POSITION_TYPE_SELL(self):
        return self._const("POSITION_TYPE_SELL", self._const("ORDER_TYPE_SELL"))

    # --- TIMEFRAMES ---
    @property
    def TIMEFRAME_M1(self):
        return self._const("TIMEFRAME_M1")

    @property
    def TIMEFRAME_M5(self):
        return self._const("TIMEFRAME_M5")

    @property
    def TIMEFRAME_M15(self):
        return self._const("TIMEFRAME_M15")

    @property
    def TIMEFRAME_M30(self):
        return self._const("TIMEFRAME_M30")

    @property
    def TIMEFRAME_H1(self):
        return self._const("TIMEFRAME_H1")

    @property
    def TIMEFRAME_H4(self):
        return self._const("TIMEFRAME_H4")

    @property
    def TIMEFRAME_D1(self):
        return self._const("TIMEFRAME_D1")

    @property
    def TIMEFRAME_W1(self):
        return self._const("TIMEFRAME_W1")

    @property
    def TIMEFRAME_MN1(self):
        return self._const("TIMEFRAME_MN1")


MT5_TIMEFRAME_MAP = {
    "1m": getattr(_mt5, "TIMEFRAME_M1", None),
    "5m": getattr(_mt5, "TIMEFRAME_M5", None),
    "15m": getattr(_mt5, "TIMEFRAME_M15", None),
    "30m": getattr(_mt5, "TIMEFRAME_M30", None),
    "1h": getattr(_mt5, "TIMEFRAME_H1", None),
    "4h": getattr(_mt5, "TIMEFRAME_H4", None),
    "1d": getattr(_mt5, "TIMEFRAME_D1", None),
    "1w": getattr(_mt5, "TIMEFRAME_W1", None),
    "1M": getattr(_mt5, "TIMEFRAME_MN1", None),
}
