from datetime import datetime, timezone

try:
    from mt5linux import MetaTrader5 as _MT5Class
    _mt5 = _MT5Class()          #  BUAT INSTANCE
    IS_MT5_LINUX = True
except ImportError:
    import MetaTrader5 as _mt5  #  MODULE
    IS_MT5_LINUX = False


class MT5Adapter:
    def __init__(self, logger=None):
        self._mt5 = _mt5
        self._initialized = False
        self._logged_in = False
        self._logger = logger if logger else print

        self._log(
            "Using mt5linux bridge." if IS_MT5_LINUX
            else "Using native MetaTrader5."
        )

    def _log(self, msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._logger(f"[{ts}] [MT5Adapter] {msg}")

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
        return self._mt5.symbol_select(symbol, enable)

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
        return self._mt5._MetaTrader5__conn.eval(code)

    def account_info(self):
        return self._mt5.account_info()

#from datetime import datetime, timezone

    def copy_rates_range(self, symbol, timeframe, date_from, date_to):

        # HARD NORMALIZATION (ANTI mt5linux BUG)
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
            date_to
        )

    # --- CONSTANTS (PASS-THROUGH) ---
    TRADE_ACTION_DEAL = _mt5.TRADE_ACTION_DEAL
    TRADE_ACTION_SLTP = _mt5.TRADE_ACTION_SLTP

    ORDER_TYPE_BUY = _mt5.ORDER_TYPE_BUY
    ORDER_TYPE_SELL = _mt5.ORDER_TYPE_SELL

    ORDER_TIME_GTC = _mt5.ORDER_TIME_GTC
    ORDER_FILLING_FOK = _mt5.ORDER_FILLING_FOK
    ORDER_FILLING_RETURN = _mt5.ORDER_FILLING_RETURN

    TRADE_RETCODE_DONE = _mt5.TRADE_RETCODE_DONE

    POSITION_TYPE_BUY = _mt5.POSITION_TYPE_BUY
    POSITION_TYPE_SELL = _mt5.POSITION_TYPE_SELL

    # --- TIMEFRAMES ---
    TIMEFRAME_M1 = _mt5.TIMEFRAME_M1
    TIMEFRAME_M5 = _mt5.TIMEFRAME_M5
    TIMEFRAME_M15 = _mt5.TIMEFRAME_M15
    TIMEFRAME_M30 = _mt5.TIMEFRAME_M30
    TIMEFRAME_H1 = _mt5.TIMEFRAME_H1
    TIMEFRAME_H4 = _mt5.TIMEFRAME_H4
    TIMEFRAME_D1 = _mt5.TIMEFRAME_D1
    TIMEFRAME_W1 = _mt5.TIMEFRAME_W1
    TIMEFRAME_MN1 = _mt5.TIMEFRAME_MN1

MT5_TIMEFRAME_MAP = {
    "1m": MT5Adapter.TIMEFRAME_M1,
    "5m": MT5Adapter.TIMEFRAME_M5,
    "15m": MT5Adapter.TIMEFRAME_M15,
    "30m": MT5Adapter.TIMEFRAME_M30,
    "1h": MT5Adapter.TIMEFRAME_H1,
    "4h": MT5Adapter.TIMEFRAME_H4,
    "1d": MT5Adapter.TIMEFRAME_D1,
    "1w": MT5Adapter.TIMEFRAME_W1,
    "1M": MT5Adapter.TIMEFRAME_MN1,
}

