import collections
from datetime import datetime, timezone
import random
from typing import Dict, Optional

import numpy as np
import pandas as pd

# --- Mock MT5 Constants ---
TRADE_ACTION_DEAL = 0
TRADE_ACTION_SLTP = 1
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
POSITION_TYPE_BUY = ORDER_TYPE_BUY
POSITION_TYPE_SELL = ORDER_TYPE_SELL
ORDER_TIME_GTC = 0
ORDER_FILLING_FOC = 0
ORDER_FILLING_FOK = ORDER_FILLING_FOC
ORDER_FILLING_RETURN = 1

TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_REJECT = 10013
TRADE_RETCODE_NO_CONNECTION = 10001

TIMEFRAME_M1 = 1
TIMEFRAME_M5 = 5
TIMEFRAME_M15 = 15
TIMEFRAME_M30 = 30
TIMEFRAME_H1 = 60
TIMEFRAME_H4 = 240
TIMEFRAME_D1 = 1440
TIMEFRAME_W1 = 10080
TIMEFRAME_MN1 = 43200

# --- Mock MT5 Data Structures ---
TradeResult = collections.namedtuple(
    'TradeResult',
    ['retcode', 'deal', 'order', 'volume', 'price', 'comment', 'request_id', 'retcode_external', 'position'],
)
SymbolInfo = collections.namedtuple(
    'SymbolInfo',
    ['name', 'point', 'digits', 'trade_tick_size', 'trade_volume_min', 'trade_volume_max', 'trade_volume_step'],
)
SymbolInfoTick = collections.namedtuple('SymbolInfoTick', ['time', 'bid', 'ask', 'last', 'volume'])
Position = collections.namedtuple('Position', ['ticket', 'symbol', 'type', 'volume', 'price_open', 'time', 'sl', 'tp', 'magic', 'comment'])
TerminalInfo = collections.namedtuple('TerminalInfo', ['community_account', 'community_connection', 'connected', 'dll_s_allowed', 'trade_allowed', 'trade_expert', 'max_deviation', 'product_id', 'product_name', 'company', 'language', 'path', 'max_bar_variation', 'max_calc_threads', 'cpu_core_count', 'ping_last', 'ping_prev'])
AccountInfo = collections.namedtuple('AccountInfo', ['login', 'balance', 'equity', 'margin', 'margin_free'])

# --- Mock MT5 State Variables ---
is_initialized = False
is_logged_in = False
logged_in_account = None
selected_symbols = set()

# --- Mock Symbol Data ---
mock_symbol_data = {
    "USDCAD": {"point": 0.0001, "digits": 5, "bid": 1.3750, "ask": 1.3755, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01},
    "USDDXY": {"point": 0.001, "digits": 3, "bid": 98.200, "ask": 98.250, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01},
    "USDJPY": {"point": 0.001, "digits": 3, "bid": 150.00, "ask": 150.05, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01},
    "EURUSD": {"point": 0.0001, "digits": 5, "bid": 1.0800, "ask": 1.0805, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01},
    "XAUUSD": {"point": 0.01, "digits": 2, "bid": 2000.00, "ask": 2000.50, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01},
    "XAGUSD": {"point": 0.001, "digits": 3, "bid": 23.00, "ask": 23.02, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01},
    "XCUUSD": {"point": 0.0001, "digits": 4, "bid": 4.0000, "ask": 4.0010, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01},
    "XPBUSD": {"point": 0.01, "digits": 2, "bid": 950.00, "ask": 950.25, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01},
}

mock_open_positions = []
next_ticket = 1000000
mock_account_state = {"balance": 10_000.0, "equity": 10_000.0, "margin": 0.0, "margin_free": 10_000.0}
_historical_buffer: Dict[str, object] = {}
_current_step = 0


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").upper()


def _ensure_symbol(symbol: str) -> dict:
    normalized = _normalize_symbol(symbol)
    data = mock_symbol_data.setdefault(
        normalized,
        {"point": 0.0001, "digits": 5, "bid": 1.0, "ask": 1.0002, "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01},
    )
    return data


def _to_timestamp_seconds(value) -> int:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp())
    try:
        return int(datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp())
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(datetime.now(tz=timezone.utc).timestamp())


def _history_row_for_symbol(symbol: str):
    df = _historical_buffer.get(_normalize_symbol(symbol))
    if df is None or len(df) == 0:
        return None
    idx = min(max(_current_step, 0), len(df) - 1)
    return df.iloc[idx]


# --- Historical Playback Controls ---
def inject_historical_data(symbol, df):
    normalized = _normalize_symbol(symbol)
    local_df = df.reset_index(drop=True).copy()

    if 'time' in local_df.columns:
        local_df['time'] = pd.to_datetime(local_df['time'], utc=True, errors='coerce')
    elif 'timestamp' in local_df.columns:
        local_df['timestamp'] = pd.to_datetime(local_df['timestamp'], utc=True, errors='coerce')

    _historical_buffer[normalized] = local_df
    row = _history_row_for_symbol(normalized)
    if row is not None:
        _apply_history_row(normalized, row)


def set_current_step(index: int):
    global _current_step
    _current_step = max(int(index), 0)
    for symbol in list(_historical_buffer):
        row = _history_row_for_symbol(symbol)
        if row is not None:
            _apply_history_row(symbol, row)


def reset_simulation():
    global _current_step, mock_open_positions, next_ticket
    _current_step = 0
    mock_open_positions = []
    next_ticket = 1000000


def set_account_equity(equity: float, balance: Optional[float] = None):
    equity = float(equity)
    mock_account_state["equity"] = equity
    mock_account_state["balance"] = float(balance) if balance is not None else equity
    mock_account_state["margin_free"] = mock_account_state["equity"] - mock_account_state["margin"]




def set_simulation_step(index: int):
    """Compat helper for backtest runner naming."""
    set_current_step(index)


def get_current_sim_time(symbol: Optional[str] = None):
    if symbol:
        row = _history_row_for_symbol(symbol)
        if row is not None:
            return row.get('time', row.get('timestamp'))

    for sym in _historical_buffer:
        row = _history_row_for_symbol(sym)
        if row is not None:
            return row.get('time', row.get('timestamp'))
    return None


def eval(code: str):
    """Best-effort eval shim used by MT5Adapter when backend has no RPC eval."""
    local_ctx = {
        'mt5': MetaTrader5(),
        '__import__': __import__,
    }
    import builtins
    return builtins.eval(code, {'__builtins__': builtins.__dict__}, local_ctx)
def _apply_history_row(symbol: str, row) -> None:
    symbol_data = _ensure_symbol(symbol)
    close_price = float(row.get('close', row.get('Close', row.get('last', row.get('mid', row.get('price', symbol_data['bid']))))))
    bid = float(row.get('bid', row.get('Bid', close_price)))
    ask = float(row.get('ask', row.get('Ask', close_price)))
    if ask < bid:
        ask = bid
    symbol_data['bid'] = bid
    symbol_data['ask'] = ask
    symbol_data['last'] = close_price
    symbol_data['volume'] = float(row.get('tick_volume', row.get('volume', row.get('Volume', 0))))
    symbol_data['current_sim_time'] = row.get('timestamp', row.get('time', datetime.now(tz=timezone.utc).isoformat()))


# --- Mock MT5 Functions ---
def initialize(path=None, portable=False, profile=None, delay=0, timeout=1000, login=None, password=None, server=None) -> bool:
    global is_initialized
    is_initialized = True
    return True


def login(login, password, server) -> bool:
    global is_logged_in, logged_in_account
    if login and password and server:
        is_logged_in = True
        logged_in_account = login
        return True
    return False


def shutdown() -> None:
    global is_initialized, is_logged_in, logged_in_account
    is_initialized = False
    is_logged_in = False
    logged_in_account = None


def last_error() -> tuple:
    if not is_initialized:
        return (-10005, 'IPC timeout')
    return (0, 'No error')


def symbol_select(symbol: str, enable: bool = True) -> bool:
    normalized = _normalize_symbol(symbol)
    if enable:
        selected_symbols.add(normalized)
    else:
        selected_symbols.discard(normalized)
    _ensure_symbol(normalized)
    return True


def symbol_info(symbol: str) -> SymbolInfo:
    normalized = _normalize_symbol(symbol)
    data = mock_symbol_data.get(normalized)
    if data:
        return SymbolInfo(
            name=normalized,
            point=data['point'],
            digits=data['digits'],
            trade_tick_size=data.get('trade_tick_size', data['point']),
            trade_volume_min=data['volume_min'],
            trade_volume_max=data['volume_max'],
            trade_volume_step=data['volume_step'],
        )
    return None


def symbol_info_tick(symbol: str) -> SymbolInfoTick:
    normalized = _normalize_symbol(symbol)
    data = _ensure_symbol(normalized)
    row = _history_row_for_symbol(normalized)
    if row is not None:
        _apply_history_row(normalized, row)
        bid = float(data['bid'])
        ask = float(data['ask'])
        timestamp = _to_timestamp_seconds(data.get('current_sim_time'))
        volume = int(data.get('volume', 0) or 0)
    else:
        bid = data['bid'] + random.uniform(-data['point'], data['point'])
        ask = data['ask'] + random.uniform(-data['point'], data['point'])
        timestamp = int(datetime.now(tz=timezone.utc).timestamp())
        volume = random.randint(10, 100)
    return SymbolInfoTick(
        time=timestamp,
        bid=bid,
        ask=max(ask, bid),
        last=(bid + max(ask, bid)) / 2,
        volume=volume,
    )


def order_send(request: dict) -> TradeResult:
    global next_ticket, mock_open_positions

    if not is_logged_in:
        return TradeResult(TRADE_RETCODE_NO_CONNECTION, 0, 0, 0, 0, 'No connection', 0, 0, 0)

    action = request.get('action')

    if action == TRADE_ACTION_SLTP:
        position_ticket = request.get('position')
        updated_positions = []
        matched = None
        for pos in mock_open_positions:
            if pos.ticket == position_ticket:
                matched = pos._replace(sl=float(request.get('sl', pos.sl)), tp=float(request.get('tp', pos.tp)))
                updated_positions.append(matched)
            else:
                updated_positions.append(pos)
        mock_open_positions = updated_positions
        if matched is None:
            return TradeResult(TRADE_RETCODE_REJECT, 0, 0, 0, 0, 'Unknown position', 0, 0, 0)
        return TradeResult(TRADE_RETCODE_DONE, matched.ticket, matched.ticket, matched.volume, matched.price_open, 'Dummy Position Modified', random.randint(1, 10000), 0, matched.ticket)

    if not all(k in request for k in ['action', 'symbol', 'volume', 'type', 'price']):
        return TradeResult(TRADE_RETCODE_REJECT, 0, 0, 0, 0, 'Invalid request fields', 0, 0, 0)

    symbol = _normalize_symbol(request['symbol'])
    if symbol not in mock_symbol_data:
        return TradeResult(TRADE_RETCODE_REJECT, 0, 0, 0, 0, f'Unknown symbol {symbol}', 0, 0, 0)

    if action == TRADE_ACTION_DEAL:
        order_type = request['type']
        volume = float(request['volume'])
        price = float(request['price'])
        sl = float(request.get('sl', 0.0) or 0.0)
        tp = float(request.get('tp', 0.0) or 0.0)
        magic = request.get('magic', 0)
        comment = request.get('comment', '')

        if volume < mock_symbol_data[symbol]['volume_min']:
            return TradeResult(TRADE_RETCODE_REJECT, 0, 0, 0, 0, f'Volume {volume} too low for {symbol}, must be >= {mock_symbol_data[symbol]["volume_min"]}', 0, 0, 0)

        deal_price = price
        order_id = next_ticket
        next_ticket += 1

        if order_type in (ORDER_TYPE_BUY, ORDER_TYPE_SELL):
            mock_open_positions.append(Position(
                ticket=order_id,
                symbol=symbol,
                type=order_type,
                volume=volume,
                price_open=deal_price,
                time=int(datetime.now(tz=timezone.utc).timestamp()),
                sl=sl,
                tp=tp,
                magic=magic,
                comment=comment,
            ))

        return TradeResult(TRADE_RETCODE_DONE, order_id, order_id, volume, deal_price, 'Dummy Order Executed', random.randint(1, 10000), 0, order_id)

    return TradeResult(TRADE_RETCODE_REJECT, 0, 0, 0, 0, 'Unsupported action', 0, 0, 0)


def positions_get(ticket=None, symbol=None, magic=None) -> list:
    if not is_logged_in:
        return None

    normalized_symbol = _normalize_symbol(symbol) if symbol is not None else None
    filtered_positions = []
    for pos in mock_open_positions:
        match = True
        if ticket is not None and pos.ticket != ticket:
            match = False
        if normalized_symbol is not None and pos.symbol != normalized_symbol:
            match = False
        if magic is not None and pos.magic != magic:
            match = False
        if match:
            filtered_positions.append(pos)

    return filtered_positions


def account_info() -> AccountInfo:
    return AccountInfo(
        login=logged_in_account or 0,
        balance=float(mock_account_state['balance']),
        equity=float(mock_account_state['equity']),
        margin=float(mock_account_state['margin']),
        margin_free=float(mock_account_state['margin_free']),
    )


def terminal_info() -> TerminalInfo:
    return TerminalInfo(
        community_account=12345,
        community_connection=True,
        connected=is_logged_in,
        dll_s_allowed=True,
        trade_allowed=True,
        trade_expert=True,
        max_deviation=200,
        product_id='dummy',
        product_name='Dummy MT5',
        company='Mock Broker',
        language='EN',
        path='/dummy/path',
        max_bar_variation=0.0,
        max_calc_threads=0,
        cpu_core_count=4,
        ping_last=10,
        ping_prev=12,
    )


def copy_rates_range(symbol, timeframe, date_from, date_to):
    df = _historical_buffer.get(_normalize_symbol(symbol))
    if df is None or len(df) == 0:
        return np.array([], dtype=[])

    import pandas as pd

    local_df = df.copy()
    if 'timestamp' in local_df.columns:
        ts = pd.to_datetime(local_df['timestamp'], utc=True)
    elif 'time' in local_df.columns:
        ts = pd.to_datetime(local_df['time'], utc=True, errors='coerce')
    else:
        ts = pd.RangeIndex(start=0, stop=len(local_df), step=1)

    start_ts = pd.Timestamp(date_from)
    end_ts = pd.Timestamp(date_to)
    mask = (ts >= start_ts) & (ts <= end_ts)
    filtered = local_df.loc[mask].copy()
    if filtered.empty:
        return np.array([], dtype=[])

    if 'timestamp' in filtered.columns:
        filtered['time'] = pd.to_datetime(filtered['timestamp'], utc=True).astype('int64') // 10**9
    elif 'time' not in filtered.columns:
        filtered['time'] = np.arange(len(filtered), dtype=np.int64)

    for src, dst in [('open', 'open'), ('high', 'high'), ('low', 'low'), ('close', 'close')]:
        if dst not in filtered.columns:
            fallback = filtered.get(src.capitalize(), filtered.get('bid', filtered.get('ask', 0.0)))
            filtered[dst] = fallback
    if 'tick_volume' not in filtered.columns:
        filtered['tick_volume'] = filtered.get('volume', 0)
    if 'spread' not in filtered.columns:
        spread_points = ((filtered.get('ask', filtered['close']) - filtered.get('bid', filtered['close'])) / max(_ensure_symbol(symbol)['point'], 1e-12)).fillna(0)
        filtered['spread'] = spread_points.astype(int)
    if 'real_volume' not in filtered.columns:
        filtered['real_volume'] = filtered.get('volume', 0)

    fields = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    filtered = filtered[fields]
    records = filtered.to_records(index=False)
    return np.array(records, dtype=records.dtype)


class MetaTrader5:
    def __getattr__(self, item):
        attr = globals().get(item)
        if attr is None:
            raise AttributeError(item)
        return attr


print("[DUMMY MT5] Mock MetaTrader5 module loaded.")
