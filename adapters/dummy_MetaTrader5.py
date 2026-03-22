import collections
from datetime import datetime
import random
import numpy as np

# --- Mock MT5 Constants ---
# Define constants that trade_engine.py might use
TRADE_ACTION_DEAL = 0
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
ORDER_TIME_GTC = 0  # Good Till Cancel
ORDER_FILLING_FOC = 0 # Fill Or Kill
ORDER_FILLING_RETURN = 1 # Return

TRADE_RETCODE_DONE = 10009 # Success code
TRADE_RETCODE_REJECT = 10013 # Rejected
TRADE_RETCODE_NO_CONNECTION = 10001

# --- Mock MT5 Data Structures ---
# Simulate the named tuples returned by MT5 functions
TradeResult = collections.namedtuple('TradeResult', ['retcode', 'deal', 'order', 'volume', 'price', 'comment', 'request_id', 'retcode_external', 'position'])
SymbolInfo = collections.namedtuple('SymbolInfo', ['name', 'point', 'digits', 'trade_tick_size', 'trade_volume_min', 'trade_volume_max', 'trade_volume_step'])
SymbolInfoTick = collections.namedtuple('SymbolInfoTick', ['time', 'bid', 'ask', 'last', 'volume'])
Position = collections.namedtuple('Position', ['ticket', 'symbol', 'type', 'volume', 'price_open', 'time', 'sl', 'tp', 'magic', 'comment'])
TerminalInfo = collections.namedtuple('TerminalInfo', ['community_account', 'community_connection', 'connected', 'dll_s_allowed', 'trade_allowed', 'trade_expert', 'max_deviation', 'product_id', 'product_name', 'company', 'language', 'path', 'max_bar_variation', 'max_calc_threads', 'cpu_core_count', 'ping_last', 'ping_prev'])

# --- Mock MT5 State Variables ---
is_initialized = False
is_logged_in = False
logged_in_account = None

# --- Mock Symbol Data ---
mock_symbol_data = {
    "USDCAD": {"point": 0.0001, "digits": 5, "bid": 1.3750, "ask": 1.3755, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01},
    "USDDXY": {"point": 0.001, "digits": 3, "bid": 98.200, "ask": 98.250, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01}, # NEW: Added USDDXY
    "USDJPY": {"point": 0.001, "digits": 3, "bid": 150.00, "ask": 150.05, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01},
    "EURUSD": {"point": 0.0001, "digits": 5, "bid": 1.0800, "ask": 1.0805, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01},
    "XAUUSD": {"point": 0.01, "digits": 2, "bid": 2000.00, "ask": 2000.50, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01},
    "XAGUSD": {"point": 0.001, "digits": 3, "bid": 23.00, "ask": 23.02, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01},
    "XCUUSD": {"point": 0.0001, "digits": 4, "bid": 4.0000, "ask": 4.0010, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01},
    "XPBUSD": {"point": 0.01, "digits": 2, "bid": 950.00, "ask": 950.25, "volume_min": 0.05, "volume_max": 100.0, "volume_step": 0.01}
}

mock_open_positions = []
next_ticket = 1000000

# --- Mock MT5 Functions ---
def initialize(path=None, portable=False, profile=None, delay=0, timeout=1000, login=None, password=None, server=None) -> bool:
    global is_initialized
    print("[DUMMY MT5] Initializing...")
    is_initialized = True
    return True

def login(login, password, server) -> bool:
    global is_logged_in, logged_in_account
    print(f"[DUMMY MT5] Logging in with account {login}...")
    if login and password and server:
        is_logged_in = True
        logged_in_account = login
        return True
    return False

def shutdown() -> None:
    global is_initialized, is_logged_in, logged_in_account
    print("[DUMMY MT5] Shutting down...")
    is_initialized = False
    is_logged_in = False
    logged_in_account = None

def last_error() -> tuple:
    if not is_initialized:
        return (-10005, 'IPC timeout')
    return (0, 'No error')

def symbol_info(symbol: str) -> SymbolInfo:
    data = mock_symbol_data.get(symbol.upper())
    if data:
        return SymbolInfo(
            name=symbol,
            point=data["point"],
            digits=data["digits"],
            trade_tick_size=data.get("trade_tick_size", data["point"]),
            trade_volume_min=data["volume_min"],
            trade_volume_max=data["volume_max"],
            trade_volume_step=data["volume_step"]
        )
    return None

def symbol_info_tick(symbol: str) -> SymbolInfoTick:
    data = mock_symbol_data.get(symbol.upper())
    if data:
        # Simulate slight price movement
        bid = data["bid"] + random.uniform(-0.0001, 0.0001)
        ask = data["ask"] + random.uniform(-0.0001, 0.0001)
        return SymbolInfoTick(
            time=int(datetime.now().timestamp()),
            bid=bid,
            ask=ask,
            last=(bid + ask) / 2,
            volume=random.randint(10, 100)
        )
    return None

def order_send(request: dict) -> TradeResult:
    global next_ticket, mock_open_positions
    print(f"[DUMMY MT5] Order send request: {request}")

    if not is_logged_in:
        return TradeResult(TRADE_RETCODE_NO_CONNECTION, 0, 0, 0, 0, 'No connection', 0, 0, 0)

    # Basic validation for essential fields
    if not all(k in request for k in ['action', 'symbol', 'volume', 'type', 'price']):
        return TradeResult(TRADE_RETCODE_REJECT, 0, 0, 0, 0, 'Invalid request fields', 0, 0, 0)

    symbol = request['symbol'].upper()
    if symbol not in mock_symbol_data:
        return TradeResult(TRADE_RETCODE_REJECT, 0, 0, 0, 0, f'Unknown symbol {symbol}', 0, 0, 0)

    if request['action'] == TRADE_ACTION_DEAL:
        order_type = request['type']
        volume = request['volume']
        price = request['price']
        sl = request.get('sl', 0.0)
        tp = request.get('tp', 0.0)
        magic = request.get('magic', 0)
        comment = request.get('comment', '')

        # --- Simulate failure condition (volume too low) ---
        if volume < mock_symbol_data[symbol]["volume_min"]:
            return TradeResult(TRADE_RETCODE_REJECT, 0, 0, 0, 0, f'Volume {volume} too low for {symbol}, must be >= {mock_symbol_data[symbol]["volume_min"]}', 0, 0, 0)

        # Simulate filling the order
        deal_price = price # For simplicity, assume requested price is filled
        order_id = next_ticket
        next_ticket += 1

        # Add to mock open positions
        if order_type == ORDER_TYPE_BUY or order_type == ORDER_TYPE_SELL:
            mock_open_positions.append(Position(
                ticket=order_id,
                symbol=symbol,
                type=order_type,
                volume=volume,
                price_open=deal_price,
                time=int(datetime.now().timestamp()),
                sl=sl,
                tp=tp,
                magic=magic,
                comment=comment
            ))

        return TradeResult(TRADE_RETCODE_DONE, order_id, order_id, volume, deal_price, 'Dummy Order Executed', random.randint(1,10000), 0, order_id)
    else:
        return TradeResult(TRADE_RETCODE_REJECT, 0, 0, 0, 0, 'Unsupported action', 0, 0, 0)

def positions_get(ticket=None, symbol=None, magic=None) -> list:
    if not is_logged_in:
        return None # Mimic MT5 behavior for no connection

    filtered_positions = []
    for pos in mock_open_positions:
        match = True
        if ticket is not None and pos.ticket != ticket: match = False
        if symbol is not None and pos.symbol != symbol.upper(): match = False
        if magic is not None and pos.magic != magic: match = False
        if match: filtered_positions.append(pos)

    return filtered_positions

def terminal_info() -> TerminalInfo:
    return TerminalInfo(
        community_account=12345,
        community_connection=True,
        connected=is_logged_in,
        dll_s_allowed=True,
        trade_allowed=True,
        trade_expert=True,
        max_deviation=200, # Example max deviation in points
        product_id='dummy',
        product_name='Dummy MT5',
        company='Mock Broker',
        language='EN',
        path='/dummy/path',
        max_bar_variation=0.0,
        max_calc_threads=0,
        cpu_core_count=4,
        ping_last=10,
        ping_prev=12
    )

print("[DUMMY MT5] Mock MetaTrader5 module loaded.")
