"""Database models for the trading application."""
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class OHLCV(Base):
    """OHLCV (Open, High, Low, Close, Volume) data model."""
    __tablename__ = "ohlcv"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    source = Column(String(100), nullable=True, index=True)  # Source attribution (yahoo, alphavantage, ccxt, custom)
    created_at = Column(DateTime, server_default=func.now())
    
    def __repr__(self):
        return f"<OHLCV(symbol={self.symbol}, timestamp={self.timestamp}, close={self.close}, source={self.source})>"


class Trade(Base):
    """Trade execution model."""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    trade_type = Column(String(10), nullable=False)  # 'buy' or 'sell'
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    order_id = Column(String(100), nullable=True, index=True)
    status = Column(String(20), default="executed")  # 'executed', 'pending', 'cancelled'
    created_at = Column(DateTime, server_default=func.now())
    
    def __repr__(self):
        return f"<Trade(symbol={self.symbol}, type={self.trade_type}, quantity={self.quantity}, price={self.price})>"


class Position(Base):
    """Trading position model."""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    quantity = Column(Float, nullable=False)
    average_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    status = Column(String(20), default="open")  # 'open', 'closed'
    opened_at = Column(DateTime, nullable=False, index=True)
    closed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<Position(symbol={self.symbol}, quantity={self.quantity}, status={self.status})>"


class DatasetMetadata(Base):
    """Dataset metadata model for storing information about datasets."""
    __tablename__ = "dataset_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    source = Column(String(200), nullable=True)
    symbol = Column(String(50), nullable=True, index=True)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    row_count = Column(Integer, default=0)
    file_path = Column(String(500), nullable=True)
    metadata_json = Column(Text, nullable=True)  # JSON string for additional metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<DatasetMetadata(name={self.name}, symbol={self.symbol}, row_count={self.row_count})>"


class Strategy(Base):
    """Trading strategy model."""
    __tablename__ = "strategies"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    strategy_type = Column(String(50), nullable=False)  # 'technical', 'ai', 'custom'
    enabled = Column(Boolean, default=False, nullable=False)
    parameters = Column(Text, nullable=True)  # JSON string for strategy parameters
    performance = Column(Float, default=0.0, nullable=False)  # Performance metric
    trades_count = Column(Integer, default=0, nullable=False)  # Number of trades executed
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<Strategy(name={self.name}, type={self.strategy_type}, enabled={self.enabled})>"


class MT5ConnectionStatus(Base):
    """MT5 connection status persistence."""
    __tablename__ = "mt5_connection_status"
    
    id = Column(Integer, primary_key=True, index=True)
    connected = Column(Boolean, default=False, nullable=False)  # Connection status
    mode = Column(String(20), nullable=False, default="mock")  # 'real' or 'mock'
    login = Column(Integer, nullable=True)  # MT5 login ID if connected
    server = Column(String(200), nullable=True)  # MT5 server name
    last_connected_at = Column(DateTime, nullable=True)  # Last successful connection time
    last_disconnected_at = Column(DateTime, nullable=True)  # Last disconnection time
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<MT5ConnectionStatus(connected={self.connected}, mode={self.mode})>"


class AutotradingSettings(Base):
    """Autotrading settings and controls."""
    __tablename__ = "autotrading_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    enabled = Column(Boolean, default=False, nullable=False)  # Enable/disable autotrading
    emergency_stop = Column(Boolean, default=False, nullable=False)  # Emergency stop flag
    stop_loss_percent = Column(Float, nullable=True)  # Stop loss as percentage (e.g., 2.0 for 2%)
    take_profit_percent = Column(Float, nullable=True)  # Take profit as percentage (e.g., 5.0 for 5%)
    max_daily_loss = Column(Float, nullable=True)  # Maximum daily loss amount
    position_size = Column(Float, nullable=True)  # Default position size
    auto_mode = Column(Boolean, default=False, nullable=False)  # Auto vs manual mode
    selected_strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=True)  # Selected strategy
    daily_loss_reset_date = Column(DateTime, nullable=True)  # Date when daily loss was last reset
    daily_loss_amount = Column(Float, default=0.0, nullable=False)  # Current daily loss amount
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<AutotradingSettings(enabled={self.enabled}, emergency_stop={self.emergency_stop}, auto_mode={self.auto_mode})>"
