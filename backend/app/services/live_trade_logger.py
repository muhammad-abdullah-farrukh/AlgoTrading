from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LiveTradeEvent:
    timestamp: str
    symbol: str
    side: str
    quantity: float
    price: float
    order_id: Optional[str] = None
    comment: Optional[str] = None
    source: str = "mt5"
    timeframe: Optional[str] = None
    confidence: Optional[float] = None


class LiveTradeLogger:
    def __init__(self) -> None:
        self._events: List[LiveTradeEvent] = []

    def record_trade(self, event: LiveTradeEvent) -> None:
        self._events.append(event)

    def count(self) -> int:
        return len(self._events)

    def flush_to_datasets(self) -> Tuple[int, Optional[str], Optional[str]]:
        try:
            if not self._events:
                return 0, None, None

            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent
            datasets_dir = (project_root / "Datasets").resolve()
            datasets_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_path = datasets_dir / f"mt5_live_trades_{ts}.csv"

            rows = []
            for e in self._events:
                rows.append({
                    "date": e.timestamp,
                    "currency_pair": e.symbol,
                    "close_price": e.price,
                    "side": e.side,
                    "quantity": e.quantity,
                    "order_id": e.order_id,
                    "comment": e.comment,
                    "source": e.source,
                    "timeframe": e.timeframe,
                    "confidence": e.confidence,
                })

            df = pd.DataFrame(rows)
            df.to_csv(out_path, index=False)

            count = len(self._events)
            self._events.clear()
            logger.info(f"[LiveTradeLogger] Flushed {count} trade(s) to dataset file: {out_path}")
            return count, str(out_path), None
        except Exception as e:
            err = f"Failed to flush live trades to dataset: {str(e)}"
            logger.error(err)
            return 0, None, err


live_trade_logger = LiveTradeLogger()
