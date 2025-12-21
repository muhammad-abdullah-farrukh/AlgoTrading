// Dummy data generators for trading bot frontend

export interface OHLCVData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Trade {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  timestamp: Date;
  profitLoss: number;
  status: 'completed' | 'pending' | 'cancelled';
}

export interface Strategy {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  type: 'technical' | 'ai' | 'custom';
  parameters: Record<string, number | string>;
  performance: number;
  trades: number;
}

// Generate OHLCV data for charts
export const generateOHLCVData = (days: number = 30, basePrice: number = 1.0850): OHLCVData[] => {
  const data: OHLCVData[] = [];
  let currentPrice = basePrice;
  const now = Date.now();
  const msPerDay = 24 * 60 * 60 * 1000;

  for (let i = days; i >= 0; i--) {
    const volatility = 0.002;
    const change = (Math.random() - 0.5) * volatility;
    const open = currentPrice;
    const close = currentPrice * (1 + change);
    const high = Math.max(open, close) * (1 + Math.random() * 0.001);
    const low = Math.min(open, close) * (1 - Math.random() * 0.001);
    const volume = Math.floor(Math.random() * 10000) + 5000;

    data.push({
      time: now - (i * msPerDay),
      open: Number(open.toFixed(5)),
      high: Number(high.toFixed(5)),
      low: Number(low.toFixed(5)),
      close: Number(close.toFixed(5)),
      volume
    });

    currentPrice = close;
  }

  return data;
};

// Generate sample trades
export const generateTrades = (count: number = 20): Trade[] => {
  const symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'];
  const trades: Trade[] = [];

  for (let i = 0; i < count; i++) {
    const action = Math.random() > 0.5 ? 'BUY' : 'SELL';
    const profitLoss = (Math.random() - 0.4) * 500;

    trades.push({
      id: `TRD-${String(i + 1).padStart(6, '0')}`,
      symbol: symbols[Math.floor(Math.random() * symbols.length)],
      action,
      quantity: Math.floor(Math.random() * 10) * 0.1 + 0.1,
      price: 1.0850 + (Math.random() - 0.5) * 0.05,
      timestamp: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000),
      profitLoss: Number(profitLoss.toFixed(2)),
      status: Math.random() > 0.1 ? 'completed' : 'pending'
    });
  }

  return trades.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
};

// Generate sample strategies
export const generateStrategies = (): Strategy[] => [
  {
    id: 'strat-001',
    name: 'RSI Reversal',
    description: 'Buy when RSI drops below 30, sell when above 70',
    enabled: true,
    type: 'technical',
    parameters: { oversold: 30, overbought: 70, period: 14 },
    performance: 12.5,
    trades: 45
  },
  {
    id: 'strat-002',
    name: 'EMA Crossover',
    description: 'Trade on 9/21 EMA crossover signals',
    enabled: true,
    type: 'technical',
    parameters: { fastPeriod: 9, slowPeriod: 21 },
    performance: 8.3,
    trades: 32
  },
  {
    id: 'strat-003',
    name: 'MACD Momentum',
    description: 'Follow MACD histogram for trend direction',
    enabled: false,
    type: 'technical',
    parameters: { fast: 12, slow: 26, signal: 9 },
    performance: -2.1,
    trades: 28
  },
  {
    id: 'strat-004',
    name: 'AI Pattern Recognition',
    description: 'ML-based candlestick pattern detection',
    enabled: true,
    type: 'ai',
    parameters: { confidence: 0.85, lookback: 50 },
    performance: 15.8,
    trades: 18
  },
  {
    id: 'strat-005',
    name: 'Sentiment Analysis',
    description: 'Trade based on market sentiment indicators',
    enabled: false,
    type: 'ai',
    parameters: { threshold: 0.6 },
    performance: 5.2,
    trades: 12
  }
];

// Calculate RSI
export const calculateRSI = (data: OHLCVData[], period: number = 14): number[] => {
  const rsi: number[] = [];
  const changes: number[] = [];

  for (let i = 1; i < data.length; i++) {
    changes.push(data[i].close - data[i - 1].close);
  }

  for (let i = period; i < changes.length; i++) {
    const slice = changes.slice(i - period, i);
    const gains = slice.filter(c => c > 0).reduce((a, b) => a + b, 0) / period;
    const losses = Math.abs(slice.filter(c => c < 0).reduce((a, b) => a + b, 0)) / period;
    const rs = gains / (losses || 0.001);
    rsi.push(100 - (100 / (1 + rs)));
  }

  return rsi;
};

// Calculate EMA
export const calculateEMA = (data: OHLCVData[], period: number): number[] => {
  const ema: number[] = [];
  const multiplier = 2 / (period + 1);

  // Start with SMA
  let sum = 0;
  for (let i = 0; i < period; i++) {
    sum += data[i].close;
  }
  ema.push(sum / period);

  for (let i = period; i < data.length; i++) {
    ema.push((data[i].close - ema[ema.length - 1]) * multiplier + ema[ema.length - 1]);
  }

  return ema;
};

// Calculate MACD
export const calculateMACD = (data: OHLCVData[]): { macd: number[], signal: number[], histogram: number[] } => {
  const ema12 = calculateEMA(data, 12);
  const ema26 = calculateEMA(data, 26);

  const macd: number[] = [];
  const startIndex = Math.max(0, ema26.length - ema12.length);

  for (let i = startIndex; i < ema12.length; i++) {
    macd.push(ema12[i] - ema26[i - startIndex]);
  }

  // Signal line (9-period EMA of MACD)
  const signal: number[] = [];
  const multiplier = 2 / 10;
  signal.push(macd.slice(0, 9).reduce((a, b) => a + b, 0) / 9);

  for (let i = 9; i < macd.length; i++) {
    signal.push((macd[i] - signal[signal.length - 1]) * multiplier + signal[signal.length - 1]);
  }

  const histogram = macd.slice(9).map((m, i) => m - signal[i]);

  return { macd: macd.slice(9), signal, histogram };
};
