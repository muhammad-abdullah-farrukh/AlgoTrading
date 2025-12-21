export type OhlcvPoint = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export function calculateEMA(data: OhlcvPoint[], period: number): Array<number | null> {
  if (period <= 0) return data.map(() => null);
  if (data.length < period) return data.map(() => null);

  const closes = data.map((d) => d.close);
  const multiplier = 2 / (period + 1);
  const out: Array<number | null> = Array(data.length).fill(null);

  // SMA seed
  let sma = 0;
  for (let i = 0; i < period; i++) sma += closes[i];
  sma /= period;
  out[period - 1] = sma;

  let prev = sma;
  for (let i = period; i < closes.length; i++) {
    const ema = (closes[i] - prev) * multiplier + prev;
    out[i] = ema;
    prev = ema;
  }

  return out;
}

export function calculateRSI(data: OhlcvPoint[], period: number = 14): Array<number | null> {
  if (period <= 0) return data.map(() => null);
  if (data.length < period + 1) return data.map(() => null);

  const closes = data.map((d) => d.close);
  const out: Array<number | null> = Array(data.length).fill(null);

  let gain = 0;
  let loss = 0;
  for (let i = 1; i <= period; i++) {
    const delta = closes[i] - closes[i - 1];
    if (delta >= 0) gain += delta;
    else loss += -delta;
  }

  let avgGain = gain / period;
  let avgLoss = loss / period;

  const rs = avgLoss === 0 ? 0 : avgGain / avgLoss;
  out[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + rs);

  for (let i = period + 1; i < closes.length; i++) {
    const delta = closes[i] - closes[i - 1];
    const g = delta > 0 ? delta : 0;
    const l = delta < 0 ? -delta : 0;

    avgGain = (avgGain * (period - 1) + g) / period;
    avgLoss = (avgLoss * (period - 1) + l) / period;

    if (avgLoss === 0) {
      out[i] = 100;
    } else {
      const rs2 = avgGain / avgLoss;
      out[i] = 100 - 100 / (1 + rs2);
    }
  }

  return out;
}

export function calculateMACD(
  data: OhlcvPoint[],
  fastPeriod: number = 12,
  slowPeriod: number = 26,
  signalPeriod: number = 9
): {
  macdLine: Array<number | null>;
  signalLine: Array<number | null>;
  histogram: Array<number | null>;
} {
  const fast = calculateEMA(data, fastPeriod);
  const slow = calculateEMA(data, slowPeriod);

  const macdLine: Array<number | null> = data.map((_, i) => {
    const f = fast[i];
    const s = slow[i];
    return f === null || s === null ? null : f - s;
  });

  // Build pseudo-series to reuse EMA: map null to 0 but mask output back to null
  const macdAsPrice: OhlcvPoint[] = data.map((d, i) => ({
    ...d,
    close: macdLine[i] ?? 0,
  }));

  const signalRaw = calculateEMA(macdAsPrice, signalPeriod);
  const signalLine: Array<number | null> = signalRaw.map((v, i) => (macdLine[i] === null ? null : v));

  const histogram: Array<number | null> = data.map((_, i) => {
    const m = macdLine[i];
    const s = signalLine[i];
    return m === null || s === null ? null : m - s;
  });

  return { macdLine, signalLine, histogram };
}
