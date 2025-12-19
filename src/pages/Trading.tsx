import { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogDescription } from '@/components/ui/dialog';
import { Switch } from '@/components/ui/switch';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { generateOHLCVData, calculateRSI, calculateEMA, calculateMACD } from '@/utils/dummyData';
import { useTrades } from '@/contexts/TradesContext';
import { useWebSocket } from '@/hooks/useWebSocket';
import { toast } from 'sonner';
import api from '@/utils/api';
import { 
  TrendingUp, TrendingDown, Minus, Brain, 
  CandlestickChart, BarChart3, LineChart, Activity,
  ShoppingCart, Wallet, AlertTriangle, ChevronDown,
  Settings2, Layers
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { ResponsiveContainer, ComposedChart, Line, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, ReferenceLine, Cell, Customized } from 'recharts';

// Time intervals grouped by category
const TIME_INTERVALS = {
  minutes: ['1m', '3m', '5m', '15m', '30m', '45m'],
  hours: ['1h', '2h', '3h', '4h'],
  days: ['1d', '1w', '1M', '3M', '6M', '12M'], // M = Month
};

// Flattened list for backward compatibility
const TIMEFRAMES = [...TIME_INTERVALS.minutes, ...TIME_INTERVALS.hours, ...TIME_INTERVALS.days];

const CHART_TYPES = [
  { id: 'candles', label: 'Candles', icon: CandlestickChart, implemented: true },
  { id: 'bars', label: 'Bars', icon: BarChart3, implemented: true },
  { id: 'line', label: 'Line', icon: LineChart, implemented: true },
  { id: 'heikin-ashi', label: 'Heikin Ashi', icon: CandlestickChart, implemented: true },
  { id: 'renko', label: 'Renko', icon: BarChart3, implemented: false },
  { id: 'line-break', label: 'Line Break', icon: LineChart, implemented: false },
  { id: 'kagi', label: 'Kagi', icon: Activity, implemented: false },
  { id: 'point-figure', label: 'P&F', icon: BarChart3, implemented: false },
  { id: 'range', label: 'Range', icon: BarChart3, implemented: false },
];

type ChartType =
  | 'candles'
  | 'bars'
  | 'line'
  | 'heikin-ashi'
  | 'renko'
  | 'line-break'
  | 'kagi'
  | 'point-figure'
  | 'range';

const INDICATORS = [
  { id: 'rsi', label: 'RSI', description: 'Relative Strength Index', implemented: true },
  { id: 'ema', label: 'EMA', description: 'Exponential Moving Average', implemented: true },
  { id: 'macd', label: 'MACD', description: 'Moving Average Convergence Divergence', implemented: true },
  { id: 'volume', label: 'Volume', description: 'Trading Volume', implemented: true },
  { id: 'custom', label: 'Custom', description: 'Custom Indicator Placeholder', implemented: true },
];

const SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSDT', 'AAPL', 'AUDUSD'];

const OPEN_POSITIONS = [
  { id: 1, symbol: 'EURUSD', side: 'BUY', entry: 1.0820, current: 1.0855, pnl: 35.00, status: 'Open' },
  { id: 2, symbol: 'BTCUSDT', side: 'SELL', entry: 43500, current: 43200, pnl: 120.00, status: 'Open' },
  { id: 3, symbol: 'GBPUSD', side: 'BUY', entry: 1.2650, current: 1.2630, pnl: -20.00, status: 'Open' },
];

const Trading = () => {
  const { addTrade } = useTrades();
  const [selectedSymbol, setSelectedSymbol] = useState('EURUSD');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1d');
  const [timeIntervalOpen, setTimeIntervalOpen] = useState(false);
  const [chartType, setChartType] = useState<ChartType>('candles');
  const [indicatorState, setIndicatorState] = useState<Record<'ema' | 'rsi' | 'macd' | 'volume' | 'custom', boolean>>({
    ema: false,
    rsi: false,
    macd: false,
    volume: true,
    custom: false,
  });
  const [ohlcvData, setOhlcvData] = useState(() => generateOHLCVData(60));
  const [currentPrice, setCurrentPrice] = useState(1.0850);
  const [tradeQuantity, setTradeQuantity] = useState('0.1');
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const [pendingTrade, setPendingTrade] = useState<{ type: 'BUY' | 'SELL' } | null>(null);
  const [indicatorsOpen, setIndicatorsOpen] = useState(false);
  const [chartTypeOpen, setChartTypeOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [positions, setPositions] = useState<Array<{
    id: number;
    symbol: string;
    quantity: number;
    average_price: number;
    current_price: number | null;
    unrealized_pnl: number;
    realized_pnl: number;
    status: string;
    opened_at: string;
    closed_at: string | null;
  }>>([]);
  const [isLoadingPositions, setIsLoadingPositions] = useState(true);
  const [isTrading, setIsTrading] = useState(false);
  const [aiSignals, setAiSignals] = useState<any[]>([]);
  const [isLoadingSignals, setIsLoadingSignals] = useState(false);

  // Fetch positions from backend on mount
  useEffect(() => {
    const fetchPositions = async () => {
      try {
        setIsLoadingPositions(true);
        const response = await api.get<{
          positions: Array<{
            id: number;
            symbol: string;
            quantity: number;
            average_price: number;
            current_price: number | null;
            unrealized_pnl: number;
            realized_pnl: number;
            status: string;
            opened_at: string;
            closed_at: string | null;
          }>;
          count: number;
        }>('/api/trade/positions', {
          params: { status: 'open' }
        });
        setPositions(response.positions || []);
      } catch (error) {
        console.error('[Trading] Failed to fetch positions:', error);
        // Don't show error toast - positions might be empty
      } finally {
        setIsLoadingPositions(false);
      }
    };
    
    fetchPositions();
  }, []);

  // Fetch AI signals with caching for instant display
  useEffect(() => {
    const fetchAiSignals = async () => {
      const startTime = performance.now();
      try {
        // Only show loading if we have no signals (first load)
        if (aiSignals.length === 0) {
          setIsLoadingSignals(true);
        }
        
        const response = await api.get<{
          signals: Array<{
            id: number;
            symbol: string;
            signal: string;
            confidence: number;
            reason: string;
            price?: number;
            timeframe: string;
            timestamp?: string;
          }>;
          count: number;
          timeframe: string;
          model_available: boolean;
          model_accuracy?: number;
        }>('/api/ml/signals', {
          params: { 
            timeframe: selectedTimeframe,
            min_confidence: 0.5
          }
        });
        
        const fetchTime = performance.now() - startTime;
        console.log(`[Trading] Signals fetched in ${fetchTime.toFixed(0)}ms (cached: ${fetchTime < 100 ? 'yes' : 'no'})`);
        
        setAiSignals(response.signals || []);
      } catch (error: any) {
        console.error('[Trading] Failed to fetch AI signals:', error);
        // Don't clear signals on error - keep showing cached ones
        if (error?.response?.status !== 404) {
          console.warn('[Trading] Error details:', error?.response?.data || error?.message);
        }
      } finally {
        setIsLoadingSignals(false);
      }
    };
    
    // Fetch immediately when timeframe changes (force refresh)
    fetchAiSignals();
    
    // Refresh signals every 30 seconds (silent - uses cache if available)
    const interval = setInterval(() => {
      fetchAiSignals(); // Will use cache if < 30s old
    }, 30000);
    
    return () => clearInterval(interval);
  }, [selectedTimeframe]); // Re-fetch when timeframe changes (removed aiSignals.length to avoid infinite loop)

  // WebSocket connection for positions updates
  const { isConnected: isPositionsConnected } = useWebSocket({
    url: '/ws/positions',
    onMessage: (data) => {
      if (data && typeof data === 'object' && 'type' in data) {
        const msg = data as { type: string; data?: unknown[] };
        if (msg.type === 'positions' && Array.isArray(msg.data)) {
          setPositions(msg.data as typeof positions);
        }
      }
    },
    onError: (error) => {
      console.error('[Trading] Positions WebSocket error:', error);
    },
    autoConnect: true,
    reconnectAttempts: 5,
  });

  // WebSocket connection for live tick data
  const { isConnected: isTickConnected, lastMessage: tickMessage } = useWebSocket({
    url: `/ws/ticks/${selectedSymbol}`,
    onMessage: (data) => {
      // Handle tick messages from backend
      if (data && typeof data === 'object' && 'type' in data) {
        const msg = data as { type: string; bid?: number; ask?: number; symbol?: string };
        if (msg.type === 'tick' && msg.bid !== undefined) {
          // Update current price with bid price (or mid price)
          const midPrice = msg.ask !== undefined 
            ? (msg.bid + msg.ask) / 2 
            : msg.bid;
          setCurrentPrice(Number(midPrice.toFixed(5)));
        }
      }
    },
    onError: (error) => {
      console.error('[Trading] WebSocket error:', error);
    },
    autoConnect: true,
    reconnectAttempts: 5,
  });

  // Fallback: Simulate live price updates ONLY if WebSocket not connected
  // This prevents duplicate updates when WebSocket is active
  useEffect(() => {
    if (isTickConnected) {
      // WebSocket is connected, don't use mock updates
      return;
    }

    // Only use mock updates when WebSocket is disconnected
    const interval = setInterval(() => {
      setCurrentPrice(prev => {
        const change = (Math.random() - 0.5) * 0.0010;
        return Number((prev + change).toFixed(5));
      });
      // Note: Positions are now updated via WebSocket, not mock updates
      // Mock price updates only affect currentPrice display
    }, 1000);
    return () => clearInterval(interval);
  }, [isTickConnected]);

  // Regenerate data when symbol/timeframe changes
  useEffect(() => {
    setOhlcvData(generateOHLCVData(60));
  }, [selectedSymbol, selectedTimeframe]);

  const toggleIndicator = (indicator: 'ema' | 'rsi' | 'macd' | 'volume' | 'custom') => {
    setIndicatorState(prev => ({ ...prev, [indicator]: !prev[indicator] }));
  };

  const buildEMA = (period: number) => {
    const emaVals = calculateEMA(ohlcvData, period);
    const series = Array(ohlcvData.length).fill(null) as (number | null)[];
    emaVals.forEach((v, idx) => {
      const target = period - 1 + idx;
      if (target < series.length) series[target] = Number(v.toFixed(5));
    });
    return series;
  };

  const buildRSI = (period = 14) => {
    const rsiVals = calculateRSI(ohlcvData, period);
    const series = Array(ohlcvData.length).fill(null) as (number | null)[];
    rsiVals.forEach((v, idx) => {
      const target = period + idx;
      if (target < series.length) series[target] = Number(v.toFixed(2));
    });
    return series;
  };

  const buildMACD = () => {
    const macd = calculateMACD(ohlcvData);
    const len = ohlcvData.length;
    const macdLine = Array(len).fill(null) as (number | null)[];
    const signal = Array(len).fill(null) as (number | null)[];
    const hist = Array(len).fill(null) as (number | null)[];
    macd.macd.forEach((v, idx) => (macdLine[len - macd.macd.length + idx] = Number(v.toFixed(5))));
    macd.signal.forEach((v, idx) => (signal[len - macd.signal.length + idx] = Number(v.toFixed(5))));
    macd.histogram.forEach((v, idx) => (hist[len - macd.histogram.length + idx] = Number(v.toFixed(5))));
    return { macdLine, signal, hist };
  };

  const computeHeikinAshi = (src: typeof ohlcvData) => {
    const res: typeof ohlcvData = [];
    src.forEach((d, idx) => {
      const close = (d.open + d.high + d.low + d.close) / 4;
      const open = idx === 0 ? (d.open + d.close) / 2 : (res[idx - 1].open + res[idx - 1].close) / 2;
      const high = Math.max(d.high, open, close);
      const low = Math.min(d.low, open, close);
      res.push({ time: d.time, open, high, low, close, volume: d.volume });
    });
    return res;
  };

  const { chartData, chartDataHeikin } = useMemo(() => {
    const rsi = buildRSI(14);
    const ema9 = buildEMA(9);
    const ema21 = buildEMA(21);
    const macd = buildMACD();

    const base = ohlcvData.map((d, i) => ({
      time: new Date(d.time).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      open: d.open,
      close: d.close,
      high: d.high,
      low: d.low,
      volume: d.volume,
      rsi: rsi[i] ?? null,
      ema9: ema9[i] ?? null,
      ema21: ema21[i] ?? null,
      macdLine: macd.macdLine[i] ?? null,
      macdSignal: macd.signal[i] ?? null,
      macdHist: macd.hist[i] ?? null,
      isUp: d.close >= d.open,
    }));

    const ha = computeHeikinAshi(ohlcvData).map((d, i) => ({
      ...base[i],
      open: d.open,
      close: d.close,
      high: d.high,
      low: d.low,
      volume: d.volume,
      isUp: d.close >= d.open,
    }));

    return { chartData: base, chartDataHeikin: ha };
  }, [ohlcvData]);

  const handleTrade = (type: 'BUY' | 'SELL') => {
    // Validate quantity
    const qty = parseFloat(tradeQuantity);
    if (isNaN(qty) || qty <= 0) {
      toast.error('Invalid quantity', {
        description: 'Please enter a valid lot size greater than 0',
      });
      return;
    }
    
    // Validate minimum lot size (0.01)
    if (qty < 0.01) {
      toast.error('Quantity too small', {
        description: 'Minimum lot size is 0.01',
      });
      return;
    }
    
    setPendingTrade({ type });
    setIsConfirmOpen(true);
  };

  const confirmTrade = async () => {
    if (!pendingTrade) return;
    
    const qty = parseFloat(tradeQuantity);
    const tradeType = pendingTrade.type.toLowerCase(); // 'buy' or 'sell'
    
    setIsTrading(true);
    
    try {
      // Call backend API to place order
      const response = await api.post<{
        status: string;
        message: string;
        position: any;
      }>(`/api/trade/${tradeType}`, {
        symbol: selectedSymbol,
        quantity: qty,
        price: currentPrice, // Optional - backend will use market price if not provided
      });
      
      // Add trade to shared context for UI
      addTrade({
        symbol: selectedSymbol,
        action: pendingTrade.type,
        quantity: qty,
        price: currentPrice,
      });
      
      // Refresh positions after trade
      try {
        const positionsResponse = await api.get<{
          positions: typeof positions;
          count: number;
        }>('/api/trade/positions', {
          params: { status: 'open' }
        });
        setPositions(positionsResponse.positions || []);
      } catch (posError) {
        console.error('[Trading] Failed to refresh positions:', posError);
        // Non-critical, continue
      }
      
      toast.success(`${pendingTrade.type} order executed`, {
        description: `${qty} lots of ${selectedSymbol} @ ${currentPrice.toFixed(5)}`,
      });
      
      console.log(`[Trading] ${pendingTrade.type} order executed: ${qty} ${selectedSymbol} @ ${currentPrice}`);
      
      setIsConfirmOpen(false);
      setPendingTrade(null);
    } catch (error) {
      console.error(`[Trading] Failed to execute ${pendingTrade.type} order:`, error);
      
      // Extract error message from APIError
      let errorMessage = 'Unknown error';
      if (error instanceof Error) {
        errorMessage = error.message;
        // If it's an APIError, try to get detail from response
        if ('response' in error && error.response && typeof error.response === 'object') {
          const apiResponse = error.response as { detail?: string; message?: string };
          errorMessage = apiResponse.detail || apiResponse.message || errorMessage;
        }
      }
      
      toast.error(`${pendingTrade.type} order failed`, {
        description: errorMessage,
      });
    } finally {
      setIsTrading(false);
    }
  };

  const priceChange = currentPrice - ohlcvData[0]?.close || 0;
  const priceChangePercent = ((priceChange / (ohlcvData[0]?.close || 1)) * 100);

  const selectedChartType = CHART_TYPES.find(c => c.id === chartType);
  const priceSeriesBase = useMemo(
    () => (chartType === 'heikin-ashi' ? computeHeikinAshi(ohlcvData) : ohlcvData),
    [chartType, ohlcvData]
  );

  // Professional palette
  const gridColor = '#1f2937';
  const axisColor = '#94a3b8';
  const bullish = '#16a34a';
  const bearish = '#dc2626';

  const CandleShapes = (props: any) => {
    const { xAxisMap, yAxisMap, data } = props;
    const yAxis = yAxisMap?.price || Object.values(yAxisMap)[0];
    const xAxis = Object.values(xAxisMap)[0] as any;
    const xScale = xAxis?.scale;
    const yScale = yAxis?.scale;
    if (!xScale || !yScale) return null;
    const positions = data.map((d: any) => xScale(d.time));
    const width = positions.length > 1 ? Math.max((positions[1] - positions[0]) * 0.6, 3) : 6;
    return (
      <g>
        {data.map((d: any, idx: number) => {
          const x = positions[idx];
          const color = d.isUp ? bullish : bearish;
          const openY = yScale(d.open);
          const closeY = yScale(d.close);
          const highY = yScale(d.high);
          const lowY = yScale(d.low);
          const bodyTop = Math.min(openY, closeY);
          const bodyHeight = Math.max(Math.abs(closeY - openY), 1.5);
          return (
            <g key={`candle-${idx}`}>
              <line x1={x} x2={x} y1={highY} y2={lowY} stroke={color} strokeWidth={1} />
              <rect
                x={x - width / 2}
                y={bodyTop}
                width={width}
                height={bodyHeight}
                fill={color}
                opacity={0.9}
                rx={1}
              />
            </g>
          );
        })}
      </g>
    );
  };

  const OHLCBars = (props: any) => {
    const { xAxisMap, yAxisMap, data } = props;
    const yAxis = yAxisMap?.price || Object.values(yAxisMap)[0];
    const xAxis = Object.values(xAxisMap)[0] as any;
    const xScale = xAxis?.scale;
    const yScale = yAxis?.scale;
    if (!xScale || !yScale) return null;
    const positions = data.map((d: any) => xScale(d.time));
    const tick = positions.length > 1 ? Math.max((positions[1] - positions[0]) * 0.25, 3) : 4;
    return (
      <g>
        {data.map((d: any, idx: number) => {
          const x = positions[idx];
          const color = d.isUp ? bullish : bearish;
          const highY = yScale(d.high);
          const lowY = yScale(d.low);
          const openY = yScale(d.open);
          const closeY = yScale(d.close);
          return (
            <g key={`ohlc-${idx}`}>
              <line x1={x} x2={x} y1={highY} y2={lowY} stroke={color} strokeWidth={1} />
              <line x1={x - tick} x2={x} y1={openY} y2={openY} stroke={color} strokeWidth={1.2} />
              <line x1={x} x2={x + tick} y1={closeY} y2={closeY} stroke={color} strokeWidth={1.2} />
            </g>
          );
        })}
      </g>
    );
  };

  const priceSeries = useMemo(() => {
    const ema9 = calculateEMA(priceSeriesBase, 9);
    const ema21 = calculateEMA(priceSeriesBase, 21);
    const rsi = calculateRSI(priceSeriesBase, 14);
    const macd = calculateMACD(priceSeriesBase);

    const len = priceSeriesBase.length;
    const macdLine: (number | null)[] = Array(len).fill(null);
    const macdSignal: (number | null)[] = Array(len).fill(null);
    const macdHist: (number | null)[] = Array(len).fill(null);
    macd.macd.forEach((v, idx) => (macdLine[len - macd.macd.length + idx] = Number(v.toFixed(5))));
    macd.signal.forEach((v, idx) => (macdSignal[len - macd.signal.length + idx] = Number(v.toFixed(5))));
    macd.histogram.forEach((v, idx) => (macdHist[len - macd.histogram.length + idx] = Number(v.toFixed(5))));

    return priceSeriesBase.map((d, i) => ({
      ...d,
      time: new Date(d.time).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      isUp: d.close >= d.open,
      ema9: ema9[i - 8] ?? null,
      ema21: ema21[i - 20] ?? null,
      rsi: rsi[i - 14] ?? null,
      macdLine: macdLine[i],
      macdSignal: macdSignal[i],
      macdHist: macdHist[i],
    }));
  }, [priceSeriesBase]);

  const rsiPanelData = useMemo(() => priceSeries.filter(d => d.rsi !== null), [priceSeries]);
  const macdPanelData = useMemo(
    () => priceSeries.filter(d => d.macdHist !== null && d.macdLine !== null && d.macdSignal !== null),
    [priceSeries]
  );

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Professional Chart Header - TradingView Style */}
      <div className="bg-card rounded-lg border border-border p-3 transition-all duration-200 hover:border-border/80">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          {/* Symbol & Price */}
          <div className="flex items-center gap-4">
            <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
              <SelectTrigger className="w-[130px] bg-secondary border-0 font-bold text-lg transition-all duration-200 hover:bg-secondary/80 hover:scale-[1.02]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SYMBOLS.map(s => (
                  <SelectItem 
                    key={s} 
                    value={s}
                    className="transition-colors duration-150 hover:bg-secondary"
                  >
                    {s}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="flex items-baseline gap-3">
              <span className="text-2xl font-bold tabular-nums text-foreground">
                {currentPrice.toFixed(5)}
              </span>
              <span className={cn(
                "text-sm font-semibold px-2 py-0.5 rounded",
                priceChange >= 0 ? "text-success bg-success/10" : "text-destructive bg-destructive/10"
              )}>
                {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(5)} ({priceChangePercent.toFixed(2)}%)
              </span>
            </div>
          </div>

          {/* Time Interval Selector - TradingView Style */}
          <DropdownMenu open={timeIntervalOpen} onOpenChange={setTimeIntervalOpen}>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className={cn(
                  "h-8 px-3 text-xs font-medium bg-secondary/50 border-border hover:bg-secondary transition-all duration-200 hover:scale-[1.02]",
                  timeIntervalOpen && "bg-secondary border-ai-purple/50"
                )}
              >
                <span className="font-semibold">{selectedTimeframe.toUpperCase()}</span>
                <ChevronDown className={cn(
                  "h-3 w-3 ml-2 transition-transform duration-200",
                  timeIntervalOpen && "rotate-180"
                )} />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent 
              align="end" 
              className="w-40 p-1 bg-popover border-border"
            >
              <DropdownMenuLabel className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
                Minutes
              </DropdownMenuLabel>
              {TIME_INTERVALS.minutes.map(interval => (
                <DropdownMenuItem
                  key={interval}
                  onClick={() => {
                    setSelectedTimeframe(interval);
                    setTimeIntervalOpen(false);
                  }}
                  className={cn(
                    "px-2 py-1.5 text-xs cursor-pointer transition-colors duration-150",
                    selectedTimeframe === interval 
                      ? "bg-ai-purple/20 text-ai-purple font-semibold" 
                      : "hover:bg-secondary"
                  )}
                >
                  {interval.toUpperCase()}
                </DropdownMenuItem>
              ))}
              <DropdownMenuSeparator className="my-1" />
              <DropdownMenuLabel className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
                Hours
              </DropdownMenuLabel>
              {TIME_INTERVALS.hours.map(interval => (
                <DropdownMenuItem
                  key={interval}
                  onClick={() => {
                    setSelectedTimeframe(interval);
                    setTimeIntervalOpen(false);
                  }}
                  className={cn(
                    "px-2 py-1.5 text-xs cursor-pointer transition-colors duration-150",
                    selectedTimeframe === interval 
                      ? "bg-ai-purple/20 text-ai-purple font-semibold" 
                      : "hover:bg-secondary"
                  )}
                >
                  {interval.toUpperCase()}
                </DropdownMenuItem>
              ))}
              <DropdownMenuSeparator className="my-1" />
              <DropdownMenuLabel className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
                Days
              </DropdownMenuLabel>
              {TIME_INTERVALS.days.map(interval => (
                <DropdownMenuItem
                  key={interval}
                  onClick={() => {
                    setSelectedTimeframe(interval);
                    setTimeIntervalOpen(false);
                  }}
                  className={cn(
                    "px-2 py-1.5 text-xs cursor-pointer transition-colors duration-150",
                    selectedTimeframe === interval 
                      ? "bg-ai-purple/20 text-ai-purple font-semibold" 
                      : "hover:bg-secondary"
                  )}
                >
                  {interval.toUpperCase()}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Chart Toolbar - Refined with Better Grouping */}
      <div className="bg-card rounded-lg border border-border p-2 transition-all duration-200 hover:border-border/80">
        <div className="flex flex-wrap items-center gap-2">
          {/* Chart Type Dropdown */}
          <Popover open={chartTypeOpen} onOpenChange={setChartTypeOpen}>
            <PopoverTrigger asChild>
              <Button 
                variant="ghost" 
                size="sm" 
                className={cn(
                  "gap-2 text-muted-foreground hover:text-foreground transition-all duration-200 hover:bg-secondary/80",
                  chartTypeOpen && "bg-secondary text-foreground"
                )}
              >
                {selectedChartType && <selectedChartType.icon className="h-4 w-4 transition-transform duration-200" />}
                <span className="hidden sm:inline">{selectedChartType?.label}</span>
                <ChevronDown className={cn(
                  "h-3 w-3 transition-transform duration-200",
                  chartTypeOpen && "rotate-180"
                )} />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-48 p-1" align="start">
                {CHART_TYPES.map(ct => (
                <Button
                  key={ct.id}
                  variant="ghost"
                  size="sm"
                    className={cn(
                      "w-full justify-start gap-2 transition-all duration-150",
                      chartType === ct.id 
                        ? "bg-ai-purple/20 text-ai-purple font-semibold hover:bg-ai-purple/30" 
                        : "hover:bg-secondary",
                      !ct.implemented && "opacity-60 cursor-not-allowed"
                    )}
                    onClick={() => {
                      if (!ct.implemented) return;
                      setChartType(ct.id as ChartType);
                      setChartTypeOpen(false);
                    }}
                    title={ct.implemented ? ct.label : "Not implemented yet"}
                    disabled={!ct.implemented}
                >
                  <ct.icon className="h-4 w-4" />
                  {ct.label}
                </Button>
              ))}
            </PopoverContent>
          </Popover>

          <div className="h-5 w-px bg-border" />

          {/* Indicators Dropdown */}
          <Popover open={indicatorsOpen} onOpenChange={setIndicatorsOpen}>
            <PopoverTrigger asChild>
              <Button 
                variant="ghost" 
                size="sm" 
                className={cn(
                  "gap-2 text-muted-foreground hover:text-foreground transition-all duration-200 hover:bg-secondary/80",
                  indicatorsOpen && "bg-secondary text-foreground"
                )}
              >
                <Layers className="h-4 w-4 transition-transform duration-200" />
                <span className="hidden sm:inline">Indicators</span>
                {Object.values(indicatorState).filter(Boolean).length > 0 && (
                  <Badge variant="secondary" className="h-5 px-1.5 text-xs bg-ai-purple/20 text-ai-purple animate-in fade-in zoom-in-95 duration-200">
                    {Object.values(indicatorState).filter(Boolean).length}
                  </Badge>
                )}
                <ChevronDown className={cn(
                  "h-3 w-3 transition-transform duration-200",
                  indicatorsOpen && "rotate-180"
                )} />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-64 p-2" align="start">
              <div className="space-y-1">
                <p className="text-xs font-medium text-muted-foreground px-2 pb-2">Technical Indicators</p>
                {INDICATORS.map(ind => (
                  <Button
                    key={ind.id}
                    variant={indicatorState[ind.id as keyof typeof indicatorState] ? "default" : "outline"}
                    size="sm"
                    className={cn(
                      "w-full justify-between text-xs transition-all duration-150",
                      !ind.implemented && "opacity-60 cursor-not-allowed"
                    )}
                    onClick={() => ind.implemented && toggleIndicator(ind.id as keyof typeof indicatorState)}
                    title={ind.implemented ? ind.description : "Not implemented yet"}
                    disabled={!ind.implemented}
                  >
                    <span className="flex flex-col items-start gap-0.5">
                      <span className="font-medium">{ind.label}</span>
                      <span className="text-[11px] text-muted-foreground">{ind.description}</span>
                    </span>
                    <span className={cn(
                      "h-2 w-2 rounded-full",
                      indicatorState[ind.id as keyof typeof indicatorState] ? "bg-ai-purple" : "bg-muted-foreground/50"
                    )} />
                  </Button>
                ))}
              </div>
            </PopoverContent>
          </Popover>

          <div className="h-5 w-px bg-border" />

          {/* Advanced Settings Dropdown */}
          <DropdownMenu open={settingsOpen} onOpenChange={setSettingsOpen}>
            <DropdownMenuTrigger asChild>
              <Button 
                variant="ghost" 
                size="sm" 
                className={cn(
                  "gap-2 text-muted-foreground hover:text-foreground transition-all duration-200 hover:bg-secondary/80",
                  settingsOpen && "bg-secondary text-foreground"
                )}
              >
                <Settings2 className={cn(
                  "h-4 w-4 transition-transform duration-200",
                  settingsOpen && "rotate-90"
                )} />
                <span className="hidden sm:inline">Settings</span>
                <ChevronDown className={cn(
                  "h-3 w-3 transition-transform duration-200",
                  settingsOpen && "rotate-180"
                )} />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56 p-1">
              <DropdownMenuLabel className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
                Chart Options
              </DropdownMenuLabel>
              <DropdownMenuItem className="px-2 py-1.5 text-sm cursor-pointer hover:bg-secondary transition-colors duration-150">
                Grid Lines
              </DropdownMenuItem>
              <DropdownMenuItem className="px-2 py-1.5 text-sm cursor-pointer hover:bg-secondary transition-colors duration-150">
                Crosshair
              </DropdownMenuItem>
              <DropdownMenuItem className="px-2 py-1.5 text-sm cursor-pointer hover:bg-secondary transition-colors duration-150">
                Price Scale
              </DropdownMenuItem>
              <DropdownMenuSeparator className="my-1" />
              <DropdownMenuLabel className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
                Display
              </DropdownMenuLabel>
              <DropdownMenuItem className="px-2 py-1.5 text-sm cursor-pointer hover:bg-secondary transition-colors duration-150">
                Theme Settings
              </DropdownMenuItem>
              <DropdownMenuItem className="px-2 py-1.5 text-sm cursor-pointer hover:bg-secondary transition-colors duration-150">
                Layout Options
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-4">
        {/* Main Chart Area */}
        <div className="lg:col-span-3 space-y-4">
          {/* Price Chart */}
          <Card className="bg-card border-border overflow-hidden transition-all duration-200 hover:border-border/80">
            <CardContent className="p-0 relative">
              {/* OHLC Info Bar */}
              <div className="flex items-center gap-4 px-4 py-2 border-b border-border text-xs">
                <span className="text-muted-foreground">O <span className="text-success font-mono">{priceSeries[priceSeries.length - 1]?.open?.toFixed(5)}</span></span>
                <span className="text-muted-foreground">H <span className="text-success font-mono">{priceSeries[priceSeries.length - 1]?.high?.toFixed(5)}</span></span>
                <span className="text-muted-foreground">L <span className="text-destructive font-mono">{priceSeries[priceSeries.length - 1]?.low?.toFixed(5)}</span></span>
                <span className="text-muted-foreground">C <span className="text-foreground font-mono">{priceSeries[priceSeries.length - 1]?.close?.toFixed(5)}</span></span>
                <span className="text-muted-foreground">Vol <span className="text-ai-purple font-mono">{(priceSeries[priceSeries.length - 1]?.volume / 1000).toFixed(1)}K</span></span>
              </div>
              
              {/* Chart */}
              <div className="h-[350px] px-2">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={priceSeries} margin={{ top: 10, right: 60, left: 0, bottom: 0 }}>
                    <XAxis 
                      dataKey="time" 
                      tick={{ fill: '#94a3b8', fontSize: 10 }}
                      axisLine={{ stroke: '#1f2937' }}
                      tickLine={false}
                    />
                    <YAxis 
                      yAxisId="price"
                      domain={['auto', 'auto']}
                      tick={{ fill: '#94a3b8', fontSize: 10 }}
                      axisLine={{ stroke: '#1f2937' }}
                      tickLine={false}
                      orientation="right"
                      tickFormatter={(v) => v.toFixed(4)}
                    />
                    {indicatorState.volume && (
                      <YAxis 
                        yAxisId="volume"
                        orientation="left"
                        tick={false}
                        axisLine={{ stroke: '#1f2937' }}
                        tickLine={false}
                      />
                    )}
                    <RechartsTooltip
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        border: '1px solid #1f2937',
                        borderRadius: '8px',
                        fontSize: '12px',
                        color: '#e2e8f0',
                      }}
                      labelStyle={{ color: '#94a3b8' }}
                    />
                    {currentPrice && (
                      <ReferenceLine 
                        y={currentPrice} 
                        stroke="#3b82f6"
                        strokeDasharray="3 3" 
                        yAxisId="price"
                        label={{ value: currentPrice.toFixed(5), position: 'right', fill: '#94a3b8', fontSize: 10 }}
                      />
                    )}
                    
                    {/* Price Layer based on chart type */}
                    {chartType === 'line' ? (
                      <Line
                        yAxisId="price"
                        type="monotone"
                        dataKey="close"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={false}
                        name="Price"
                      />
                    ) : chartType === 'bars' ? (
                      <Customized component={<OHLCBars data={priceSeries} />} />
                    ) : chartType === 'heikin-ashi' || chartType === 'candles' ? (
                      <Customized component={<CandleShapes data={priceSeries} />} />
                    ) : (
                      <ReferenceLine
                        yAxisId="price"
                        y={priceSeries[priceSeries.length - 1]?.close}
                        stroke="#1f2937"
                        label={{ value: 'Not implemented', position: 'insideTopLeft', fill: '#94a3b8' }}
                      />
                    )}
                    
                    {/* EMA Lines */}
                    {indicatorState.ema && (
                      <>
                        <Line type="monotone" dataKey="ema9" stroke="#0ea5e9" dot={false} strokeWidth={1.5} yAxisId="price" />
                        <Line type="monotone" dataKey="ema21" stroke="#38bdf8" dot={false} strokeWidth={1.2} yAxisId="price" />
                      </>
                    )}
                    
                    {/* Volume */}
                    {indicatorState.volume && (
                      <Bar dataKey="volume" yAxisId="volume" opacity={0.8} barSize={6}>
                        {priceSeries.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={entry.isUp ? '#16a34a99' : '#dc262699'} 
                          />
                        ))}
                      </Bar>
                    )}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              {/* Buy/Sell Controls - Right Side of Chart */}
              <div className="absolute right-4 top-1/2 -translate-y-1/2 flex flex-col items-center gap-2 z-10">
                {/* Buy Button */}
                <Button 
                  className="w-16 h-12 bg-success hover:bg-success/90 text-success-foreground transition-all duration-200 hover:scale-[1.05] active:scale-[0.95] hover:shadow-lg hover:shadow-success/20 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
                  onClick={() => handleTrade('BUY')}
                  disabled={isTrading}
                >
                  <div className="flex flex-col items-center gap-0.5">
                    {isTrading ? (
                      <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <TrendingUp className="h-4 w-4" />
                    )}
                    <span className="text-xs font-semibold">{isTrading ? '...' : 'Buy'}</span>
                  </div>
                </Button>

                {/* Quantity Input */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="w-16">
                      <Input
                        type="number"
                        value={tradeQuantity}
                        onChange={(e) => setTradeQuantity(e.target.value)}
                        min="0.01"
                        step="0.01"
                        className="text-sm font-mono bg-secondary border-border text-center h-10 px-1 transition-all duration-200 hover:border-ai-purple/50 focus:border-ai-purple"
                      />
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="left" className="text-xs bg-popover/95 backdrop-blur-sm">
                    Quantity (lots)
                  </TooltipContent>
                </Tooltip>

                {/* Sell Button */}
                <Button 
                  className="w-16 h-12 bg-destructive hover:bg-destructive/90 text-destructive-foreground transition-all duration-200 hover:scale-[1.05] active:scale-[0.95] hover:shadow-lg hover:shadow-destructive/20 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
                  onClick={() => handleTrade('SELL')}
                  disabled={isTrading}
                >
                  <div className="flex flex-col items-center gap-0.5">
                    {isTrading ? (
                      <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <TrendingDown className="h-4 w-4" />
                    )}
                    <span className="text-xs font-semibold">{isTrading ? '...' : 'Sell'}</span>
                  </div>
                </Button>
              </div>

              {/* RSI Panel */}
              {indicatorState.rsi && (
                <div className="h-[100px] mt-4 px-2">
                  <ResponsiveContainer width="100%" height="100%">
                    {rsiPanelData.length > 0 ? (
                      <ComposedChart data={rsiPanelData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                        <XAxis dataKey="time" hide />
                        <YAxis domain={[0, 100]} tick={{ fill: axisColor, fontSize: 10 }} ticks={[30, 50, 70]} />
                        <ReferenceLine y={70} stroke={bearish} strokeDasharray="3 3" />
                        <ReferenceLine y={30} stroke={bullish} strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="rsi" stroke="#8b5cf6" strokeWidth={1.3} dot={false} name="RSI" />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#0f172a',
                            border: '1px solid #1f2937',
                            borderRadius: 'var(--radius)',
                          }}
                        />
                      </ComposedChart>
                    ) : (
                      <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                        RSI data not available
                      </div>
                    )}
                  </ResponsiveContainer>
                </div>
              )}

              {/* MACD Panel */}
              {indicatorState.macd && (
                <div className="h-[100px] mt-4 px-2">
                  <ResponsiveContainer width="100%" height="100%">
                    {macdPanelData.length > 0 ? (
                      <ComposedChart data={macdPanelData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                        <XAxis dataKey="time" hide />
                        <YAxis tick={{ fill: axisColor, fontSize: 10 }} />
                        <ReferenceLine y={0} stroke={gridColor} />
                        <Bar dataKey="macdHist" fill="#0ea5e9" name="MACD Histogram" barSize={6} />
                        <Line type="monotone" dataKey="macdLine" stroke="#14b8a6" dot={false} strokeWidth={1.5} name="MACD" />
                        <Line type="monotone" dataKey="macdSignal" stroke="#f97316" dot={false} strokeWidth={1} name="Signal" />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#0f172a',
                            border: '1px solid #1f2937',
                            borderRadius: 'var(--radius)',
                          }}
                        />
                      </ComposedChart>
                    ) : (
                      <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                        MACD data not available
                      </div>
                    )}
                  </ResponsiveContainer>
                </div>
              )}

              {/* Custom Placeholder */}
              {indicatorState.custom && (
                <div className="mt-4 p-4 rounded-lg border border-dashed border-border bg-muted/30">
                  <p className="text-sm text-muted-foreground font-mono">User-defined indicator (backend required)</p>
                  <p className="text-xs text-muted-foreground mt-1">Provide server-side logic to render custom overlays.</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Open Positions Table */}
          <Card className="bg-card border-border transition-all duration-200 hover:border-border/80">
            <CardHeader className="py-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Activity className="h-4 w-4 text-ai-purple" />
                Open Positions
                <Badge variant="outline" className="ml-auto">{positions.length} Active</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-muted-foreground">
                      <th className="text-left p-3 font-medium">Symbol</th>
                      <th className="text-left p-3 font-medium">Side</th>
                      <th className="text-right p-3 font-medium">Entry</th>
                      <th className="text-right p-3 font-medium">Current</th>
                      <th className="text-right p-3 font-medium">P/L</th>
                      <th className="text-center p-3 font-medium">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {isLoadingPositions ? (
                      <tr>
                        <td colSpan={6} className="p-6 text-center text-muted-foreground">
                          <div className="flex items-center justify-center gap-2">
                            <div className="h-4 w-4 border-2 border-ai-purple border-t-transparent rounded-full animate-spin" />
                            Loading positions...
                          </div>
                        </td>
                      </tr>
                    ) : positions.length === 0 ? (
                      <tr>
                        <td colSpan={6} className="p-6 text-center text-muted-foreground">
                          No open positions
                        </td>
                      </tr>
                    ) : (
                      positions.map((pos) => {
                        const side = pos.quantity > 0 ? 'BUY' : 'SELL';
                        const entry = pos.average_price;
                        const current = pos.current_price || entry;
                        const pnl = pos.unrealized_pnl;
                        const isCrypto = pos.symbol.includes('BTC') || pos.symbol.includes('ETH') || pos.symbol.includes('USDT');
                        
                        return (
                          <tr key={pos.id} className="border-b border-border/50 hover:bg-muted/30 transition-all duration-150 hover:scale-[1.01]">
                            <td className="p-3 font-medium">{pos.symbol}</td>
                            <td className="p-3">
                              <Badge variant={side === 'BUY' ? 'default' : 'destructive'} className="text-xs">
                                {side}
                              </Badge>
                            </td>
                            <td className="p-3 text-right font-mono text-muted-foreground">
                              {entry.toFixed(isCrypto ? 2 : 5)}
                            </td>
                            <td className="p-3 text-right font-mono">
                              {current.toFixed(isCrypto ? 2 : 5)}
                            </td>
                            <td className={cn(
                              "p-3 text-right font-mono font-semibold transition-all",
                              pnl >= 0 ? "text-success" : "text-destructive"
                            )}>
                              {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
                            </td>
                            <td className="p-3 text-center">
                              <span className="inline-flex items-center gap-1.5">
                                <span className={cn(
                                  "h-2 w-2 rounded-full",
                                  pos.status === 'open' ? "bg-success animate-pulse" : "bg-muted-foreground"
                                )} />
                                <span className="text-xs text-muted-foreground">{pos.status}</span>
                              </span>
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

        </div>

        {/* AI Suggestions Panel */}
        <div className="space-y-4">
          <Card className="bg-card border-border transition-all duration-200 hover:border-border/80">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Brain className="h-4 w-4 text-ai-purple" />
                AI Signals
              </CardTitle>
              <CardDescription className="text-xs">ML-powered recommendations</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {isLoadingSignals ? (
                <div className="p-6 text-center text-muted-foreground">
                  <div className="flex items-center justify-center gap-2">
                    <div className="h-4 w-4 border-2 border-ai-purple border-t-transparent rounded-full animate-spin" />
                    Loading AI signals...
                  </div>
                </div>
              ) : aiSignals.length === 0 ? (
                <div className="p-6 text-center text-muted-foreground text-sm">
                  No AI signals available. Train a model first.
                </div>
              ) : (
                aiSignals.map((suggestion) => (
                  <div
                    key={suggestion.id}
                    className={cn(
                      "p-3 rounded-lg border transition-all duration-300 hover:scale-[1.02] hover:shadow-md cursor-pointer active:scale-[0.98]",
                      suggestion.signal === 'BUY' && "bg-success/5 border-success/20 hover:border-success/40 hover:bg-success/10",
                      suggestion.signal === 'SELL' && "bg-destructive/5 border-destructive/20 hover:border-destructive/40 hover:bg-destructive/10",
                      suggestion.signal === 'HOLD' && "bg-muted/30 border-border hover:border-muted-foreground/30 hover:bg-muted/50"
                    )}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <Badge className="text-[10px] px-1.5 py-0" variant={
                        suggestion.signal === 'BUY' ? 'default' : 
                        suggestion.signal === 'SELL' ? 'destructive' : 'secondary'
                      }>
                        {suggestion.signal === 'BUY' && <TrendingUp className="h-2.5 w-2.5 mr-0.5" />}
                        {suggestion.signal === 'SELL' && <TrendingDown className="h-2.5 w-2.5 mr-0.5" />}
                        {suggestion.signal === 'HOLD' && <Minus className="h-2.5 w-2.5 mr-0.5" />}
                        {suggestion.signal}
                      </Badge>
                      <span className="text-[10px] text-muted-foreground">{suggestion.symbol}</span>
                    </div>
                    <p className="text-[10px] text-muted-foreground line-clamp-2 mb-1.5">{suggestion.reason}</p>
                    <div className="flex items-center justify-between">
                      <span className="text-lg font-bold text-ai-purple">{suggestion.confidence}%</span>
                      <span className="text-[10px] text-muted-foreground">confidence</span>
                    </div>
                  </div>
                ))
              )}
            </CardContent>
          </Card>

          {/* Quick Stats */}
          <Card className="bg-card border-border transition-all duration-200 hover:border-border/80">
            <CardContent className="pt-4 space-y-2.5 text-sm">
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Spread</span>
                <span className="font-medium font-mono">0.8</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Leverage</span>
                <span className="font-medium font-mono">1:100</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Margin</span>
                <span className="font-medium font-mono">$10.85</span>
              </div>
              <div className="flex justify-between items-center pt-2 border-t border-border">
                <span className="text-muted-foreground">Free Margin</span>
                <span className="font-medium text-success font-mono">$9,856.20</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Trade Confirmation Modal */}
      <Dialog open={isConfirmOpen} onOpenChange={setIsConfirmOpen}>
        <DialogContent className="bg-card border-border">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-warning" />
              Confirm Trade
            </DialogTitle>
            <DialogDescription>
              Review your order details before confirming.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="grid grid-cols-2 gap-4 p-4 rounded-lg bg-muted/50">
              <div>
                <p className="text-sm text-muted-foreground">Action</p>
                <p className={cn(
                  "text-lg font-bold",
                  pendingTrade?.type === 'BUY' ? 'text-success' : 'text-destructive'
                )}>
                  {pendingTrade?.type}
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Symbol</p>
                <p className="text-lg font-bold">{selectedSymbol}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Quantity</p>
                <p className="text-lg font-bold">{tradeQuantity} lots</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Price</p>
                <p className="text-lg font-bold">{currentPrice.toFixed(5)}</p>
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsConfirmOpen(false)}>
              Cancel
            </Button>
            <Button 
              onClick={confirmTrade}
              disabled={isTrading}
              className={pendingTrade?.type === 'BUY' ? 'bg-success hover:bg-success/90' : 'bg-destructive hover:bg-destructive/90'}
            >
              {isTrading ? (
                <>
                  <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                  Processing...
                </>
              ) : (
                <>
                  <ShoppingCart className="h-4 w-4 mr-2" />
                  Confirm {pendingTrade?.type}
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Trading;