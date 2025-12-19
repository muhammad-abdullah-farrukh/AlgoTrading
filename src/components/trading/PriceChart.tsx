import { useMemo, useState } from 'react';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { OHLCVData, calculateRSI, calculateEMA, calculateMACD } from '@/utils/dummyData';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface PriceChartProps {
  data: OHLCVData[];
  symbol?: string;
  currentPrice?: number;
}

type Indicator = 'rsi' | 'ema9' | 'ema21' | 'macd' | 'volume';

export const PriceChart = ({ data, symbol = 'EURUSD', currentPrice }: PriceChartProps) => {
  const [activeIndicators, setActiveIndicators] = useState<Set<Indicator>>(new Set(['volume']));

  const toggleIndicator = (indicator: Indicator) => {
    setActiveIndicators(prev => {
      const next = new Set(prev);
      if (next.has(indicator)) {
        next.delete(indicator);
      } else {
        next.add(indicator);
      }
      return next;
    });
  };

  const chartData = useMemo(() => {
    const rsi = calculateRSI(data);
    const ema9 = calculateEMA(data, 9);
    const ema21 = calculateEMA(data, 21);
    const macdData = calculateMACD(data);

    return data.map((d, i) => ({
      time: new Date(d.time).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      close: d.close,
      high: d.high,
      low: d.low,
      volume: d.volume,
      rsi: rsi[i - 14] || null,
      ema9: ema9[i - 9] || null,
      ema21: ema21[i - 21] || null,
      macd: macdData.histogram[i - 35] || null,
      isUp: d.close >= d.open
    }));
  }, [data]);

  const priceChange = data.length > 1 
    ? ((data[data.length - 1].close - data[data.length - 2].close) / data[data.length - 2].close) * 100
    : 0;

  const indicators: { key: Indicator; label: string }[] = [
    { key: 'volume', label: 'Volume' },
    { key: 'ema9', label: 'EMA 9' },
    { key: 'ema21', label: 'EMA 21' },
    { key: 'rsi', label: 'RSI' },
    { key: 'macd', label: 'MACD' }
  ];

  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-2">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex items-center gap-4">
            <CardTitle className="text-xl font-semibold">{symbol}</CardTitle>
            <div className="flex items-center gap-2">
              <span className="text-2xl font-mono font-bold text-foreground">
                {currentPrice?.toFixed(5) || data[data.length - 1]?.close.toFixed(5)}
              </span>
              <Badge 
                variant={priceChange >= 0 ? "default" : "destructive"}
                className="flex items-center gap-1"
              >
                {priceChange >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
              </Badge>
            </div>
          </div>
          
          <div className="flex flex-wrap gap-2">
            {indicators.map(({ key, label }) => (
              <Button
                key={key}
                variant={activeIndicators.has(key) ? "default" : "outline"}
                size="sm"
                onClick={() => toggleIndicator(key)}
                className="text-xs"
              >
                {label}
              </Button>
            ))}
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
              <XAxis 
                dataKey="time" 
                tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                axisLine={{ stroke: 'hsl(var(--border))' }}
              />
              <YAxis 
                yAxisId="price"
                domain={['auto', 'auto']}
                tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                axisLine={{ stroke: 'hsl(var(--border))' }}
                tickFormatter={(val) => val.toFixed(4)}
              />
              {activeIndicators.has('volume') && (
                <YAxis 
                  yAxisId="volume"
                  orientation="right"
                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                  axisLine={{ stroke: 'hsl(var(--border))' }}
                />
              )}
              
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: 'var(--radius)',
                  color: 'hsl(var(--foreground))'
                }}
                labelStyle={{ color: 'hsl(var(--muted-foreground))' }}
              />
              
              {/* Price line */}
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="close"
                stroke="hsl(var(--primary))"
                strokeWidth={2}
                dot={false}
                name="Price"
              />
              
              {/* EMA lines */}
              {activeIndicators.has('ema9') && (
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="ema9"
                  stroke="hsl(var(--chart-3))"
                  strokeWidth={1}
                  dot={false}
                  name="EMA 9"
                />
              )}
              
              {activeIndicators.has('ema21') && (
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="ema21"
                  stroke="hsl(var(--chart-4))"
                  strokeWidth={1}
                  dot={false}
                  name="EMA 21"
                />
              )}
              
              {/* Volume bars */}
              {activeIndicators.has('volume') && (
                <Bar
                  yAxisId="volume"
                  dataKey="volume"
                  fill="hsl(var(--muted))"
                  opacity={0.5}
                  name="Volume"
                />
              )}

              {currentPrice && (
                <ReferenceLine
                  yAxisId="price"
                  y={currentPrice}
                  stroke="hsl(var(--primary))"
                  strokeDasharray="5 5"
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        
        {/* RSI Sub-chart */}
        {activeIndicators.has('rsi') && (
          <div className="h-[100px] mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis dataKey="time" hide />
                <YAxis 
                  domain={[0, 100]} 
                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 10 }}
                  ticks={[30, 50, 70]}
                />
                <ReferenceLine y={70} stroke="hsl(var(--destructive))" strokeDasharray="3 3" />
                <ReferenceLine y={30} stroke="hsl(var(--success))" strokeDasharray="3 3" />
                <Line
                  type="monotone"
                  dataKey="rsi"
                  stroke="hsl(var(--chart-5))"
                  strokeWidth={1}
                  dot={false}
                  name="RSI"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: 'var(--radius)'
                  }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
        
        {/* MACD Sub-chart */}
        {activeIndicators.has('macd') && (
          <div className="h-[100px] mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis dataKey="time" hide />
                <YAxis 
                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 10 }}
                />
                <ReferenceLine y={0} stroke="hsl(var(--border))" />
                <Bar
                  dataKey="macd"
                  fill="hsl(var(--chart-4))"
                  name="MACD Histogram"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: 'var(--radius)'
                  }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
