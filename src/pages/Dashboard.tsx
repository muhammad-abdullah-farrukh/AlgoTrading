import { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogDescription } from '@/components/ui/dialog';
import { PriceChart } from '@/components/trading/PriceChart';
import { StatusIndicator } from '@/components/trading/StatusIndicator';
import { generateOHLCVData } from '@/utils/dummyData';
import { useTrades } from '@/contexts/TradesContext';
import { useMockPriceStream, useWebSocket } from '@/hooks/useWebSocket';
import { TrendingUp, TrendingDown, Activity, DollarSign, BarChart3, AlertTriangle, Zap } from 'lucide-react';
import { cn } from '@/lib/utils';

const Dashboard = () => {
  const { trades } = useTrades();
  const [ohlcvData] = useState(() => generateOHLCVData(60));
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [showModeModal, setShowModeModal] = useState(false);
  const [positions, setPositions] = useState<Array<{
    id: number;
    symbol: string;
    quantity: number;
    average_price: number;
    current_price: number | null;
    unrealized_pnl: number;
    realized_pnl: number;
    status: string;
    opened_at: string | null;
  }>>([]);
  // WebSocket connection for positions
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
      console.error('[Dashboard] Positions WebSocket error:', error);
    },
    autoConnect: true,
    reconnectAttempts: 5,
  });

  // WebSocket connection for trades
  useWebSocket({
    url: '/ws/trades',
    onMessage: (data) => {
      if (data && typeof data === 'object' && 'type' in data) {
        const msg = data as { type: string; data?: unknown[] };
        if (msg.type === 'trades' && Array.isArray(msg.data)) {
          // Note: Trades are managed by TradesContext, this is just for monitoring
          console.log('[Dashboard] New trades received:', msg.data.length);
        }
      }
    },
    onError: (error) => {
      console.error('[Dashboard] Trades WebSocket error:', error);
    },
    autoConnect: true,
    reconnectAttempts: 5,
  });

  // Mock price stream - only use if WebSocket is not connected
  const { price, isStreaming, startStream, stopStream } = useMockPriceStream('EURUSD', ohlcvData[ohlcvData.length - 1]?.close);

  useEffect(() => {
    // Only start mock stream if WebSocket is not connected
    if (!isPositionsConnected) {
      startStream();
    } else {
      stopStream();
    }
    return () => stopStream();
  }, [isPositionsConnected, startStream, stopStream]);

  const stats = useMemo(() => {
    const totalPnL = trades.reduce((sum, t) => sum + t.profitLoss, 0);
    const winningTrades = trades.filter(t => t.profitLoss > 0);
    const winRate = (winningTrades.length / trades.length) * 100;
    const avgWin = winningTrades.reduce((sum, t) => sum + t.profitLoss, 0) / winningTrades.length || 0;
    const losingTrades = trades.filter(t => t.profitLoss < 0);
    const avgLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.profitLoss, 0) / losingTrades.length) || 0;
    
    return {
      totalPnL,
      winRate,
      totalTrades: trades.length,
      avgWin,
      avgLoss,
      profitFactor: avgWin / avgLoss || 0
    };
  }, [trades]);

  const recentTrades = trades.slice(0, 5);

  const handleModeToggle = () => {
    setShowModeModal(true);
  };

  const confirmLiveMode = () => {
    setIsLiveMode(true);
    setShowModeModal(false);
  };

  const stayInDemo = () => {
    setIsLiveMode(false);
    setShowModeModal(false);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Dashboard</h1>
          <p className="text-muted-foreground">Real-time market overview and trading performance</p>
        </div>
        <div className="flex items-center gap-4">
          <StatusIndicator status={isStreaming ? 'connected' : 'disconnected'} label="Live Feed" />
          <Button
            variant={isLiveMode ? "default" : "secondary"}
            size="sm"
            onClick={handleModeToggle}
            className={cn(
              "transition-all duration-300",
              isLiveMode 
                ? "bg-ai-purple hover:bg-ai-purple/90 text-ai-purple-foreground shadow-lg shadow-ai-purple/20" 
                : "hover:bg-secondary/80"
            )}
          >
            {isLiveMode ? (
              <>
                <Zap className="h-4 w-4 mr-1.5" />
                Live Trading
              </>
            ) : (
              <>
                <Activity className="h-4 w-4 mr-1.5" />
                Demo Mode
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Live Mode Banner */}
      {isLiveMode && (
        <div className="p-3 rounded-lg bg-ai-purple/10 border border-ai-purple/20 flex items-center gap-3 animate-fade-in">
          <div className="h-2 w-2 rounded-full bg-ai-purple animate-pulse" />
          <p className="text-sm text-ai-purple font-medium">Live Trading Mode Active</p>
          <p className="text-xs text-muted-foreground ml-auto">Real trades will be executed</p>
        </div>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-card border-border card-hover">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total P/L</p>
                <p className={cn(
                  "text-2xl font-bold tabular-nums transition-all duration-500",
                  stats.totalPnL >= 0 ? "text-success" : "text-destructive"
                )}>
                  {stats.totalPnL >= 0 ? '+' : ''}${stats.totalPnL.toFixed(2)}
                </p>
              </div>
              <div className={cn(
                "p-3 rounded-full transition-colors",
                stats.totalPnL >= 0 ? "bg-success/10" : "bg-destructive/10"
              )}>
                <DollarSign className={cn(
                  "h-6 w-6",
                  stats.totalPnL >= 0 ? "text-success" : "text-destructive"
                )} />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border card-hover">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Win Rate</p>
                <p className="text-2xl font-bold text-foreground tabular-nums">{stats.winRate.toFixed(1)}%</p>
              </div>
              <div className="p-3 rounded-full bg-ai-purple/10">
                <TrendingUp className="h-6 w-6 text-ai-purple" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border card-hover">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Trades</p>
                <p className="text-2xl font-bold text-foreground tabular-nums">{stats.totalTrades}</p>
              </div>
              <div className="p-3 rounded-full bg-chart-4/10">
                <Activity className="h-6 w-6 text-chart-4" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border card-hover">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Profit Factor</p>
                <p className="text-2xl font-bold text-foreground tabular-nums">{stats.profitFactor.toFixed(2)}</p>
              </div>
              <div className="p-3 rounded-full bg-chart-3/10">
                <BarChart3 className="h-6 w-6 text-chart-3" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Price Chart */}
      <PriceChart data={ohlcvData} symbol="EURUSD" currentPrice={price} />

      {/* Recent Trades */}
      <Card className="bg-card border-border">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-lg">Recent Trades</CardTitle>
          <Badge variant="secondary">{trades.length} trades</Badge>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {recentTrades.map((trade) => (
              <div 
                key={trade.id} 
                className="flex items-center justify-between p-3 rounded-lg bg-muted/50 hover:bg-muted/70 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className={cn(
                    "p-2 rounded-full transition-colors",
                    trade.action === 'BUY' ? "bg-success/10" : "bg-destructive/10"
                  )}>
                    {trade.action === 'BUY' ? (
                      <TrendingUp className="h-4 w-4 text-success" />
                    ) : (
                      <TrendingDown className="h-4 w-4 text-destructive" />
                    )}
                  </div>
                  <div>
                    <p className="font-medium text-foreground">{trade.symbol}</p>
                    <p className="text-xs text-muted-foreground">
                      {trade.action} {trade.quantity.toFixed(2)} @ {trade.price.toFixed(5)}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className={cn(
                    "font-semibold tabular-nums",
                    trade.profitLoss >= 0 ? "text-success" : "text-destructive"
                  )}>
                    {trade.profitLoss >= 0 ? '+' : ''}{trade.profitLoss.toFixed(2)} USD
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {trade.timestamp.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Demo/Live Mode Modal */}
      <Dialog open={showModeModal} onOpenChange={setShowModeModal}>
        <DialogContent className="bg-card border-border">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-warning" />
              Enable Live Trading?
            </DialogTitle>
            <DialogDescription>
              Switching to live mode will execute real trades using your connected trading account.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <div className="p-4 rounded-lg bg-warning/10 border border-warning/20">
              <p className="text-sm text-muted-foreground">
                <strong className="text-warning">Warning:</strong> Live trading involves real money and carries financial risk. 
                Make sure you understand the risks before proceeding.
              </p>
            </div>
          </div>
          <DialogFooter className="flex-col sm:flex-row gap-2">
            <Button variant="outline" onClick={stayInDemo} className="flex-1">
              Stay in Demo
            </Button>
            <Button 
              onClick={confirmLiveMode}
              className="flex-1 bg-ai-purple hover:bg-ai-purple/90 text-ai-purple-foreground"
            >
              <Zap className="h-4 w-4 mr-2" />
              Enable Live Trading
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Dashboard;