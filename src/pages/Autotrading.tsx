import { useState, useEffect, useMemo, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { StatusIndicator } from '@/components/trading/StatusIndicator';
import { 
  Bot, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  Shield, 
  Zap,
  Brain,
  Activity
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from '@/hooks/use-toast';
import api, { APIError } from '@/utils/api';

interface AIRecommendation {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reason: string;
  timestamp: Date;
}

interface RiskControl {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
}

const initialRiskControls: RiskControl[] = [
  { id: 'stopLoss', label: 'Stop Loss', value: 2.0, min: 0.5, max: 10, step: 0.5, unit: '%' },
  { id: 'takeProfit', label: 'Take Profit', value: 4.0, min: 1, max: 20, step: 0.5, unit: '%' },
  { id: 'maxDailyLoss', label: 'Max Daily Loss', value: 5.0, min: 1, max: 15, step: 0.5, unit: '%' },
  { id: 'positionSize', label: 'Position Size', value: 0.1, min: 0.01, max: 1, step: 0.01, unit: 'Lots' },
];

const Autotrading = () => {
  // Initialize with null to distinguish between "not loaded" and "disabled"
  const [isAutotradingEnabled, setIsAutotradingEnabled] = useState<boolean | null>(null);
  const [recommendations, setRecommendations] = useState<AIRecommendation[]>([]);
  const [riskControls, setRiskControls] = useState<RiskControl[]>(initialRiskControls);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState('EURUSD');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1d');
  const hasLoadedRecsRef = useRef(false);

  const SYMBOLS = useMemo(
    () => ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'BTCUSDT', 'AAPL'],
    []
  );
  const TIMEFRAMES = useMemo(
    () => [
      { value: '1m', label: '1m' },
      { value: '3m', label: '3m' },
      { value: '5m', label: '5m' },
      { value: '6m', label: '6m' },
      { value: '12m', label: '12m' },
      { value: '15m', label: '15m' },
      { value: '30m', label: '30m' },
      { value: '45m', label: '45m' },
      { value: '1h', label: '1h' },
      { value: '2h', label: '2h' },
      { value: '3h', label: '3h' },
      { value: '4h', label: '4h' },
      { value: '1d', label: '1d' },
      { value: '1w', label: '1w' },
      { value: '1M', label: '1M' },
      { value: '3M', label: '3M' },
      { value: '6M', label: '6M' },
      { value: '12M', label: '12M' },
    ],
    []
  );

  const timeframeToMs = (tf: string): number => {
    const norm = String(tf || '').trim();
    const map: Record<string, number> = {
      '1m': 60_000,
      '3m': 180_000,
      '5m': 300_000,
      '6m': 360_000,
      '12m': 720_000,
      '15m': 900_000,
      '30m': 1_800_000,
      '45m': 2_700_000,
      '1h': 3_600_000,
      '2h': 7_200_000,
      '3h': 10_800_000,
      '4h': 14_400_000,
      '1d': 86_400_000,
      '1w': 604_800_000,
      '1M': 2_592_000_000,
      '3M': 7_776_000_000,
      '6M': 15_552_000_000,
      '12M': 31_536_000_000,
    };
    return map[norm] ?? map[norm.toLowerCase()] ?? 60_000;
  };

  const fetchRecommendations = async () => {
    try {
      const res = await api.get<{
        signals: Array<{ symbol: string; signal: string; confidence: number; reason: string; timestamp?: string }>;
      }>('/api/ml/signals', {
        params: { symbols: selectedSymbol, timeframe: selectedTimeframe, min_confidence: 0.5 }
      });

      const recs: AIRecommendation[] = (res.signals || []).slice(0, 10).map((s, idx) => ({
        id: `${s.symbol}-${s.timestamp || idx}`,
        symbol: s.symbol,
        action: (s.signal || 'HOLD') as 'BUY' | 'SELL' | 'HOLD',
        confidence: Number(s.confidence || 0),
        reason: s.reason || '',
        timestamp: s.timestamp ? new Date(s.timestamp) : new Date(),
      }));
      setRecommendations(recs);
      hasLoadedRecsRef.current = true;
    } catch (error) {
      console.error('[Autotrading] Failed to load AI recommendations:', error);
      if (!hasLoadedRecsRef.current) setRecommendations([]);
    }
  };

  // Fetch settings from backend on mount
  useEffect(() => {
    let isMounted = true; // Track if component is still mounted
    
    const fetchSettings = async () => {
      try {
        setIsLoading(true);
        const settings = await api.get<{
          enabled: boolean;
          stop_loss_percent: number | null;
          take_profit_percent: number | null;
          max_daily_loss: number | null;
          position_size: number | null;
          selected_strategy_id: number | null;
        }>('/api/autotrading/settings');
        
        // Only update state if component is still mounted
        if (isMounted) {
          // Update state from backend (ALWAYS sync with backend)
          setIsAutotradingEnabled(settings.enabled || false);
        
          // Update risk controls from backend
          setRiskControls(prev => prev.map(rc => {
            if (rc.id === 'stopLoss' && settings.stop_loss_percent !== null) {
              return { ...rc, value: settings.stop_loss_percent };
            }
            if (rc.id === 'takeProfit' && settings.take_profit_percent !== null) {
              return { ...rc, value: settings.take_profit_percent };
            }
            if (rc.id === 'maxDailyLoss' && settings.max_daily_loss !== null) {
              return { ...rc, value: settings.max_daily_loss };
            }
            if (rc.id === 'positionSize' && settings.position_size !== null) {
              return { ...rc, value: settings.position_size };
            }
            return rc;
          }));
          
          console.log('[Autotrading] Loaded persisted settings from backend:', settings);
        }
      } catch (error) {
        console.error('[Autotrading] Failed to load settings:', error);
        if (isMounted) {
          if (error instanceof APIError && error.status !== 404) {
            toast({
              title: "Failed to Load Settings",
              description: "Could not load autotrading settings from backend.",
              variant: "destructive"
            });
          }
          setIsAutotradingEnabled(false);
          setIsLoading(false);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };
    
    // Fetch immediately
    fetchSettings();
    
    return () => {
      isMounted = false; // Mark as unmounted
    };
  }, []);

  useEffect(() => {
    (async () => {
      try {
        await api.put('/api/autotrading/settings', { timeframe: selectedTimeframe });
      } catch (e) {
        console.error('[Autotrading] Failed to persist timeframe:', e);
      }
    })();
  }, [selectedTimeframe]);

  useEffect(() => {
    fetchRecommendations();
    const intervalId = window.setInterval(
      fetchRecommendations,
      Math.min(60_000, Math.max(10_000, timeframeToMs(selectedTimeframe)))
    );

    return () => {
      window.clearInterval(intervalId);
    };
  }, [selectedSymbol, selectedTimeframe]);

  const handleToggleAutotrading = async (enabled: boolean) => {
    // Prevent toggling while loading or if state is null
    if (isAutotradingEnabled === null || isSaving) {
      return;
    }
    
    try {
      setIsSaving(true);
      const endpoint = enabled ? '/api/autotrading/enable' : '/api/autotrading/disable';
      await api.post(endpoint);
      
      // Update state immediately for responsive UI
      setIsAutotradingEnabled(enabled);
      console.log('[Autotrading] State persisted to backend: enabled=', enabled);
      
      // Loop is automatically started/stopped by backend when enabling/disabling
      toast({
        title: enabled ? "Autotrading Enabled" : "Autotrading Disabled",
        description: enabled 
          ? "AI is now automatically executing trades based on signals every minute." 
          : "Automatic trading has been paused.",
        variant: enabled ? "default" : "destructive"
      });
    } catch (error) {
      console.error('[Autotrading] Failed to update autotrading state:', error);
      // Revert state on error
      setIsAutotradingEnabled(!enabled);
      toast({
        title: "Update Failed",
        description: "Could not save autotrading state to backend.",
        variant: "destructive"
      });
    } finally {
      setIsSaving(false);
    }
  };


  const updateRiskControl = async (id: string, value: number) => {
    // Update local state immediately
    setRiskControls(prev => prev.map(rc => 
      rc.id === id ? { ...rc, value } : rc
    ));
    
    // Persist to backend
    try {
      const updateMap: Record<string, string> = {
        'stopLoss': 'stop_loss_percent',
        'takeProfit': 'take_profit_percent',
        'maxDailyLoss': 'max_daily_loss',
        'positionSize': 'position_size'
      };
      
      const backendField = updateMap[id];
      if (backendField) {
        await api.put('/api/autotrading/settings', {
          [backendField]: value
        });
        console.log(`[Autotrading] Risk control persisted: ${id}=${value}`);
      }
    } catch (error) {
      console.error(`[Autotrading] Failed to persist risk control ${id}:`, error);
      toast({
        title: "Save Failed",
        description: `Could not save ${id} to backend.`,
        variant: "destructive"
      });
    }
  };

  const getActionColor = (action: string) => {
    switch (action) {
      case 'BUY': return 'text-success';
      case 'SELL': return 'text-destructive';
      default: return 'text-warning';
    }
  };

  const getActionIcon = (action: string) => {
    switch (action) {
      case 'BUY': return TrendingUp;
      case 'SELL': return TrendingDown;
      default: return Activity;
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
    {/* Header */}
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Autotrading Control</h1>
        <p className="text-muted-foreground">Automated trading with AI-powered risk management</p>
      </div>
      <div className="flex items-center gap-4">
        <StatusIndicator 
          status={isAutotradingEnabled === true ? 'connected' : 'disconnected'} 
          label={isAutotradingEnabled === true ? 'Active' : isAutotradingEnabled === null ? 'Loading...' : 'Inactive'} 
        />
      </div>
    </div>

    <div className="grid gap-6 md:grid-cols-2">
      <Card className="bg-card/50 border-border/50 md:col-span-2">
        <CardHeader>
          <CardTitle className="text-lg font-semibold">AI Recommendations Filter</CardTitle>
          <CardDescription className="text-xs">Recommendations update every 10s</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label>Symbol</Label>
            <select
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
            >
              {SYMBOLS.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-2">
            <Label>Timeframe</Label>
            <select
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
            >
              {TIMEFRAMES.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
          </div>
        </CardContent>
      </Card>

      {/* Main Control Panel */}
      <Card className={cn(
        "bg-card border-2 transition-all duration-300",
        isAutotradingEnabled === true ? "border-ai-purple shadow-lg shadow-ai-purple/10" : "border-border"
      )}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bot className="h-5 w-5 text-ai-purple" />
            Autotrading Control
          </CardTitle>
          <CardDescription>
            Enable or disable AI-powered automatic trading execution
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
            <div className="flex items-center gap-3">
              <div className={cn(
                "p-3 rounded-full transition-colors",
                isAutotradingEnabled === true ? "bg-ai-purple/20" : "bg-muted"
              )}>
                <Zap className={cn(
                  "h-6 w-6 transition-colors",
                  isAutotradingEnabled === true ? "text-ai-purple" : "text-muted-foreground"
                )} />
              </div>
              <div>
                <p className="font-semibold text-foreground">
                  {isAutotradingEnabled === true ? 'Autotrading Active' : isAutotradingEnabled === null ? 'Loading...' : 'Autotrading Inactive'}
                </p>
                <p className="text-sm text-muted-foreground">
                  {isAutotradingEnabled === true
                    ? 'AI is monitoring markets and executing trades'
                    : isAutotradingEnabled === null
                    ? 'Loading autotrading status...'
                    : 'Toggle to enable automatic trade execution'}
                </p>
              </div>
            </div>
            <Switch
              checked={isAutotradingEnabled ?? false}
              onCheckedChange={handleToggleAutotrading}
              disabled={isLoading || isSaving || isAutotradingEnabled === null}
            />
          </div>

          {isAutotradingEnabled === true && (
            <div className="p-4 rounded-lg bg-warning/10 border border-warning/20 animate-fade-in">
              <div className="flex items-start gap-3">
                <AlertTriangle className="h-5 w-5 text-warning mt-0.5" />
                <div>
                  <p className="font-medium text-warning">Caution: Live Trading</p>
                  <p className="text-sm text-muted-foreground">
                    The bot will execute real trades based on AI signals. Monitor closely.
                  </p>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

        {/* Risk Settings */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-ai-purple" />
              Risk Settings
            </CardTitle>
            <CardDescription>
              Configure persisted risk parameters
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid gap-4 sm:grid-cols-2">
              {riskControls.map((control) => (
                <div key={control.id} className="p-4 rounded-lg border bg-muted/30 border-border">
                  <div className="flex items-center justify-between mb-3">
                    <Label className="font-medium">{control.label}</Label>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Input
                        type="number"
                        value={control.value}
                        onChange={(e) => updateRiskControl(control.id, parseFloat(e.target.value) || 0)}
                        min={control.min}
                        max={control.max}
                        step={control.step}
                        className="flex-1 bg-secondary border-0"
                      />
                      <span className="text-sm text-muted-foreground w-12">{control.unit}</span>
                    </div>
                    <Slider
                      value={[control.value]}
                      onValueChange={([v]) => updateRiskControl(control.id, v)}
                      min={control.min}
                      max={control.max}
                      step={control.step}
                      className="w-full"
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* AI Recommendations */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-ai-purple" />
              AI Trading Recommendations
            </CardTitle>
            <CardDescription>
              Real-time AI-generated trading signals based on market analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recommendations.length === 0 ? (
                <div className="p-4 rounded-lg bg-muted/50">
                  <p className="text-sm text-muted-foreground">No recommendations available.</p>
                </div>
              ) : (
                recommendations.map((rec) => {
                  const ActionIcon = getActionIcon(rec.action);
                  return (
                    <div
                      key={rec.id}
                      className="flex items-start gap-4 p-4 rounded-lg bg-muted/50 hover:bg-muted/70 transition-colors"
                    >
                      <div
                        className={cn(
                          "p-2 rounded-full",
                          rec.action === 'BUY'
                            ? 'bg-success/10'
                            : rec.action === 'SELL'
                              ? 'bg-destructive/10'
                              : 'bg-warning/10'
                        )}
                      >
                        <ActionIcon className={cn('h-5 w-5', getActionColor(rec.action))} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-semibold text-foreground">{rec.symbol}</span>
                          <Badge
                            variant={rec.action === 'BUY' ? 'default' : rec.action === 'SELL' ? 'destructive' : 'secondary'}
                          >
                            {rec.action}
                          </Badge>
                          <Badge variant="outline" className="bg-ai-purple/10 text-ai-purple border-0">
                            {(rec.confidence * 100).toFixed(0)}% confidence
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">{rec.reason}</p>
                        <p className="text-xs text-muted-foreground mt-1">{rec.timestamp.toLocaleTimeString()}</p>
                      </div>
                      <Button 
                        variant="outline" 
                        size="sm"
                        disabled={!isAutotradingEnabled || isAutotradingEnabled === null}
                        className="hover:bg-ai-purple/10 hover:text-ai-purple hover:border-ai-purple/30"
                      >
                        Execute
                      </Button>
                    </div>
                  );
                })
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Autotrading;