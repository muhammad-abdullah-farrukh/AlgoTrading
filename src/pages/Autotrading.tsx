import { useState, useEffect } from 'react';
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
  Activity,
  Sparkles,
  Play
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
  aiEnabled: boolean;
}

const mockRecommendations: AIRecommendation[] = [
  {
    id: '1',
    symbol: 'EURUSD',
    action: 'BUY',
    confidence: 0.85,
    reason: 'Strong bullish divergence on RSI with EMA crossover confirmation',
    timestamp: new Date()
  },
  {
    id: '2',
    symbol: 'GBPUSD',
    action: 'SELL',
    confidence: 0.72,
    reason: 'Bearish engulfing pattern at key resistance level',
    timestamp: new Date(Date.now() - 300000)
  },
  {
    id: '3',
    symbol: 'USDJPY',
    action: 'HOLD',
    confidence: 0.45,
    reason: 'Market indecision - awaiting clearer price action signals',
    timestamp: new Date(Date.now() - 600000)
  }
];

const initialRiskControls: RiskControl[] = [
  { id: 'stopLoss', label: 'Stop Loss', value: 2.0, min: 0.5, max: 10, step: 0.5, unit: '%', aiEnabled: false },
  { id: 'takeProfit', label: 'Take Profit', value: 4.0, min: 1, max: 20, step: 0.5, unit: '%', aiEnabled: false },
  { id: 'maxDailyLoss', label: 'Max Daily Loss', value: 5.0, min: 1, max: 15, step: 0.5, unit: '%', aiEnabled: true },
  { id: 'positionSize', label: 'Position Size', value: 0.1, min: 0.01, max: 1, step: 0.01, unit: 'Lots', aiEnabled: false },
];

const Autotrading = () => {
  const [isAutotradingEnabled, setIsAutotradingEnabled] = useState(false);
  const [riskLevel, setRiskLevel] = useState([50]);
  const [maxPositions, setMaxPositions] = useState([3]);
  const [recommendations] = useState<AIRecommendation[]>(mockRecommendations);
  const [riskControls, setRiskControls] = useState<RiskControl[]>(initialRiskControls);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  // Fetch settings from backend on mount
  useEffect(() => {
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
        
        // Update state from backend
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
      } catch (error) {
        console.error('[Autotrading] Failed to load settings:', error);
        if (error instanceof APIError && error.status !== 404) {
          toast({
            title: "Failed to Load Settings",
            description: "Could not load autotrading settings from backend.",
            variant: "destructive"
          });
        }
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchSettings();
  }, []);

  const handleToggleAutotrading = async (enabled: boolean) => {
    try {
      setIsSaving(true);
      const endpoint = enabled ? '/api/autotrading/enable' : '/api/autotrading/disable';
      await api.post(endpoint);
      
      setIsAutotradingEnabled(enabled);
      console.log('[Autotrading] State persisted to backend: enabled=', enabled);
      
      // Auto-start the loop when enabling
      if (enabled) {
        try {
          await api.post('/api/autotrading/start-loop');
          console.log('[Autotrading] Background loop started');
        } catch (loopError) {
          console.warn('[Autotrading] Failed to start loop:', loopError);
          // Non-critical, continue
        }
      } else {
        // Stop the loop when disabling
        try {
          await api.post('/api/autotrading/stop-loop');
          console.log('[Autotrading] Background loop stopped');
        } catch (loopError) {
          console.warn('[Autotrading] Failed to stop loop:', loopError);
          // Non-critical, continue
        }
      }
      
      toast({
        title: enabled ? "Autotrading Enabled" : "Autotrading Disabled",
        description: enabled 
          ? "AI will now execute trades based on strategy signals." 
          : "Automatic trading has been paused.",
        variant: enabled ? "default" : "destructive"
      });
    } catch (error) {
      console.error('[Autotrading] Failed to update autotrading state:', error);
      toast({
        title: "Update Failed",
        description: "Could not save autotrading state to backend.",
        variant: "destructive"
      });
    } finally {
      setIsSaving(false);
    }
  };

  const handleExecuteStrategy = async () => {
    try {
      setIsSaving(true);
      const response = await api.post<{
        status: string;
        message: string;
        executed_trades?: Array<any>;
        skipped_signals?: Array<any>;
      }>('/api/autotrading/execute-strategy');
      
      toast({
        title: "Strategy Executed",
        description: response.message || "Trading strategy executed successfully",
        variant: "default"
      });
      
      if (response.executed_trades && response.executed_trades.length > 0) {
        console.log('[Autotrading] Executed trades:', response.executed_trades);
      }
    } catch (error) {
      console.error('[Autotrading] Failed to execute strategy:', error);
      toast({
        title: "Execution Failed",
        description: "Could not execute trading strategy.",
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

  const toggleAIControl = (id: string) => {
    setRiskControls(prev => prev.map(rc => {
      if (rc.id === id) {
        const newAiEnabled = !rc.aiEnabled;
        toast({
          title: newAiEnabled ? "AI Control Enabled" : "Manual Control",
          description: `${rc.label} is now ${newAiEnabled ? 'controlled by AI' : 'manually configured'}`,
        });
        return { ...rc, aiEnabled: newAiEnabled };
      }
      return rc;
    }));
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
          <p className="text-muted-foreground">Configure AI-powered automated trading</p>
        </div>
        <div className="flex items-center gap-4">
          <StatusIndicator 
            status={isAutotradingEnabled ? 'connected' : 'disconnected'} 
            label={isAutotradingEnabled ? 'Active' : 'Inactive'} 
          />
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Main Control Panel */}
        <Card className={cn(
          "bg-card border-2 transition-all duration-300",
          isAutotradingEnabled ? "border-ai-purple shadow-lg shadow-ai-purple/10" : "border-border"
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
                  isAutotradingEnabled ? "bg-ai-purple/20" : "bg-muted"
                )}>
                  <Zap className={cn(
                    "h-6 w-6 transition-colors",
                    isAutotradingEnabled ? "text-ai-purple" : "text-muted-foreground"
                  )} />
                </div>
                <div>
                  <p className="font-semibold text-foreground">
                    {isAutotradingEnabled ? 'Autotrading Active' : 'Autotrading Inactive'}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {isAutotradingEnabled 
                      ? 'AI is monitoring markets and executing trades' 
                      : 'Toggle to enable automatic trade execution'}
                  </p>
                </div>
              </div>
              <Switch
                checked={isAutotradingEnabled}
                onCheckedChange={handleToggleAutotrading}
                disabled={isLoading || isSaving}
              />
            </div>

            {isAutotradingEnabled && (
              <div className="flex gap-2">
                <Button
                  onClick={handleExecuteStrategy}
                  disabled={isSaving}
                  className="flex-1"
                >
                  <Play className="h-4 w-4 mr-2" />
                  Execute Strategy Now
                </Button>
              </div>
            )}

            {isAutotradingEnabled && (
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

        {/* Basic Risk Settings */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-ai-purple" />
              Risk Settings
            </CardTitle>
            <CardDescription>
              Configure trading limits and risk parameters
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label>Risk Level</Label>
                <Badge variant="outline">{riskLevel[0]}%</Badge>
              </div>
              <Slider
                value={riskLevel}
                onValueChange={setRiskLevel}
                max={100}
                step={5}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                {riskLevel[0] < 30 ? 'Conservative' : riskLevel[0] < 70 ? 'Moderate' : 'Aggressive'} risk tolerance
              </p>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label>Max Concurrent Positions</Label>
                <Badge variant="outline">{maxPositions[0]}</Badge>
              </div>
              <Slider
                value={maxPositions}
                onValueChange={setMaxPositions}
                max={10}
                min={1}
                step={1}
                className="w-full"
              />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Advanced Risk Management with AI Toggle */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-ai-purple" />
            Advanced Risk Management
          </CardTitle>
          <CardDescription>
            Configure individual risk controls with optional AI automation
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2">
            {riskControls.map((control) => (
              <div 
                key={control.id} 
                className={cn(
                  "p-4 rounded-lg border transition-all duration-300",
                  control.aiEnabled 
                    ? "bg-ai-purple/5 border-ai-purple/30" 
                    : "bg-muted/30 border-border"
                )}
              >
                <div className="flex items-center justify-between mb-3">
                  <Label className="font-medium">{control.label}</Label>
                  <div className="flex items-center gap-2">
                    {control.aiEnabled && (
                      <Badge variant="secondary" className="bg-ai-purple/20 text-ai-purple text-xs">
                        <Brain className="h-3 w-3 mr-1" />
                        AI
                      </Badge>
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => toggleAIControl(control.id)}
                      className={cn(
                        "h-7 px-2 text-xs",
                        control.aiEnabled ? "text-ai-purple" : "text-muted-foreground"
                      )}
                    >
                      {control.aiEnabled ? 'Manual' : 'Auto (AI)'}
                    </Button>
                  </div>
                </div>
                
                {control.aiEnabled ? (
                  <div className="flex items-center gap-2 p-3 rounded bg-ai-purple/10">
                    <Brain className="h-4 w-4 text-ai-purple" />
                    <span className="text-sm text-muted-foreground">AI is managing this parameter</span>
                  </div>
                ) : (
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
                )}
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
            {recommendations.map((rec) => {
              const ActionIcon = getActionIcon(rec.action);
              return (
                <div 
                  key={rec.id}
                  className="flex items-start gap-4 p-4 rounded-lg bg-muted/50 hover:bg-muted/70 transition-colors"
                >
                  <div className={cn(
                    "p-2 rounded-full",
                    rec.action === 'BUY' ? 'bg-success/10' : 
                    rec.action === 'SELL' ? 'bg-destructive/10' : 'bg-warning/10'
                  )}>
                    <ActionIcon className={cn("h-5 w-5", getActionColor(rec.action))} />
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
                    <p className="text-xs text-muted-foreground mt-1">
                      {rec.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                  <Button 
                    variant="outline" 
                    size="sm"
                    disabled={!isAutotradingEnabled}
                    className="hover:bg-ai-purple/10 hover:text-ai-purple hover:border-ai-purple/30"
                  >
                    Execute
                  </Button>
                </div>
              );
            })}
          </div>
          
          <div className="mt-4 p-4 rounded-lg bg-muted/30 border border-dashed border-border">
            <p className="text-sm text-muted-foreground text-center">
              <Brain className="h-4 w-4 inline mr-1" />
              Connect to Python ML backend for live AI recommendations
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Autotrading;