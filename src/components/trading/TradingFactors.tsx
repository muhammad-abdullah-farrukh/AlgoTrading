import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Shield, Target, Brain, Scale } from 'lucide-react';

interface TradingFactor {
  id: string;
  name: string;
  value: number;
  enabled: boolean;
  icon: React.ElementType;
  unit: string;
  min: number;
  max: number;
  step: number;
}

export const TradingFactors = () => {
  const [factors, setFactors] = useState<TradingFactor[]>([
    { id: 'stopLoss', name: 'Stop Loss', value: 50, enabled: true, icon: Shield, unit: 'pips', min: 5, max: 200, step: 5 },
    { id: 'takeProfit', name: 'Take Profit', value: 100, enabled: true, icon: Target, unit: 'pips', min: 10, max: 500, step: 10 },
    { id: 'aiConfidence', name: 'AI Confidence Threshold', value: 75, enabled: true, icon: Brain, unit: '%', min: 50, max: 99, step: 1 },
    { id: 'lotSize', name: 'Trade Volume / Lot Size', value: 0.1, enabled: true, icon: Scale, unit: 'lots', min: 0.01, max: 10, step: 0.01 },
  ]);

  const updateFactor = (id: string, updates: Partial<TradingFactor>) => {
    setFactors(prev => prev.map(f => f.id === id ? { ...f, ...updates } : f));
  };

  return (
    <Card className="bg-card border-border">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shield className="h-5 w-5 text-primary" />
          Trading Factors & Risk Controls
        </CardTitle>
        <CardDescription>
          Configure risk parameters and trading thresholds for your strategies
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-6 md:grid-cols-2">
          {factors.map((factor) => (
            <div
              key={factor.id}
              className={`p-4 rounded-lg border transition-all duration-300 ${
                factor.enabled 
                  ? 'bg-muted/30 border-primary/30' 
                  : 'bg-muted/10 border-border opacity-60'
              }`}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <factor.icon className={`h-4 w-4 ${factor.enabled ? 'text-primary' : 'text-muted-foreground'}`} />
                  <Label className="font-medium">{factor.name}</Label>
                </div>
                <Switch
                  checked={factor.enabled}
                  onCheckedChange={(enabled) => updateFactor(factor.id, { enabled })}
                />
              </div>
              
              {factor.id === 'aiConfidence' ? (
                <div className="space-y-3">
                  <Slider
                    value={[factor.value]}
                    onValueChange={([value]) => updateFactor(factor.id, { value })}
                    min={factor.min}
                    max={factor.max}
                    step={factor.step}
                    disabled={!factor.enabled}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>{factor.min}%</span>
                    <span className="font-medium text-foreground">{factor.value}%</span>
                    <span>{factor.max}%</span>
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <Input
                    type="number"
                    value={factor.value}
                    onChange={(e) => updateFactor(factor.id, { value: parseFloat(e.target.value) || 0 })}
                    min={factor.min}
                    max={factor.max}
                    step={factor.step}
                    disabled={!factor.enabled}
                    className="flex-1"
                  />
                  <span className="text-sm text-muted-foreground w-12">{factor.unit}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};
