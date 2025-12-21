import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Edit, Trash2, TrendingUp, TrendingDown, Cpu, Settings, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

export type Strategy = {
  id: number;
  name: string;
  description?: string | null;
  strategy_type: 'technical' | 'ai' | 'custom' | string;
  enabled: boolean;
  parameters?: string | null;
  performance: number;
  trades_count: number;
};

interface StrategyCardProps {
  strategy: Strategy;
  onToggle: (id: string, enabled: boolean) => void;
  onEdit: (id: string) => void;
  onDelete: (id: string) => void;
}

const typeIcons = {
  technical: Settings,
  ai: Sparkles,
  custom: Cpu
};

const typeLabels = {
  technical: 'Technical',
  ai: 'AI/ML',
  custom: 'Custom'
};

export const StrategyCard = ({ strategy, onToggle, onEdit, onDelete }: StrategyCardProps) => {
  const TypeIcon = typeIcons[(strategy.strategy_type as keyof typeof typeIcons) ?? 'custom'] ?? Cpu;
  
  return (
    <Card className={cn(
      "bg-card border-border transition-all duration-200",
      strategy.enabled ? "ring-1 ring-primary/50" : "opacity-75"
    )}>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <CardTitle className="text-lg truncate">{strategy.name}</CardTitle>
              <Badge variant="secondary" className="flex items-center gap-1">
                <TypeIcon className="h-3 w-3" />
                {typeLabels[(strategy.strategy_type as keyof typeof typeLabels) ?? 'custom'] ?? String(strategy.strategy_type)}
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground line-clamp-2">
              {strategy.description}
            </p>
          </div>
          <Switch
            checked={strategy.enabled}
            onCheckedChange={(checked) => onToggle(String(strategy.id), checked)}
          />
        </div>
      </CardHeader>
      
      <CardContent>
        {/* Performance Stats */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-muted/50 rounded-lg p-3">
            <div className="text-xs text-muted-foreground mb-1">Performance</div>
            <div className={cn(
              "flex items-center gap-1 text-lg font-semibold",
              strategy.performance >= 0 ? "text-success" : "text-destructive"
            )}>
              {strategy.performance >= 0 ? (
                <TrendingUp className="h-4 w-4" />
              ) : (
                <TrendingDown className="h-4 w-4" />
              )}
              {strategy.performance >= 0 ? '+' : ''}{strategy.performance}%
            </div>
          </div>
          <div className="bg-muted/50 rounded-lg p-3">
            <div className="text-xs text-muted-foreground mb-1">Total Trades</div>
            <div className="text-lg font-semibold text-foreground">
              {strategy.trades_count}
            </div>
          </div>
        </div>
        
        {/* Parameters */}
        <div className="mb-4">
          <div className="text-xs text-muted-foreground mb-2">Parameters</div>
          <div className="flex flex-wrap gap-2">
            {Object.entries((() => {
              if (!strategy.parameters) return {};
              if (typeof strategy.parameters === 'string') {
                try {
                  return JSON.parse(strategy.parameters) as Record<string, unknown>;
                } catch {
                  return {};
                }
              }
              return {};
            })()).map(([key, value]) => (
              <Badge key={key} variant="outline" className="text-xs">
                {key}: {String(value)}
              </Badge>
            ))}
          </div>
        </div>
        
        {/* Actions */}
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            className="flex-1"
            onClick={() => onEdit(String(strategy.id))}
          >
            <Edit className="h-4 w-4 mr-1" />
            Edit
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="text-destructive hover:text-destructive"
            onClick={() => onDelete(String(strategy.id))}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};
