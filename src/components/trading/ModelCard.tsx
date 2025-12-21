import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Brain, RefreshCw, Check, TrendingUp, Database, Calendar } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface MLModel {
  id: string;
  name: string;
  type: 'logistic_regression' | 'random_forest' | 'lstm' | 'ensemble';
  accuracy: number;
  lastTrained: Date | null;
  datasetSize: number;
  isActive: boolean;
  featureImportance: { feature: string; weight: number }[];
  status: 'ready' | 'training' | 'error';
}

interface ModelCardProps {
  model: MLModel;
  onSelect: (id: string) => void;
  onRetrain: (id: string) => void;
}

const modelTypeLabels: Record<MLModel['type'], string> = {
  logistic_regression: 'Logistic Regression',
  random_forest: 'Random Forest',
  lstm: 'LSTM Neural Network',
  ensemble: 'Ensemble Model'
};

const modelTypeColors: Record<MLModel['type'], string> = {
  logistic_regression: 'bg-blue-500/20 text-blue-400',
  random_forest: 'bg-emerald-500/20 text-emerald-400',
  lstm: 'bg-purple-500/20 text-purple-400',
  ensemble: 'bg-amber-500/20 text-amber-400'
};

export const ModelCard = ({ model, onSelect, onRetrain }: ModelCardProps) => {
  const isTraining = model.status === 'training';
  const lastTrainedLabel = model.lastTrained ? model.lastTrained.toLocaleDateString() : '--';

  return (
    <Card className={cn(
      "transition-all duration-300 card-hover",
      model.isActive && "ring-2 ring-primary shadow-lg shadow-primary/10"
    )}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            <CardTitle className="text-lg">{model.name}</CardTitle>
          </div>
          <div className="flex items-center gap-2">
            {model.isActive && (
              <Badge className="bg-primary/20 text-primary border-0">
                <Check className="h-3 w-3 mr-1" />
                Active
              </Badge>
            )}
            <Badge className={modelTypeColors[model.type]}>
              {modelTypeLabels[model.type]}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Key Stats */}
        <div className="grid grid-cols-3 gap-4">
          <div key={`accuracy-${model.id}-${model.accuracy}`} className="text-center p-3 bg-muted/50 rounded-lg relative">
            <TrendingUp className="h-4 w-4 mx-auto mb-1 text-muted-foreground" />
            <p key={`accuracy-value-${model.id}-${model.accuracy}`} className="text-2xl font-bold text-foreground break-words">{model.accuracy.toFixed(3)}%</p>
            <p className="text-xs text-muted-foreground">Accuracy</p>
          </div>
          <div key={`samples-${model.id}-${model.datasetSize}`} className="text-center p-3 bg-muted/50 rounded-lg relative">
            <Database className="h-4 w-4 mx-auto mb-1 text-muted-foreground" />
            <p key={`samples-value-${model.id}-${model.datasetSize}`} className="text-2xl font-bold text-foreground break-words">{(model.datasetSize / 1000).toFixed(1)}k</p>
            <p className="text-xs text-muted-foreground">Samples</p>
          </div>
          <div key={`lastTrained-${model.id}-${model.lastTrained ? model.lastTrained.getTime() : 'none'}`} className="text-center p-3 bg-muted/50 rounded-lg relative">
            <Calendar className="h-4 w-4 mx-auto mb-1 text-muted-foreground" />
            <p key={`date-value-${model.id}-${model.lastTrained ? model.lastTrained.getTime() : 'none'}`} className="text-sm font-bold text-foreground break-words">
              {lastTrainedLabel}
            </p>
            <p className="text-xs text-muted-foreground">Last Trained</p>
          </div>
        </div>

        {/* Feature Importance */}
        <div>
          <p className="text-sm font-medium text-muted-foreground mb-2">Top Features</p>
          <div className="space-y-2">
            {model.featureImportance.length === 0 ? (
              <p className="text-xs text-muted-foreground">No feature weights available.</p>
            ) : model.featureImportance.slice(0, 3).map((f) => (
              <div key={f.feature} className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-20 truncate">{f.feature}</span>
                <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-primary rounded-full transition-all"
                    style={{ width: `${f.weight * 100}%` }}
                  />
                </div>
                <span className="text-xs text-foreground w-12 text-right">
                  {(f.weight * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2 pt-2">
          <Button
            variant={model.isActive ? "secondary" : "default"}
            className="flex-1"
            onClick={() => onSelect(model.id)}
            disabled={model.isActive || isTraining}
          >
            {model.isActive ? 'Currently Active' : 'Select Model'}
          </Button>
          <Button
            variant="outline"
            size="icon"
            onClick={() => onRetrain(model.id)}
            disabled={isTraining}
          >
            <RefreshCw className={cn("h-4 w-4", isTraining && "animate-spin")} />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};