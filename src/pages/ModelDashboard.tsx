import { useState, useMemo, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Brain, Settings, Zap, TrendingUp, Database, Clock, Shield, RotateCcw, Loader2 } from 'lucide-react';
import { ModelCard, MLModel } from '@/components/trading/ModelCard';
import { ModelPerformanceChart } from '@/components/trading/ModelPerformanceChart';
import { toast } from 'sonner';

// API base URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Dummy performance data (kept for chart, as backend doesn't provide historical performance yet)
const generatePerformanceData = (accuracyPct = 0) => {
  const now = new Date();
  const prev = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
  const acc = Number.isFinite(accuracyPct) ? accuracyPct : 0;
  const loss = Math.max(0, Math.min(1, 1 - acc / 100));
  return [
    { date: prev.toISOString(), accuracy: Math.max(0, Math.min(100, acc)), loss },
    { date: now.toISOString(), accuracy: Math.max(0, Math.min(100, acc)), loss },
  ];
};

const ModelDashboard = () => {
  const [models, setModels] = useState<MLModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [performanceData, setPerformanceData] = useState<Array<{ date: string; accuracy: number; loss: number }>>([]);
  const [onlineLearning, setOnlineLearning] = useState(false);
  const [autoRetrain, setAutoRetrain] = useState(false);
  const [resumeFromState, setResumeFromState] = useState(true);
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(32);
  const [accuracyThreshold, setAccuracyThreshold] = useState([75]);
  const [retrainKey, setRetrainKey] = useState(0); // Key to force chart re-render

  // Fetch real model data from API
  useEffect(() => {
    const fetchModelData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE_URL}/api/ml/status`);
        if (!response.ok) {
          throw new Error('Failed to fetch model status');
        }
        const data = await response.json();

        // Fetch feature importance (weights) if model exists
        let featureImportance: { feature: string; weight: number }[] = [];
        if (data.model_exists) {
          try {
            const weightsResponse = await fetch(`${API_BASE_URL}/api/ml/weights`);
            if (weightsResponse.ok) {
              const weightsData = await weightsResponse.json();
              // Normalize weights to 0-1 range for display
              const maxAbsWeight = Math.max(...weightsData.features.map((f: any) => Math.abs(f.abs_weight)));
              featureImportance = weightsData.features.slice(0, 6).map((f: any) => ({
                feature: f.feature.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()),
                weight: maxAbsWeight > 0 ? Math.abs(f.abs_weight) / maxAbsWeight : 0
              }));
            }
          } catch (error) {
            console.warn('Failed to fetch feature weights:', error);
            featureImportance = [];
          }
        }

        // Convert API response to MLModel format
        const logisticRegressionModel: MLModel = {
          id: 'model-lr',
          name: 'Logistic Regression',
          type: 'logistic_regression',
          accuracy: data.current_accuracy ? data.current_accuracy * 100 : 0, // Convert to percentage
          lastTrained: data.model_info?.last_trained
            ? new Date(data.model_info.last_trained)
            : null,
          datasetSize: data.model_info?.sample_size || 0,
          isActive: data.model_exists || false,
          status: data.model_exists ? 'ready' : 'error',
          featureImportance: featureImportance
        };

        setModels(data.model_exists ? [logisticRegressionModel] : []);
        setOnlineLearning(data.online_learning_enabled || false);
        setAutoRetrain(data.auto_retrain_enabled || false);

        try {
          const perfRes = await fetch(`${API_BASE_URL}/api/ml/performance`);
          if (perfRes.ok) {
            const perf = await perfRes.json();
            const arr = Array.isArray(perf?.performance) ? perf.performance : [];
            setPerformanceData(arr.length ? arr : generatePerformanceData(logisticRegressionModel.accuracy));
          } else {
            setPerformanceData(generatePerformanceData(logisticRegressionModel.accuracy));
          }
        } catch (e) {
          setPerformanceData(generatePerformanceData(logisticRegressionModel.accuracy));
        }
      } catch (error) {
        console.error('Error fetching model data:', error);
        toast.error('Failed to load model data');
        // Fallback to empty models
        setModels([]);
        setPerformanceData(generatePerformanceData(0));
      } finally {
        setLoading(false);
      }
    };

    fetchModelData();
  }, [retrainKey]);

  const activeModel = models.find(m => m.isActive);

  const featureData = useMemo(() => {
    return activeModel?.featureImportance.map(f => ({
      feature: f.feature,
      importance: f.weight * 100
    })) || [];
  }, [activeModel?.featureImportance]);

  const handleSelectModel = (id: string) => {
    setModels(prev => prev.map(m => ({
      ...m,
      isActive: m.id === id
    })));
    const model = models.find(m => m.id === id);
    toast.success(`${model?.name} is now the active model for predictions`);
  };

  const handleRetrain = async (id: string) => {
    const model = models.find(m => m.id === id);
    if (!model) return;

    // Only logistic regression is implemented
    if (id !== 'model-lr') {
      toast.error('Only Logistic Regression model is currently available for retraining');
      return;
    }

    setModels(prev => prev.map(m =>
      m.id === id ? { ...m, status: 'training' as const } : m
    ));
    toast.info(`Retraining ${model.name}...`, { description: 'This may take a few minutes' });

    try {
      const response = await fetch(`${API_BASE_URL}/api/ml/retrain`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          timeframe: '1d',
          force: true
        })
      });

      if (!response.ok) {
        throw new Error('Retraining failed');
      }

      const result = await response.json();

      if (result.status === 'success') {
        // Refresh model data
        setRetrainKey(prev => prev + 1);
        toast.success(`${model.name} retraining complete!`, {
          description: `New accuracy: ${(result.accuracy * 100).toFixed(2)}%`
        });
      } else {
        throw new Error(result.message || 'Retraining failed');
      }
    } catch (error) {
      console.error('Retraining error:', error);
      setModels(prev => prev.map(m =>
        m.id === id ? { ...m, status: 'error' as const } : m
      ));
      toast.error(`Failed to retrain ${model.name}`);
    }
  };

  const handleRetrainAll = async () => {
    toast.info('Retraining all models...', { description: 'This will update all models with latest data' });
    setModels(prev => prev.map(m => ({ ...m, status: 'training' as const })));

    try {
      // Only retrain logistic regression (only implemented model)
      const lrModel = models.find(m => m.id === 'model-lr');
      if (lrModel) {
        await handleRetrain('model-lr');
      }

      // Refresh all model data
      setRetrainKey(prev => prev + 1);
      toast.success('Model retraining initiated!');
    } catch (error) {
      console.error('Retrain all error:', error);
      setModels(prev => prev.map(m => ({ ...m, status: 'error' as const })));
      toast.error('Failed to retrain models');
    }
  };

  // Summary stats (only count models with data)
  const activeModels = models.filter(m => m.status === 'ready' && m.accuracy > 0);
  const avgAccuracy = activeModels.length > 0
    ? activeModels.reduce((sum, m) => sum + m.accuracy, 0) / activeModels.length
    : 0;
  const totalDataset = models.find(m => m.id === 'model-lr')?.datasetSize || 0;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center space-y-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto" />
          <p className="text-muted-foreground">Loading model data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground flex items-center gap-2">
            <Brain className="h-8 w-8 text-primary" />
            Model Dashboard
          </h1>
          <p className="text-muted-foreground mt-1">
            Manage and monitor your AI/ML trading models
          </p>
        </div>
        <Button onClick={handleRetrainAll} variant="outline" className="btn-hover">
          <Zap className="h-4 w-4 mr-2" />
          Retrain All Models
        </Button>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Brain className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-2xl font-bold">{models.length}</p>
                <p className="text-sm text-muted-foreground">Total Models</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <TrendingUp className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-2xl font-bold">{avgAccuracy.toFixed(1)}%</p>
                <p className="text-sm text-muted-foreground">Avg Accuracy</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Database className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-2xl font-bold">{(totalDataset / 1000).toFixed(0)}k</p>
                <p className="text-sm text-muted-foreground">Training Samples</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Clock className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-2xl font-bold">{activeModel?.name.split(' ')[0] || 'None'}</p>
                <p className="text-sm text-muted-foreground">Active Model</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Cards */}
        <div className="lg:col-span-2 space-y-4">
          <h2 className="text-xl font-semibold text-foreground">Available Models</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {models.length === 0 ? (
              <Card>
                <CardContent className="pt-6">
                  <p className="text-sm text-muted-foreground">No trained model found.</p>
                </CardContent>
              </Card>
            ) : models.map(model => (
              <ModelCard
                key={`${model.id}-${model.lastTrained ? model.lastTrained.getTime() : 'none'}-${model.accuracy}-${retrainKey}`}
                model={model}
                onSelect={handleSelectModel}
                onRetrain={handleRetrain}
              />
            ))}
          </div>
        </div>

        {/* Training Controls & Model Continuity */}
        <div className="space-y-4">
          <h2 className="text-xl font-semibold text-foreground">Training Settings</h2>
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Settings className="h-4 w-4" />
                Training Pipeline
              </CardTitle>
              <CardDescription>Configure model training parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Online Learning Toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="online-learning" className="text-sm font-medium">
                    Online Learning
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    Continuously update model with new data
                  </p>
                </div>
                <Switch
                  id="online-learning"
                  checked={onlineLearning}
                  onCheckedChange={() => toast.info('Online learning toggle is read-only (configure server-side).')}
                  disabled
                />
              </div>

              {/* Epochs */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label className="text-sm">Epochs</Label>
                  <span className="text-sm text-muted-foreground">{epochs}</span>
                </div>
                <Input
                  type="number"
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                  min={10}
                  max={500}
                />
              </div>

              {/* Batch Size */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label className="text-sm">Batch Size</Label>
                  <span className="text-sm text-muted-foreground">{batchSize}</span>
                </div>
                <Input
                  type="number"
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  min={8}
                  max={256}
                  step={8}
                />
              </div>

              {/* Accuracy Threshold */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label className="text-sm">Accuracy Threshold</Label>
                  <span className="text-sm text-muted-foreground">{accuracyThreshold[0]}%</span>
                </div>
                <Slider
                  value={accuracyThreshold}
                  onValueChange={setAccuracyThreshold}
                  min={50}
                  max={95}
                  step={1}
                />
                <p className="text-xs text-muted-foreground">
                  Minimum accuracy required before model deployment
                </p>
              </div>

              {/* Status */}
              <div className="pt-2 border-t border-border">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Pipeline Status</span>
                  <Badge variant="outline" className="bg-primary/10 text-primary">
                    Ready
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Model Continuity Section */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Shield className="h-4 w-4" />
                Model Continuity
              </CardTitle>
              <CardDescription>Control model state persistence and retraining behavior</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Resume from Last State */}
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="resume-state" className="text-sm font-medium">
                    Resume from Last State
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    Models continue from saved checkpoint
                  </p>
                </div>
                <Switch
                  id="resume-state"
                  checked={resumeFromState}
                  onCheckedChange={setResumeFromState}
                />
              </div>

              {/* Auto Retrain Toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="auto-retrain" className="text-sm font-medium">
                    Auto Retraining
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    {autoRetrain ? 'Retrain when accuracy drops below threshold' : 'Manual retraining only'}
                  </p>
                </div>
                <Switch
                  id="auto-retrain"
                  checked={autoRetrain}
                  onCheckedChange={(checked) => {
                    toast.info('Auto retrain toggle is read-only (configure server-side).');
                  }}
                  disabled
                />
              </div>

              {/* Last Checkpoint Info */}
              <div className="p-3 bg-muted/50 rounded-lg space-y-2">
                <div className="flex items-center gap-2 text-sm">
                  <RotateCcw className="h-4 w-4 text-muted-foreground" />
                  <span className="text-muted-foreground">Last Checkpoint</span>
                </div>
                <p className="text-sm font-medium">
                  {activeModel?.lastTrained ? activeModel.lastTrained.toLocaleString() : '--'}
                </p>
                <p className="text-xs text-muted-foreground">
                  All models saved • {(totalDataset / 1000).toFixed(0)}k samples processed
                </p>
              </div>

              {/* Continuity Status */}
              <div className="pt-2 border-t border-border">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">State Persistence</span>
                  <Badge variant="outline" className={resumeFromState ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"}>
                    {resumeFromState ? 'Active' : 'Disabled'}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Performance Chart */}
      {activeModel && (
        <ModelPerformanceChart
          key={`chart-${activeModel.id}-${retrainKey}-${activeModel.lastTrained?.getTime()}`}
          performanceData={performanceData}
          featureData={featureData}
          modelName={activeModel.name}
        />
      )}
    </div>
  );
};

export default ModelDashboard;