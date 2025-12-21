import { useEffect, useMemo, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { StrategyCard, type Strategy } from '@/components/trading/StrategyCard';
import { Plus, Sparkles, AlertCircle } from 'lucide-react';
import { toast } from '@/hooks/use-toast';
import { TradingFactors } from '@/components/trading/TradingFactors';
import api from '@/utils/api';

const Strategies = () => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [editingStrategy, setEditingStrategy] = useState<Strategy | null>(null);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    type: 'technical' as Strategy['strategy_type']
  });

  const refreshStrategies = async () => {
    setIsLoading(true);
    try {
      const rows = await api.get<Strategy[]>('/api/strategies');
      setStrategies(Array.isArray(rows) ? rows : []);
    } catch (e) {
      console.error('[Strategies] Failed to load strategies:', e);
      toast({
        title: 'Failed to load strategies',
        description: 'Backend did not return strategies.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    refreshStrategies();
  }, []);

  const handleToggle = async (id: string, enabled: boolean) => {
    try {
      await api.put(`/api/strategies/${id}`, { enabled });
      setStrategies((prev) => prev.map((s) => (String(s.id) === String(id) ? { ...s, enabled } : s)));
      toast({
        title: enabled ? 'Strategy Enabled' : 'Strategy Disabled',
        description: `Strategy has been ${enabled ? 'activated' : 'deactivated'}.`,
      });
    } catch (e) {
      console.error('[Strategies] Toggle failed:', e);
      toast({
        title: 'Update failed',
        description: 'Could not persist strategy state to backend.',
        variant: 'destructive',
      });
      await refreshStrategies();
    }
  };

  const handleEdit = (id: string) => {
    const strategy = strategies.find(s => String(s.id) === String(id));
    if (strategy) {
      setEditingStrategy(strategy);
      setFormData({
        name: strategy.name,
        description: strategy.description || '',
        type: strategy.strategy_type
      });
      setIsDialogOpen(true);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await api.delete(`/api/strategies/${id}`);
      setStrategies((prev) => prev.filter((s) => String(s.id) !== String(id)));
      toast({
        title: 'Strategy Deleted',
        description: 'The strategy has been removed.',
        variant: 'destructive',
      });
    } catch (e) {
      console.error('[Strategies] Delete failed:', e);
      toast({
        title: 'Delete failed',
        description: 'Could not delete strategy from backend.',
        variant: 'destructive',
      });
    }
  };

  const handleSave = async () => {
    if (!formData.name || !formData.description) {
      toast({
        title: "Missing Fields",
        description: "Please fill in all required fields.",
        variant: "destructive"
      });
      return;
    }

    if (editingStrategy) {
      try {
        const updated = await api.put<Strategy>(`/api/strategies/${editingStrategy.id}`, {
          name: formData.name,
          description: formData.description,
          strategy_type: formData.type,
        });
        setStrategies((prev) => prev.map((s) => (s.id === editingStrategy.id ? updated : s)));
        toast({ title: 'Strategy Updated' });
      } catch (e) {
        console.error('[Strategies] Update failed:', e);
        toast({
          title: 'Update failed',
          description: 'Could not save strategy to backend.',
          variant: 'destructive',
        });
        await refreshStrategies();
        return;
      }
    } else {
      try {
        const created = await api.post<Strategy>('/api/strategies', {
          name: formData.name,
          description: formData.description,
          strategy_type: formData.type,
          enabled: false,
          parameters: {},
        });
        setStrategies((prev) => [...prev, created]);
        toast({ title: 'Strategy Created' });
      } catch (e) {
        console.error('[Strategies] Create failed:', e);
        toast({
          title: 'Create failed',
          description: 'Could not create strategy in backend.',
          variant: 'destructive',
        });
        return;
      }
    }

    setIsDialogOpen(false);
    setEditingStrategy(null);
    setFormData({ name: '', description: '', type: 'technical' });
  };

  const openNewDialog = () => {
    setEditingStrategy(null);
    setFormData({ name: '', description: '', type: 'technical' });
    setIsDialogOpen(true);
  };

  const enabledCount = useMemo(() => strategies.filter(s => s.enabled).length, [strategies]);
  const totalPerformance = useMemo(() => strategies.reduce((sum, s) => sum + (s.performance || 0), 0), [strategies]);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Strategy Management</h1>
          <p className="text-muted-foreground">Create and manage your trading strategies</p>
        </div>
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button onClick={openNewDialog}>
              <Plus className="h-4 w-4 mr-2" />
              New Strategy
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-card border-border">
            <DialogHeader>
              <DialogTitle>
                {editingStrategy ? 'Edit Strategy' : 'Create New Strategy'}
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="name">Strategy Name</Label>
                <Input
                  id="name"
                  placeholder="e.g., RSI Reversal"
                  value={formData.name}
                  onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="description">Description</Label>
                <Textarea
                  id="description"
                  placeholder="Describe how this strategy works..."
                  value={formData.description}
                  onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                  rows={3}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="type">Strategy Type</Label>
                <Select 
                  value={formData.type} 
                  onValueChange={(value: Strategy['strategy_type']) => setFormData(prev => ({ ...prev, type: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="technical">Technical Analysis</SelectItem>
                    <SelectItem value="ai">AI/Machine Learning</SelectItem>
                    <SelectItem value="custom">Custom Strategy</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleSave}>
                {editingStrategy ? 'Update' : 'Create'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card className="bg-card border-border">
          <CardContent className="pt-4">
            <p className="text-sm text-muted-foreground">Total Strategies</p>
            <p className="text-2xl font-bold text-foreground">{strategies.length}</p>
          </CardContent>
        </Card>
        <Card className="bg-card border-border">
          <CardContent className="pt-4">
            <p className="text-sm text-muted-foreground">Active Strategies</p>
            <p className="text-2xl font-bold text-primary">{enabledCount}</p>
          </CardContent>
        </Card>
        <Card className="bg-card border-border">
          <CardContent className="pt-4">
            <p className="text-sm text-muted-foreground">Combined Performance</p>
            <p className={`text-2xl font-bold ${totalPerformance >= 0 ? 'text-success' : 'text-destructive'}`}>
              {totalPerformance >= 0 ? '+' : ''}{totalPerformance.toFixed(1)}%
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Trading Factors & Risk Controls */}
      <TradingFactors />

      {/* AI Recommendations - Now Active */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-ai-purple" />
            AI Strategy Recommendations
          </CardTitle>
          <CardDescription>
            Machine learning-powered strategy suggestions based on current market conditions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="p-4 rounded-lg bg-muted/40 border border-border">
            <div className="flex items-start gap-3">
              <AlertCircle className="h-5 w-5 text-muted-foreground mt-0.5" />
              <div className="flex-1">
                <p className="font-medium text-foreground">No recommendations available</p>
                <p className="text-sm text-muted-foreground mt-1">
                  AI recommendations require a backend recommendations endpoint. This panel will populate when available.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Strategy Grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {isLoading ? (
          <Card className="bg-card border-border">
            <CardContent className="pt-4">
              <p className="text-sm text-muted-foreground">Loading strategies...</p>
            </CardContent>
          </Card>
        ) : strategies.length === 0 ? (
          <Card className="bg-card border-border">
            <CardContent className="pt-4">
              <p className="text-sm text-muted-foreground">No strategies found.</p>
            </CardContent>
          </Card>
        ) : strategies.map(strategy => (
          <StrategyCard
            key={strategy.id}
            strategy={strategy}
            onToggle={handleToggle}
            onEdit={handleEdit}
            onDelete={handleDelete}
          />
        ))}
      </div>
    </div>
  );
};

export default Strategies;
