import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { StrategyCard } from '@/components/trading/StrategyCard';
import { generateStrategies, Strategy } from '@/utils/dummyData';
import { Plus, Sparkles, AlertCircle } from 'lucide-react';
import { toast } from '@/hooks/use-toast';
import { TradingFactors } from '@/components/trading/TradingFactors';

const Strategies = () => {
  const [strategies, setStrategies] = useState<Strategy[]>(() => generateStrategies());
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [editingStrategy, setEditingStrategy] = useState<Strategy | null>(null);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    type: 'technical' as Strategy['type']
  });

  const handleToggle = (id: string, enabled: boolean) => {
    setStrategies(prev => prev.map(s => 
      s.id === id ? { ...s, enabled } : s
    ));
    toast({
      title: enabled ? "Strategy Enabled" : "Strategy Disabled",
      description: `Strategy has been ${enabled ? 'activated' : 'deactivated'}.`
    });
  };

  const handleEdit = (id: string) => {
    const strategy = strategies.find(s => s.id === id);
    if (strategy) {
      setEditingStrategy(strategy);
      setFormData({
        name: strategy.name,
        description: strategy.description,
        type: strategy.type
      });
      setIsDialogOpen(true);
    }
  };

  const handleDelete = (id: string) => {
    setStrategies(prev => prev.filter(s => s.id !== id));
    toast({
      title: "Strategy Deleted",
      description: "The strategy has been removed.",
      variant: "destructive"
    });
  };

  const handleSave = () => {
    if (!formData.name || !formData.description) {
      toast({
        title: "Missing Fields",
        description: "Please fill in all required fields.",
        variant: "destructive"
      });
      return;
    }

    if (editingStrategy) {
      // Update existing
      setStrategies(prev => prev.map(s => 
        s.id === editingStrategy.id 
          ? { ...s, name: formData.name, description: formData.description, type: formData.type }
          : s
      ));
      toast({ title: "Strategy Updated" });
    } else {
      // Create new
      const newStrategy: Strategy = {
        id: `strat-${Date.now()}`,
        name: formData.name,
        description: formData.description,
        type: formData.type,
        enabled: false,
        parameters: {},
        performance: 0,
        trades: 0
      };
      setStrategies(prev => [...prev, newStrategy]);
      toast({ title: "Strategy Created" });
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

  const enabledCount = strategies.filter(s => s.enabled).length;
  const totalPerformance = strategies.reduce((sum, s) => sum + s.performance, 0);

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
                  onValueChange={(value: Strategy['type']) => setFormData(prev => ({ ...prev, type: value }))}
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

      {/* AI Recommendations Placeholder */}
      <Card className="bg-card border-border border-dashed">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            AI Strategy Recommendations
          </CardTitle>
          <CardDescription>
            Machine learning-powered strategy suggestions based on market conditions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/50">
            <AlertCircle className="h-5 w-5 text-muted-foreground" />
            <div>
              <p className="font-medium text-foreground">Backend Integration Required</p>
              <p className="text-sm text-muted-foreground">
                Connect to the Python ML backend to receive personalized strategy recommendations.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Strategy Grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {strategies.map(strategy => (
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
