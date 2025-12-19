import { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { TradeTable } from '@/components/trading/TradeTable';
import { Trade } from '@/utils/dummyData';
import { useTrades } from '@/contexts/TradesContext';
import { Search, Download, Filter, RotateCcw, Calendar } from 'lucide-react';
import { cn } from '@/lib/utils';

interface TradeFilters {
  tradeType: string;
  side: string;
  profitLoss: string;
  status: string;
  symbol: string;
  dateFrom: string;
  dateTo: string;
  search: string;
}

const defaultFilters: TradeFilters = {
  tradeType: 'all',
  side: 'all',
  profitLoss: 'all',
  status: 'all',
  symbol: 'all',
  dateFrom: '',
  dateTo: '',
  search: '',
};

const TradeHistory = () => {
  const { trades } = useTrades();
  const [filters, setFilters] = useState<TradeFilters>(defaultFilters);

  const symbols = useMemo(() => {
    const uniqueSymbols = [...new Set(trades.map(t => t.symbol))];
    return uniqueSymbols.sort();
  }, [trades]);

  const filteredTrades = useMemo(() => {
    return trades.filter((trade: Trade) => {
      // Search filter
      if (filters.search) {
        const searchLower = filters.search.toLowerCase();
        const matchesSearch = trade.id.toLowerCase().includes(searchLower) ||
                             trade.symbol.toLowerCase().includes(searchLower);
        if (!matchesSearch) return false;
      }
      
      // Side filter
      if (filters.side !== 'all' && trade.action.toLowerCase() !== filters.side) return false;
      
      // Profit/Loss filter
      if (filters.profitLoss === 'profit' && trade.profitLoss <= 0) return false;
      if (filters.profitLoss === 'loss' && trade.profitLoss >= 0) return false;
      
      // Status filter
      if (filters.status !== 'all' && trade.status !== filters.status) return false;
      
      // Symbol filter
      if (filters.symbol !== 'all' && trade.symbol !== filters.symbol) return false;
      
      // Date range filter
      if (filters.dateFrom && new Date(trade.timestamp) < new Date(filters.dateFrom)) return false;
      if (filters.dateTo && new Date(trade.timestamp) > new Date(filters.dateTo)) return false;
      
      return true;
    });
  }, [trades, filters]);

  const stats = useMemo(() => {
    const totalPnL = filteredTrades.reduce((sum, t) => sum + t.profitLoss, 0);
    const wins = filteredTrades.filter(t => t.profitLoss > 0).length;
    const losses = filteredTrades.filter(t => t.profitLoss < 0).length;
    return { totalPnL, wins, losses, total: filteredTrades.length };
  }, [filteredTrades]);

  const updateFilter = (key: keyof TradeFilters, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const resetFilters = () => {
    setFilters(defaultFilters);
  };

  const hasActiveFilters = Object.entries(filters).some(([key, value]) => {
    if (key === 'search') return value !== '';
    if (key === 'dateFrom' || key === 'dateTo') return value !== '';
    return value !== 'all';
  });

  const handleExport = () => {
    const csvContent = [
      'ID,Symbol,Action,Quantity,Price,Date,P/L,Status',
      ...filteredTrades.map(t => 
        `${t.id},${t.symbol},${t.action},${t.quantity},${t.price},${t.timestamp.toISOString()},${t.profitLoss},${t.status}`
      )
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'trade-history.csv';
    a.click();
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Trade History</h1>
          <p className="text-muted-foreground">View and analyze your trading activity</p>
        </div>
        <Button variant="outline" onClick={handleExport} className="btn-hover">
          <Download className="h-4 w-4 mr-2" />
          Export CSV
        </Button>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Card className="bg-card border-border card-hover">
          <CardContent className="pt-4">
            <p className="text-sm text-muted-foreground">Total Trades</p>
            <p className="text-2xl font-bold text-foreground tabular-nums">{stats.total}</p>
          </CardContent>
        </Card>
        <Card className="bg-card border-border card-hover">
          <CardContent className="pt-4">
            <p className="text-sm text-muted-foreground">Net P/L</p>
            <p className={cn(
              "text-2xl font-bold tabular-nums",
              stats.totalPnL >= 0 ? "text-success" : "text-destructive"
            )}>
              {stats.totalPnL >= 0 ? '+' : ''}${stats.totalPnL.toFixed(2)}
            </p>
          </CardContent>
        </Card>
        <Card className="bg-card border-border card-hover">
          <CardContent className="pt-4">
            <p className="text-sm text-muted-foreground">Winning Trades</p>
            <p className="text-2xl font-bold text-success tabular-nums">{stats.wins}</p>
          </CardContent>
        </Card>
        <Card className="bg-card border-border card-hover">
          <CardContent className="pt-4">
            <p className="text-sm text-muted-foreground">Losing Trades</p>
            <p className="text-2xl font-bold text-destructive tabular-nums">{stats.losses}</p>
          </CardContent>
        </Card>
      </div>

      {/* Comprehensive Filters */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base flex items-center gap-2">
              <Filter className="h-4 w-4 text-ai-purple" />
              Filters
              {hasActiveFilters && (
                <span className="text-xs text-ai-purple bg-ai-purple/10 px-2 py-0.5 rounded">Active</span>
              )}
            </CardTitle>
            {hasActiveFilters && (
              <Button variant="ghost" size="sm" onClick={resetFilters} className="text-muted-foreground hover:text-foreground">
                <RotateCcw className="h-4 w-4 mr-1" />
                Reset
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {/* Search */}
            <div className="space-y-2 sm:col-span-2">
              <Label className="text-xs text-muted-foreground">Search</Label>
              <div className="relative">
                <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search by ID or symbol..."
                  value={filters.search}
                  onChange={(e) => updateFilter('search', e.target.value)}
                  className="pl-10 bg-secondary border-0"
                />
              </div>
            </div>

            {/* Trade Type */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Trade Type</Label>
              <Select value={filters.tradeType} onValueChange={(v) => updateFilter('tradeType', v)}>
                <SelectTrigger className="bg-secondary border-0">
                  <SelectValue placeholder="All Types" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="market">Market</SelectItem>
                  <SelectItem value="limit">Limit</SelectItem>
                  <SelectItem value="stop">Stop</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Side */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Side</Label>
              <Select value={filters.side} onValueChange={(v) => updateFilter('side', v)}>
                <SelectTrigger className="bg-secondary border-0">
                  <SelectValue placeholder="All Sides" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Sides</SelectItem>
                  <SelectItem value="buy">Buy Only</SelectItem>
                  <SelectItem value="sell">Sell Only</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Profit/Loss */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Profit/Loss</Label>
              <Select value={filters.profitLoss} onValueChange={(v) => updateFilter('profitLoss', v)}>
                <SelectTrigger className="bg-secondary border-0">
                  <SelectValue placeholder="All P/L" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All P/L</SelectItem>
                  <SelectItem value="profit">Profit Only</SelectItem>
                  <SelectItem value="loss">Loss Only</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Status */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Status</Label>
              <Select value={filters.status} onValueChange={(v) => updateFilter('status', v)}>
                <SelectTrigger className="bg-secondary border-0">
                  <SelectValue placeholder="All Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="Open">Open</SelectItem>
                  <SelectItem value="Closed">Closed</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Symbol */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Symbol</Label>
              <Select value={filters.symbol} onValueChange={(v) => updateFilter('symbol', v)}>
                <SelectTrigger className="bg-secondary border-0">
                  <SelectValue placeholder="All Symbols" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Symbols</SelectItem>
                  {symbols.map(symbol => (
                    <SelectItem key={symbol} value={symbol}>{symbol}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Date From */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground flex items-center gap-1">
                <Calendar className="h-3 w-3" /> From
              </Label>
              <Input
                type="date"
                value={filters.dateFrom}
                onChange={(e) => updateFilter('dateFrom', e.target.value)}
                className="bg-secondary border-0"
              />
            </div>

            {/* Date To */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground flex items-center gap-1">
                <Calendar className="h-3 w-3" /> To
              </Label>
              <Input
                type="date"
                value={filters.dateTo}
                onChange={(e) => updateFilter('dateTo', e.target.value)}
                className="bg-secondary border-0"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Trade Table */}
      <TradeTable trades={filteredTrades} pageSize={10} />
    </div>
  );
};

export default TradeHistory;