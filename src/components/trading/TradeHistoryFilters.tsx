import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { RotateCcw, Filter } from 'lucide-react';

export interface TradeFilters {
  side: string;
  profitLoss: string;
  status: string;
  symbol: string;
  dateFrom: string;
  dateTo: string;
}

interface TradeHistoryFiltersProps {
  filters: TradeFilters;
  onFilterChange: (filters: TradeFilters) => void;
  onReset: () => void;
}

export const TradeHistoryFilters = ({ filters, onFilterChange, onReset }: TradeHistoryFiltersProps) => {
  const updateFilter = (key: keyof TradeFilters, value: string) => {
    onFilterChange({ ...filters, [key]: value });
  };

  return (
    <Card className="bg-card border-border">
      <CardContent className="pt-4">
        <div className="flex items-center gap-2 mb-4">
          <Filter className="h-4 w-4 text-primary" />
          <span className="font-medium text-foreground">Filters</span>
        </div>
        
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7">
          {/* Side */}
          <div className="space-y-1.5">
            <Label className="text-xs">Side</Label>
            <Select value={filters.side} onValueChange={(v) => updateFilter('side', v)}>
              <SelectTrigger className="h-9">
                <SelectValue placeholder="All" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All</SelectItem>
                <SelectItem value="buy">Buy</SelectItem>
                <SelectItem value="sell">Sell</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Profit/Loss */}
          <div className="space-y-1.5">
            <Label className="text-xs">P/L</Label>
            <Select value={filters.profitLoss} onValueChange={(v) => updateFilter('profitLoss', v)}>
              <SelectTrigger className="h-9">
                <SelectValue placeholder="All" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All</SelectItem>
                <SelectItem value="profit">Profit Only</SelectItem>
                <SelectItem value="loss">Loss Only</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Status */}
          <div className="space-y-1.5">
            <Label className="text-xs">Status</Label>
            <Select value={filters.status} onValueChange={(v) => updateFilter('status', v)}>
              <SelectTrigger className="h-9">
                <SelectValue placeholder="All" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="cancelled">Cancelled</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Symbol */}
          <div className="space-y-1.5">
            <Label className="text-xs">Symbol</Label>
            <Select value={filters.symbol} onValueChange={(v) => updateFilter('symbol', v)}>
              <SelectTrigger className="h-9">
                <SelectValue placeholder="All" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Symbols</SelectItem>
                <SelectItem value="EURUSD">EURUSD</SelectItem>
                <SelectItem value="GBPUSD">GBPUSD</SelectItem>
                <SelectItem value="USDJPY">USDJPY</SelectItem>
                <SelectItem value="AUDUSD">AUDUSD</SelectItem>
                <SelectItem value="USDCAD">USDCAD</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Date From */}
          <div className="space-y-1.5">
            <Label className="text-xs">From</Label>
            <Input
              type="date"
              value={filters.dateFrom}
              onChange={(e) => updateFilter('dateFrom', e.target.value)}
              className="h-9"
            />
          </div>

          {/* Date To */}
          <div className="space-y-1.5">
            <Label className="text-xs">To</Label>
            <Input
              type="date"
              value={filters.dateTo}
              onChange={(e) => updateFilter('dateTo', e.target.value)}
              className="h-9"
            />
          </div>
        </div>

        <div className="flex justify-end mt-4">
          <Button variant="outline" size="sm" onClick={onReset}>
            <RotateCcw className="h-3 w-3 mr-1" />
            Reset Filters
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};
