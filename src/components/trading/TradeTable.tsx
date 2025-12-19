import { useState, useMemo } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Trade } from '@/utils/dummyData';
import { ArrowUpDown, ChevronLeft, ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';

interface TradeTableProps {
  trades: Trade[];
  pageSize?: number;
}

type SortField = 'timestamp' | 'symbol' | 'action' | 'profitLoss';
type SortDirection = 'asc' | 'desc';

export const TradeTable = ({ trades, pageSize = 10 }: TradeTableProps) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [sortField, setSortField] = useState<SortField>('timestamp');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const sortedTrades = useMemo(() => {
    return [...trades].sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case 'timestamp':
          comparison = a.timestamp.getTime() - b.timestamp.getTime();
          break;
        case 'symbol':
          comparison = a.symbol.localeCompare(b.symbol);
          break;
        case 'action':
          comparison = a.action.localeCompare(b.action);
          break;
        case 'profitLoss':
          comparison = a.profitLoss - b.profitLoss;
          break;
      }
      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [trades, sortField, sortDirection]);

  const totalPages = Math.ceil(sortedTrades.length / pageSize);
  const paginatedTrades = sortedTrades.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );

  const totalPnL = trades.reduce((sum, t) => sum + t.profitLoss, 0);
  const winRate = (trades.filter(t => t.profitLoss > 0).length / trades.length) * 100;

  const SortHeader = ({ field, children }: { field: SortField; children: React.ReactNode }) => (
    <Button
      variant="ghost"
      size="sm"
      className="h-8 px-2 -ml-2 hover:bg-muted"
      onClick={() => handleSort(field)}
    >
      {children}
      <ArrowUpDown className="ml-2 h-4 w-4" />
    </Button>
  );

  return (
    <Card className="bg-card border-border">
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <CardTitle>Trade History</CardTitle>
          <div className="flex gap-4">
            <div className="text-sm">
              <span className="text-muted-foreground">Total P/L: </span>
              <span className={cn(
                "font-semibold",
                totalPnL >= 0 ? "text-success" : "text-destructive"
              )}>
                {totalPnL >= 0 ? '+' : ''}{totalPnL.toFixed(2)} USD
              </span>
            </div>
            <div className="text-sm">
              <span className="text-muted-foreground">Win Rate: </span>
              <span className="font-semibold text-foreground">{winRate.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border border-border overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/50 hover:bg-muted/50">
                <TableHead className="w-[120px]">Trade ID</TableHead>
                <TableHead><SortHeader field="symbol">Symbol</SortHeader></TableHead>
                <TableHead><SortHeader field="action">Action</SortHeader></TableHead>
                <TableHead className="text-right">Quantity</TableHead>
                <TableHead className="text-right">Price</TableHead>
                <TableHead><SortHeader field="timestamp">Date/Time</SortHeader></TableHead>
                <TableHead className="text-right"><SortHeader field="profitLoss">P/L</SortHeader></TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {paginatedTrades.map((trade) => (
                <TableRow key={trade.id} className="hover:bg-muted/30">
                  <TableCell className="font-mono text-sm text-muted-foreground">
                    {trade.id}
                  </TableCell>
                  <TableCell className="font-semibold">{trade.symbol}</TableCell>
                  <TableCell>
                    <Badge variant={trade.action === 'BUY' ? 'default' : 'destructive'}>
                      {trade.action}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right font-mono">
                    {trade.quantity.toFixed(2)}
                  </TableCell>
                  <TableCell className="text-right font-mono">
                    {trade.price.toFixed(5)}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {trade.timestamp.toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </TableCell>
                  <TableCell className={cn(
                    "text-right font-mono font-semibold",
                    trade.profitLoss >= 0 ? "text-success" : "text-destructive"
                  )}>
                    {trade.profitLoss >= 0 ? '+' : ''}{trade.profitLoss.toFixed(2)}
                  </TableCell>
                  <TableCell>
                    <Badge variant={trade.status === 'completed' ? 'outline' : 'secondary'}>
                      {trade.status}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        
        {/* Pagination */}
        <div className="flex items-center justify-between mt-4">
          <p className="text-sm text-muted-foreground">
            Showing {((currentPage - 1) * pageSize) + 1} to {Math.min(currentPage * pageSize, trades.length)} of {trades.length} trades
          </p>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
              disabled={currentPage === 1}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-sm">
              Page {currentPage} of {totalPages}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
              disabled={currentPage === totalPages}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
