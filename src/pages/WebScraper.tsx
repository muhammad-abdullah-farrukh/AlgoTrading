import { useEffect, useMemo, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Globe, Download, Play, Database, Plus, Loader2, CheckCircle, Clock, FileSpreadsheet, Brain, Layers } from 'lucide-react';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';
import api from '@/utils/api';

// Data sources configuration
interface DataSource {
  id: string;
  name: string;
  description: string;
  type: 'api' | 'scraper' | 'custom';
}

const DATA_SOURCES: DataSource[] = [
  { id: 'yahoo', name: 'Yahoo Finance', description: 'Free stock & forex data', type: 'api' },
  { id: 'alphavantage', name: 'Alpha Vantage', description: 'Premium market data API', type: 'api' },
  { id: 'ccxt', name: 'CCXT (Crypto)', description: 'Cryptocurrency exchanges', type: 'api' },
  { id: 'custom', name: 'Custom Source', description: 'User-defined data source', type: 'custom' },
];

// Dummy scraped data

// Dummy AI Model data

const symbols = [
  { value: 'EURUSD', label: 'EUR/USD', type: 'Forex' },
  { value: 'BTCUSDT', label: 'BTC/USDT', type: 'Crypto' },
  { value: 'AAPL', label: 'AAPL', type: 'Stock' },
];

const WebScraper = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('EURUSD');
  const [isScraping, setIsScraping] = useState(false);
  const [scrapedData, setScrapedData] = useState<any[]>([]);
  const [liveAppendEnabled, setLiveAppendEnabled] = useState(false);
  const [lastScrapeTime, setLastScrapeTime] = useState<Date | null>(null);
  const [scrapeHistory, setScrapeHistory] = useState<Array<{ symbol: string; date: Date; rows: number }>>([]);
  const [selectedSources, setSelectedSources] = useState<Set<string>>(new Set(['yahoo']));
  const [customSourceUrl, setCustomSourceUrl] = useState('');
  const [mlStatus, setMlStatus] = useState<any | null>(null);
  const [datasetStats, setDatasetStats] = useState<any | null>(null);

  const refreshDatasetStats = async () => {
    try {
      const stats = await api.get<any>('/api/integrity/stats');
      setDatasetStats(stats);
      const history = (stats?.metadata || [])
        .map((m: any) => ({
          symbol: String(m.symbol),
          date: new Date(m.updated_at || m.end_date || Date.now()),
          rows: Number(m.row_count || 0),
        }))
        .sort((a: any, b: any) => b.date.getTime() - a.date.getTime())
        .slice(0, 5);
      setScrapeHistory(history);
    } catch (e) {
      console.error('[WebScraper] Failed to load dataset stats:', e);
    }
  };

  const handleFlushLiveTrades = async () => {
    try {
      const res = await api.post<any>('/api/trade/live-trades/flush');
      toast.success('Live trades flushed', {
        description: res?.message || `Rows flushed: ${res?.rows_flushed ?? 0}`
      });
    } catch (error) {
      console.error('Flush live trades error:', error);
      const detail =
        (error as any)?.response?.data?.detail ||
        (error as any)?.response?.data?.message ||
        (error as any)?.message ||
        'Failed to flush live trades';
      toast.error('Flush failed', { description: `${String(detail)} (API: ${api.baseURL})` });
    }
  };

  useEffect(() => {
    (async () => {
      try {
        const status = await api.get<any>('/api/ml/status');
        setMlStatus(status);
      } catch (e) {
        console.error('[WebScraper] Failed to load ML status:', e);
      }
      await refreshDatasetStats();
    })();
  }, []);

  const modelRows = useMemo(() => {
    if (!mlStatus || !mlStatus.model_exists || !mlStatus.model_info) return [];
    const info = mlStatus.model_info;
    const lastTrained = info?.last_trained ? new Date(info.last_trained) : null;
    const accuracy = mlStatus.current_accuracy !== null && mlStatus.current_accuracy !== undefined
      ? Number(mlStatus.current_accuracy) * 100
      : null;
    return [
      {
        id: 'lr',
        name: 'Logistic Regression',
        weights: 'logistic_regression.pkl',
        metadata: JSON.stringify({
          timeframe: info?.timeframe,
          feature_count: info?.feature_count,
          sample_size: info?.sample_size,
        }),
        lastTrained,
        accuracy,
      },
    ];
  }, [mlStatus]);

  const selectedSymbolStats = useMemo(() => {
    const meta = (datasetStats?.metadata || []).find((m: any) => String(m.symbol) === String(selectedSymbol));
    return {
      rows: Number(meta?.row_count || 0),
      updatedAt: meta?.updated_at ? new Date(meta.updated_at) : null,
      startDate: meta?.start_date ? new Date(meta.start_date) : null,
      endDate: meta?.end_date ? new Date(meta.end_date) : null,
    };
  }, [datasetStats, selectedSymbol]);

  const toggleSource = (sourceId: string) => {
    setSelectedSources(prev => {
      const next = new Set(prev);
      if (next.has(sourceId)) {
        next.delete(sourceId);
      } else {
        next.add(sourceId);
      }
      return next;
    });
  };

  const handleScrape = async () => {
    if (selectedSources.size === 0) {
      toast.error('Please select at least one data source');
      return;
    }

    setIsScraping(true);
    const sourceNames = Array.from(selectedSources).map(id =>
      DATA_SOURCES.find(s => s.id === id)?.name || id
    ).join(', ');
    toast.info(`Scraping data from ${sourceNames}...`, { description: 'This may take a moment' });

    // Call backend API to scrape data
    try {
      await api.post('/api/scrape/start', {
        symbol: selectedSymbol,
        sources: Array.from(selectedSources),
        source_params: {}
      });

      // Wait for scraping to complete (poll status)
      await new Promise(resolve => setTimeout(resolve, 3000));

      // Fetch scraped data status
      const statusData = await api.get<any>(`/api/scrape/status/${selectedSymbol}`);

      // Fetch actual OHLCV data
      try {
        const ohlcvData = await api.get<any>('/api/indicators/ohlcv', {
          params: {
            symbol: selectedSymbol,
            timeframe: '1d',
            limit: 100
          }
        });
        if (ohlcvData && ohlcvData.ohlcv && ohlcvData.ohlcv.length > 0) {
          const formattedData = ohlcvData.ohlcv.map((d: any) => ({
            date: new Date(d.timestamp).toISOString().split('T')[0],
            open: d.open.toFixed(5),
            high: d.high.toFixed(5),
            low: d.low.toFixed(5),
            close: d.close.toFixed(5),
            volume: d.volume
          }));

          setScrapedData(formattedData.slice(0, 50)); // Show first 50 rows in preview
          setLastScrapeTime(new Date());
          await refreshDatasetStats();

          toast.success(`Successfully scraped ${statusData.total_rows || formattedData.length} rows for ${selectedSymbol}`, {
            description: `Showing first 50 rows. Total: ${statusData.total_rows || formattedData.length}`
          });
        } else {
          toast.info(`Scraping initiated for ${selectedSymbol}`, {
            description: `Total rows in database: ${statusData.total_rows || 0}`
          });
        }
      } catch (ohlcvError) {
        console.error('OHLCV fetch error:', ohlcvError);
        toast.info(`Scraping completed for ${selectedSymbol}`, {
          description: `Total rows: ${statusData.total_rows || 0}`
        });
      }
    } catch (error) {
      console.error('Scraping error:', error);
      const detail =
        (error as any)?.response?.data?.detail ||
        (error as any)?.response?.data?.message ||
        (error as any)?.message ||
        'Could not connect to backend or fetch data';
      toast.error('Scraping failed', {
        description: `${String(detail)} (API: ${api.baseURL})`
      });
    } finally {
      setIsScraping(false);
    }
  };

  const handleDownloadCSV = () => {
    if (scrapedData.length === 0) {
      toast.error('No data to download. Run a scrape first.');
      return;
    }

    const headers = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'];
    const csvContent = [
      headers.join(','),
      ...scrapedData.map(row =>
        [row.date, row.open, row.high, row.low, row.close, row.volume].join(',')
      ),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${selectedSymbol}_scraped_data_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    toast.success('CSV file downloaded successfully');
  };

  const handleDownloadModelCSV = async () => {
    try {
      // Call backend to export real model data
      const result = await api.post<any>('/api/ml/export');

      toast.success('Model data exported successfully', {
        description: `Real model weights and metadata saved to Pipeline folder`
      });
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Model export failed', {
        description: 'Ensure model is trained first. Run: python train_model.py 1d'
      });
    }
  };

  const handleDownloadModelXLSX = async () => {
    try {
      const response = await fetch(`${api.baseURL}/api/ml/export/xlsx`);
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Export failed: ${response.status}`);
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ai-model-export.xlsx`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      toast.success('Excel file downloaded successfully');
    } catch (error) {
      console.error('XLSX export error:', error);
      toast.error('Excel export failed', {
        description: (error as any)?.message || 'Failed to export Excel file'
      });
    }
  };

  const handleAppendLiveData = async () => {
    try {
      const res = await api.post<any>(`/api/scrape/append-live/${selectedSymbol}`);
      toast.success('Live data append finished', {
        description: res?.message || `Rows appended: ${res?.rows_appended ?? 0}`
      });
      await refreshDatasetStats();
    } catch (error) {
      console.error('Append live data error:', error);
      const detail =
        (error as any)?.response?.data?.detail ||
        (error as any)?.response?.data?.message ||
        (error as any)?.message ||
        'Failed to append live data';
      toast.error('Append live data failed', { description: String(detail) });
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground flex items-center gap-2">
            <Globe className="h-8 w-8 text-ai-purple" />
            Web Scraper & Data Pipeline
          </h1>
          <p className="text-muted-foreground mt-1">
            Collect market data and build training datasets
          </p>
        </div>
        <Badge variant="outline" className="w-fit">
          {liveAppendEnabled ? (
            <span className="flex items-center gap-1 text-ai-purple">
              <span className="h-2 w-2 rounded-full bg-ai-purple animate-pulse" />
              Live Append Active
            </span>
          ) : (
            <span className="text-muted-foreground">Live Append Disabled</span>
          )}
        </Badge>
      </div>

      {/* Data Source Selection */}
      <Card className="bg-card border-border card-hover">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5 text-ai-purple" />
            Data Source Selection
          </CardTitle>
          <CardDescription>Choose one or more data sources for scraping</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {DATA_SOURCES.map((source) => (
              <div
                key={source.id}
                onClick={() => source.id !== 'custom' && toggleSource(source.id)}
                className={cn(
                  "p-4 rounded-lg border cursor-pointer transition-all duration-200",
                  selectedSources.has(source.id)
                    ? "bg-ai-purple/10 border-ai-purple/30"
                    : "bg-muted/30 border-border hover:bg-muted/50"
                )}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="font-medium text-foreground">{source.name}</p>
                    <p className="text-xs text-muted-foreground mt-1">{source.description}</p>
                  </div>
                  <Checkbox
                    checked={selectedSources.has(source.id)}
                    onCheckedChange={() => toggleSource(source.id)}
                    className="mt-0.5"
                  />
                </div>
                {source.id === 'custom' && selectedSources.has('custom') && (
                  <div className="mt-3" onClick={(e) => e.stopPropagation()}>
                    <Input
                      placeholder="Enter custom URL..."
                      value={customSourceUrl}
                      onChange={(e) => setCustomSourceUrl(e.target.value)}
                      className="bg-secondary border-0 text-sm"
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
          {selectedSources.size > 0 && (
            <div className="mt-4 flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Selected:</span>
              {Array.from(selectedSources).map(id => (
                <Badge key={id} variant="secondary" className="bg-ai-purple/20 text-ai-purple">
                  {DATA_SOURCES.find(s => s.id === id)?.name}
                </Badge>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Main Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Scraping Controls */}
        <Card className="lg:col-span-2 card-hover">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Data Scraping
            </CardTitle>
            <CardDescription>Select a symbol and trigger web scraping</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="flex-1 space-y-2">
                <Label>Select Symbol</Label>
                <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
                  <SelectTrigger className="bg-secondary border-0">
                    <SelectValue placeholder="Select symbol" />
                  </SelectTrigger>
                  <SelectContent>
                    {symbols.map(symbol => (
                      <SelectItem key={symbol.value} value={symbol.value}>
                        <span className="flex items-center gap-2">
                          {symbol.label}
                          <Badge variant="secondary" className="text-xs">
                            {symbol.type}
                          </Badge>
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex gap-2 sm:items-end">
                <Button
                  onClick={handleScrape}
                  disabled={isScraping || selectedSources.size === 0}
                  className="flex-1 sm:flex-none btn-glow bg-ai-purple hover:bg-ai-purple/90"
                >
                  {isScraping ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Scraping...
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 mr-2" />
                      Start Scrape
                    </>
                  )}
                </Button>
                <Button
                  variant="outline"
                  onClick={handleDownloadCSV}
                  disabled={scrapedData.length === 0}
                  className="btn-hover"
                >
                  <Download className="h-4 w-4 mr-2" />
                  Download CSV
                </Button>
              </div>
            </div>

            {/* Scrape Status */}
            {lastScrapeTime && (
              <div className="flex items-center gap-4 p-3 bg-muted/50 rounded-lg animate-fade-in">
                <CheckCircle className="h-5 w-5 text-ai-purple" />
                <div className="flex-1">
                  <p className="text-sm font-medium">Last scrape completed</p>
                  <p className="text-xs text-muted-foreground">
                    {selectedSymbol} • {scrapedData.length} rows • {lastScrapeTime.toLocaleString()}
                  </p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Live Data Append */}
        <Card className="card-hover">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Plus className="h-5 w-5" />
              Live Data Append
            </CardTitle>
            <CardDescription>Append recent OHLCV bars from MT5 into the dataset</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="live-append" className="text-sm font-medium">
                  Auto Append
                </Label>
                <p className="text-xs text-muted-foreground">
                  Not available (no backend scheduler)
                </p>
              </div>
              <Switch
                id="live-append"
                checked={liveAppendEnabled}
                onCheckedChange={(checked) => {
                  setLiveAppendEnabled(checked);
                  toast.info('Auto append is disabled. Use "Append Live Data Now".');
                }}
                disabled
              />
            </div>

            <Button
              variant="secondary"
              className="w-full btn-hover"
              onClick={handleAppendLiveData}
            >
              <Plus className="h-4 w-4 mr-2" />
              Append Live Data Now
            </Button>

            <Button
              variant="outline"
              className="w-full btn-hover"
              onClick={handleFlushLiveTrades}
            >
              <Database className="h-4 w-4 mr-2" />
              Flush Live Trades to Queue
            </Button>

            <div className="p-3 bg-muted/50 rounded-lg">
              <p className="text-xs text-muted-foreground">
                Dataset rows (selected symbol): {selectedSymbolStats.rows}
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Last metadata update: {selectedSymbolStats.updatedAt ? selectedSymbolStats.updatedAt.toLocaleString() : '--'}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* AI Model Data Download */}
      <Card className="card-hover">
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-ai-purple" />
                AI Model Data Download
              </CardTitle>
              <CardDescription>Export model weights and metadata for all AI models</CardDescription>
            </div>
            <div className="flex gap-2 flex-wrap">
              <Button onClick={handleDownloadModelCSV} className="btn-glow bg-ai-purple hover:bg-ai-purple/90">
                <Download className="h-4 w-4 mr-2" />
                Export All Models CSV
              </Button>
              <Button variant="outline" onClick={handleDownloadModelXLSX} className="btn-hover">
                <Download className="h-4 w-4 mr-2" />
                Download Excel (.xlsx)
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="rounded-md border overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model Name</TableHead>
                  <TableHead>Weights File</TableHead>
                  <TableHead className="hidden md:table-cell">Metadata</TableHead>
                  <TableHead>Accuracy</TableHead>
                  <TableHead className="text-right">Last Trained</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {modelRows.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} className="text-muted-foreground">
                      No trained model found. Train a model first (e.g., run: python train_model.py 1d).
                    </TableCell>
                  </TableRow>
                ) : modelRows.map((model) => (
                  <TableRow key={model.id} className="table-row-hover">
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-2 rounded-full bg-ai-purple" />
                        {model.name}
                      </div>
                    </TableCell>
                    <TableCell>
                      <code className="text-xs bg-muted px-2 py-1 rounded font-mono">
                        {model.weights}
                      </code>
                    </TableCell>
                    <TableCell className="hidden md:table-cell">
                      <code className="text-xs text-muted-foreground font-mono">
                        {model.metadata}
                      </code>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className="bg-ai-purple/10 text-ai-purple border-0">
                        {model.accuracy !== null && model.accuracy !== undefined ? `${model.accuracy}%` : '--'}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right text-muted-foreground">
                      {model.lastTrained ? model.lastTrained.toLocaleDateString() : '--'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Scraped Data Preview */}
      {scrapedData.length > 0 && (
        <Card className="card-hover animate-fade-in">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileSpreadsheet className="h-5 w-5" />
              Scraped Data Preview
              <Badge variant="secondary">{scrapedData.length} rows</Badge>
            </CardTitle>
            <CardDescription>
              Showing first 10 rows of {selectedSymbol} data
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Date</TableHead>
                    <TableHead className="text-right">Open</TableHead>
                    <TableHead className="text-right">High</TableHead>
                    <TableHead className="text-right">Low</TableHead>
                    <TableHead className="text-right">Close</TableHead>
                    <TableHead className="text-right">Volume</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {scrapedData.slice(0, 10).map((row, idx) => (
                    <TableRow key={idx} className="table-row-hover">
                      <TableCell className="font-medium">{row.date}</TableCell>
                      <TableCell className="text-right">{row.open}</TableCell>
                      <TableCell className="text-right">{row.high}</TableCell>
                      <TableCell className="text-right">{row.low}</TableCell>
                      <TableCell className="text-right">{row.close}</TableCell>
                      <TableCell className="text-right">{row.volume.toLocaleString()}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Scrape History */}
      <Card className="card-hover">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Scrape History
          </CardTitle>
          <CardDescription>Recent data collection activity</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {scrapeHistory.length === 0 ? (
              <div className="p-3 bg-muted/50 rounded-lg">
                <p className="text-sm text-muted-foreground">No dataset metadata found yet.</p>
              </div>
            ) : scrapeHistory.map((item, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-3 bg-muted/50 rounded-lg transition-all duration-200 hover:bg-muted/80"
              >
                <div className="flex items-center gap-3">
                  <Badge variant="outline">{item.symbol}</Badge>
                  <span className="text-sm text-muted-foreground">
                    {item.date.toLocaleString()}
                  </span>
                </div>
                <span className="text-sm font-medium">{item.rows} rows</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default WebScraper;