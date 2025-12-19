import { useState } from 'react';
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
const generateDummyScrapedData = (symbol: string) => {
  const data = [];
  for (let i = 0; i < 50; i++) {
    const date = new Date(Date.now() - i * 24 * 60 * 60 * 1000);
    const basePrice = symbol === 'EURUSD' ? 1.08 : symbol === 'BTCUSDT' ? 43000 : 185;
    const variation = symbol === 'EURUSD' ? 0.02 : symbol === 'BTCUSDT' ? 2000 : 10;
    data.push({
      date: date.toISOString().split('T')[0],
      open: (basePrice + (Math.random() - 0.5) * variation).toFixed(symbol === 'EURUSD' ? 5 : 2),
      high: (basePrice + Math.random() * variation).toFixed(symbol === 'EURUSD' ? 5 : 2),
      low: (basePrice - Math.random() * variation).toFixed(symbol === 'EURUSD' ? 5 : 2),
      close: (basePrice + (Math.random() - 0.5) * variation).toFixed(symbol === 'EURUSD' ? 5 : 2),
      volume: Math.floor(Math.random() * 1000000) + 100000,
    });
  }
  return data;
};

// Dummy AI Model data
const aiModels = [
  {
    id: 'lr',
    name: 'Logistic Regression',
    weights: 'lr_weights_v2.pkl',
    metadata: '{ features: 12, regularization: L2 }',
    lastTrained: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
    accuracy: 72.5,
  },
  {
    id: 'rf',
    name: 'Random Forest',
    weights: 'rf_weights_v3.pkl',
    metadata: '{ n_estimators: 100, max_depth: 15 }',
    lastTrained: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
    accuracy: 78.3,
  },
  {
    id: 'lstm',
    name: 'LSTM Network',
    weights: 'lstm_weights_v1.h5',
    metadata: '{ layers: 3, units: 128, dropout: 0.2 }',
    lastTrained: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
    accuracy: 81.7,
  },
  {
    id: 'ensemble',
    name: 'Ensemble Model',
    weights: 'ensemble_weights_v2.pkl',
    metadata: '{ models: [LR, RF, LSTM], voting: soft }',
    lastTrained: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
    accuracy: 84.2,
  },
];

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
  const [scrapeHistory, setScrapeHistory] = useState([
    { symbol: 'EURUSD', date: new Date(Date.now() - 2 * 60 * 60 * 1000), rows: 1250 },
    { symbol: 'BTCUSDT', date: new Date(Date.now() - 24 * 60 * 60 * 1000), rows: 890 },
  ]);
  const [selectedSources, setSelectedSources] = useState<Set<string>>(new Set(['yahoo']));
  const [customSourceUrl, setCustomSourceUrl] = useState('');

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

    // Simulate scraping delay
    await new Promise(resolve => setTimeout(resolve, 2000));

    const data = generateDummyScrapedData(selectedSymbol);
    setScrapedData(data);
    setLastScrapeTime(new Date());
    setScrapeHistory(prev => [
      { symbol: selectedSymbol, date: new Date(), rows: data.length },
      ...prev.slice(0, 4),
    ]);
    setIsScraping(false);
    toast.success(`Successfully scraped ${data.length} rows for ${selectedSymbol}`);
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

  const handleDownloadModelCSV = () => {
    const headers = ['Model Name', 'Weights File', 'Metadata', 'Last Trained', 'Accuracy'];
    const csvContent = [
      headers.join(','),
      ...aiModels.map(model => 
        [
          model.name,
          model.weights,
          `"${model.metadata}"`,
          model.lastTrained.toISOString(),
          `${model.accuracy}%`
        ].join(',')
      ),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `ai_models_data_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    toast.success('AI Model data exported successfully');
  };

  const handleAppendLiveData = () => {
    if (scrapedData.length === 0) {
      toast.error('No scraped data to append to. Run a scrape first.');
      return;
    }
    
    toast.success('Live MT5/Mock data appended to dataset', {
      description: `Added ${Math.floor(Math.random() * 50) + 10} new rows`,
    });
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
            <CardDescription>Append real-time MT5/mock data to datasets</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="live-append" className="text-sm font-medium">
                  Auto Append
                </Label>
                <p className="text-xs text-muted-foreground">
                  Continuously add live data
                </p>
              </div>
              <Switch
                id="live-append"
                checked={liveAppendEnabled}
                onCheckedChange={(checked) => {
                  setLiveAppendEnabled(checked);
                  toast.info(checked ? 'Live append enabled' : 'Live append disabled');
                }}
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

            <div className="p-3 bg-muted/50 rounded-lg">
              <p className="text-xs text-muted-foreground">
                Data Source: {liveAppendEnabled ? 'MT5 Live Feed' : 'Mock Data Generator'}
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Append Interval: Every 5 minutes
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
            <Button onClick={handleDownloadModelCSV} className="btn-glow bg-ai-purple hover:bg-ai-purple/90">
              <Download className="h-4 w-4 mr-2" />
              Export All Models CSV
            </Button>
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
                {aiModels.map((model) => (
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
                        {model.accuracy}%
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right text-muted-foreground">
                      {model.lastTrained.toLocaleDateString()}
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
            {scrapeHistory.map((item, idx) => (
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