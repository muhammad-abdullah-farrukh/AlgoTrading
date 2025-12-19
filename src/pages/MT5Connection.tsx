import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { StatusIndicator } from '@/components/trading/StatusIndicator';
import api from '@/utils/api';
import { Link, Server, Shield, Key, AlertCircle, CheckCircle } from 'lucide-react';
import { toast } from '@/hooks/use-toast';

const MT5Connection = () => {
  const [credentials, setCredentials] = useState({
    server: '',
    login: '',
    password: '',
    apiKey: ''
  });
  
  const [mt5Status, setMt5Status] = useState<{
    connected: boolean;
    mode: 'real' | 'mock' | 'unavailable';
    error?: string;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Check MT5 status on mount and periodically
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const status = await api.get<{
          status: string;
          components: {
            mt5?: {
              connected: boolean;
              mode: 'real' | 'mock' | 'unavailable';
              error?: string;
            };
          };
        }>('/status');
        
        if (status.components?.mt5) {
          setMt5Status({
            connected: status.components.mt5.connected,
            mode: status.components.mt5.mode,
            error: status.components.mt5.error,
          });
          setError(null);
        }
      } catch (err) {
        console.error('Failed to check MT5 status:', err);
        setError('Failed to check connection status');
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 5000); // Check every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const handleConnect = async () => {
    if (!credentials.server || !credentials.login || !credentials.password) {
      toast({
        title: "Missing Credentials",
        description: "Please fill in all required fields.",
        variant: "destructive"
      });
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Note: Backend MT5 connection is handled server-side via environment variables
      // This form is for display purposes. Real connection happens on backend startup.
      toast({
        title: "MT5 Configuration",
        description: "MT5 connection is managed server-side. Check backend logs for connection status.",
      });
      
      // Refresh status after a moment
      setTimeout(async () => {
        try {
          const status = await api.get<{
            components: {
              mt5?: {
                connected: boolean;
                mode: 'real' | 'mock' | 'unavailable';
              };
            };
          }>('/status');
          if (status.components?.mt5) {
            setMt5Status({
              connected: status.components.mt5.connected,
              mode: status.components.mt5.mode,
            });
          }
        } catch (err) {
          console.error('Failed to refresh status:', err);
        }
        setIsLoading(false);
      }, 1000);
    } catch (err) {
      setIsLoading(false);
      setError('Failed to check connection status');
      toast({
        title: "Error",
        description: "Failed to check MT5 connection status.",
        variant: "destructive"
      });
    }
  };

  const connectionStatus = isLoading 
    ? 'loading' 
    : mt5Status?.connected 
      ? 'connected' 
      : mt5Status?.mode === 'mock' 
        ? 'mock' 
        : 'disconnected';

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground">MT5 Connection</h1>
        <p className="text-muted-foreground">Connect to your MetaTrader 5 account for live trading</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Connection Form */}
        <Card className="bg-card border-border card-hover">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5 text-primary" />
              Server Configuration
            </CardTitle>
            <CardDescription>
              Enter your MT5 broker server details
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="server">Broker Server</Label>
              <div className="relative">
                <Server className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  id="server"
                  placeholder="e.g., broker-server.com:443"
                  value={credentials.server}
                  onChange={(e) => setCredentials(prev => ({ ...prev, server: e.target.value }))}
                  className="pl-10"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="login">Account Login</Label>
              <div className="relative">
                <Shield className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  id="login"
                  placeholder="Your MT5 account number"
                  value={credentials.login}
                  onChange={(e) => setCredentials(prev => ({ ...prev, login: e.target.value }))}
                  className="pl-10"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Key className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  id="password"
                  type="password"
                  placeholder="Your MT5 password"
                  value={credentials.password}
                  onChange={(e) => setCredentials(prev => ({ ...prev, password: e.target.value }))}
                  className="pl-10"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="apiKey">API Key (Optional)</Label>
              <div className="relative">
                <Key className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  id="apiKey"
                  type="password"
                  placeholder="Optional API key for extended features"
                  value={credentials.apiKey}
                  onChange={(e) => setCredentials(prev => ({ ...prev, apiKey: e.target.value }))}
                  className="pl-10"
                />
              </div>
            </div>

            <div className="flex gap-3 pt-4">
              <Button 
                onClick={handleConnect} 
                disabled={isLoading}
                className="flex-1"
              >
                <Link className="h-4 w-4 mr-2" />
                {isLoading ? 'Checking...' : 'Check Status'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Connection Status */}
        <div className="space-y-6">
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle>Connection Status</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                <span className="text-muted-foreground">Status</span>
                <StatusIndicator status={connectionStatus} />
              </div>
              
              {error && (
                <div className="flex items-start gap-3 p-4 rounded-lg bg-destructive/10 border border-destructive/20">
                  <AlertCircle className="h-5 w-5 text-destructive mt-0.5" />
                  <div>
                    <p className="font-medium text-destructive">Connection Error</p>
                    <p className="text-sm text-muted-foreground">{error}</p>
                  </div>
                </div>
              )}

              {mt5Status && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                    <span className="text-sm text-muted-foreground">Mode</span>
                    <Badge variant="outline">
                      {mt5Status.mode === 'real' ? 'Real MT5' : mt5Status.mode === 'mock' ? 'Mock' : 'Unavailable'}
                    </Badge>
                  </div>
                  
                  {mt5Status.connected && (
                    <div className="flex items-start gap-3 p-4 rounded-lg bg-success/10 border border-success/20">
                      <CheckCircle className="h-5 w-5 text-success mt-0.5" />
                      <div>
                        <p className="font-medium text-success">Connected Successfully</p>
                        <p className="text-sm text-muted-foreground">
                          {mt5Status.mode === 'real' 
                            ? 'Real MT5 connection active' 
                            : 'Mock MT5 client active (simulation mode)'}
                        </p>
                      </div>
                    </div>
                  )}

                  {!mt5Status.connected && mt5Status.mode !== 'mock' && (
                    <div className="flex items-start gap-3 p-4 rounded-lg bg-warning/10 border border-warning/20">
                      <AlertCircle className="h-5 w-5 text-warning mt-0.5" />
                      <div>
                        <p className="font-medium text-warning">Not Connected</p>
                        <p className="text-sm text-muted-foreground">
                          MT5 connection is not active. Check backend configuration.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Info Card */}
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle className="text-base">Integration Notes</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground">
              <p>
                <strong className="text-foreground">Backend Integration:</strong> MT5 connection is managed 
                server-side via environment variables (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER) or falls back to mock mode.
              </p>
              <p>
                <strong className="text-foreground">Status Check:</strong> Click "Check Status" to query 
                the backend for current MT5 connection status via the /status endpoint.
              </p>
              <p>
                <strong className="text-foreground">Configuration:</strong> Configure MT5 credentials in 
                backend .env file or environment variables. Never expose credentials in frontend code.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default MT5Connection;
