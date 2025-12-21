import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import api from '@/utils/api';

export type Trade = {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  timestamp: Date;
  profitLoss: number;
  status: 'completed' | 'pending' | 'cancelled';
};

interface TradesContextType {
  trades: Trade[];
  addTrade: (trade: Omit<Trade, 'id' | 'timestamp' | 'profitLoss' | 'status'>) => void;
  refreshTrades: () => void;
}

const TradesContext = createContext<TradesContextType | undefined>(undefined);

export const TradesProvider = ({ children }: { children: ReactNode }) => {
  const [trades, setTrades] = useState<Trade[]>([]);

  const addTrade = (tradeData: Omit<Trade, 'id' | 'timestamp' | 'profitLoss' | 'status'>) => {
    // Trade will be fetched from backend via refreshTrades
    // This is just for immediate UI update
    refreshTrades();
  };

  const refreshTrades = async () => {
    try {
      // Fetch real trades from backend
      const response = await api.get<{
        trades: Array<{
          id: number;
          symbol: string;
          trade_type: string;
          quantity: number;
          price: number;
          timestamp: string;
          profit_loss: number | null;
          status: string;
        }>;
        count: number;
      }>('/api/trade/history', {
        params: { limit: 50 }
      });
      
      // Convert to Trade format
      const formattedTrades: Trade[] = response.trades.map(t => ({
        id: `TRD-${t.id}`,
        symbol: t.symbol,
        action: t.trade_type.toUpperCase() as 'BUY' | 'SELL',
        quantity: t.quantity,
        price: t.price,
        timestamp: new Date(t.timestamp),
        profitLoss: t.profit_loss || 0,
        status: (t.status === 'executed' ? 'completed' : t.status) as 'completed' | 'pending' | 'cancelled'
      }));
      
      setTrades(formattedTrades.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()));
    } catch (error) {
      console.error('[TradesContext] Failed to refresh trades:', error);
    }
  };
  
  // Load trades from backend on mount
  useEffect(() => {
    refreshTrades();
  }, []);

  return (
    <TradesContext.Provider value={{ trades, addTrade, refreshTrades }}>
      {children}
    </TradesContext.Provider>
  );
};

export const useTrades = () => {
  const context = useContext(TradesContext);
  if (context === undefined) {
    throw new Error('useTrades must be used within a TradesProvider');
  }
  return context;
};

