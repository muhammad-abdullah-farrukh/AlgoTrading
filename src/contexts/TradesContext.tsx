import { createContext, useContext, useState, ReactNode } from 'react';
import { Trade, generateTrades } from '@/utils/dummyData';

interface TradesContextType {
  trades: Trade[];
  addTrade: (trade: Omit<Trade, 'id' | 'timestamp' | 'profitLoss' | 'status'>) => void;
  refreshTrades: () => void;
}

const TradesContext = createContext<TradesContextType | undefined>(undefined);

export const TradesProvider = ({ children }: { children: ReactNode }) => {
  const [trades, setTrades] = useState<Trade[]>(() => {
    const initialTrades = generateTrades(50);
    // Ensure trades are sorted by timestamp (newest first)
    return initialTrades.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  });

  const addTrade = (tradeData: Omit<Trade, 'id' | 'timestamp' | 'profitLoss' | 'status'>) => {
    const newTrade: Trade = {
      ...tradeData,
      id: `TRD-${String(Date.now()).slice(-6)}`,
      timestamp: new Date(),
      profitLoss: (Math.random() - 0.4) * 500, // Simulated P/L
      status: 'completed' as const,
    };
    
    // Add new trade at the beginning (newest first)
    setTrades(prev => [newTrade, ...prev]);
  };

  const refreshTrades = () => {
    const newTrades = generateTrades(50);
    // Ensure trades are sorted by timestamp (newest first)
    setTrades(newTrades.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()));
  };

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

