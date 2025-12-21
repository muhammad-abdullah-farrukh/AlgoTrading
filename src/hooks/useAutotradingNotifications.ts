import { useEffect } from 'react';
import { useWebSocket } from './useWebSocket';
import { toast } from 'sonner';
import { TrendingUp, TrendingDown } from 'lucide-react';

/**
 * Hook to listen for autotrading trade notifications and display toasts
 */
export const useAutotradingNotifications = () => {
  const { lastMessage, isConnected } = useWebSocket({
    url: '/ws/autotrading-notifications',
    onMessage: (data) => {
      // Handle autotrading trade notifications
      if (data && typeof data === 'object' && 'type' in data) {
        const msg = data as { type: string; trade?: any };
        
        if (msg.type === 'autotrading_trade' && msg.trade) {
          const { symbol, action, quantity, price, confidence } = msg.trade;
          
          // Show toast notification
          const isBuy = action === 'BUY';
          const icon = isBuy ? '📈' : '📉';
          
          toast.success(`${icon} Auto-Trade Executed`, {
            description: `${action} ${quantity} ${symbol} @ ${price?.toFixed(5) || 'market'} (${(confidence * 100).toFixed(0)}% confidence)`,
            duration: 5000,
            icon: isBuy ? TrendingUp : TrendingDown,
          });
          
          console.log('[AutotradingNotifications] Trade notification:', msg.trade);
        }
      }
    },
    onError: (error) => {
      console.error('[AutotradingNotifications] WebSocket error:', error);
    },
    autoConnect: true,
    reconnectAttempts: 5,
  });

  return {
    isConnected,
    lastMessage,
  };
};

