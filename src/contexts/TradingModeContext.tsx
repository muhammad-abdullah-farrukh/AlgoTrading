import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import api, { APIError } from '@/utils/api';
import { toast } from 'sonner';

interface TradingModeContextType {
  isLiveMode: boolean;
  setLiveMode: (enabled: boolean) => void;
}

const TradingModeContext = createContext<TradingModeContextType | undefined>(undefined);

const LIVE_MODE_KEY = 'trading_mode_live';

export const TradingModeProvider = ({ children }: { children: ReactNode }) => {
  const [isLiveMode, setIsLiveModeState] = useState<boolean>(() => {
    const stored = localStorage.getItem(LIVE_MODE_KEY);
    return stored === 'true';
  });

  const setLiveMode = (enabled: boolean) => {
    const prev = isLiveMode;
    setIsLiveModeState(enabled);
    localStorage.setItem(LIVE_MODE_KEY, String(enabled));

    (async () => {
      try {
        const res = await api.put<{ live_mode: boolean }>('/api/trade/mode', { live_mode: enabled });
        setIsLiveModeState(!!res.live_mode);
        localStorage.setItem(LIVE_MODE_KEY, String(!!res.live_mode));
      } catch (e) {
        const err = e as unknown;
        if (err instanceof APIError) {
          console.error('[TradingMode] Failed to persist mode to backend:', err.message);
          toast.error('Failed to update trading mode', {
            description: err.message,
          });
        } else {
          console.error('[TradingMode] Failed to persist mode to backend:', err);
          toast.error('Failed to update trading mode', {
            description: 'Unexpected error while contacting backend.',
          });
        }
        setIsLiveModeState(prev);
        localStorage.setItem(LIVE_MODE_KEY, String(prev));
      }
    })();
  };

  useEffect(() => {
    let isMounted = true;

    (async () => {
      try {
        const res = await api.get<{ live_mode: boolean }>('/api/trade/mode');
        if (!isMounted) return;
        setIsLiveModeState(!!res.live_mode);
        localStorage.setItem(LIVE_MODE_KEY, String(!!res.live_mode));
      } catch (e) {
        // If backend is unavailable, fall back to local cache.
        const err = e as unknown;
        if (err instanceof APIError) {
          console.error('[TradingMode] Failed to load mode from backend:', err.message);
        } else {
          console.error('[TradingMode] Failed to load mode from backend:', err);
        }
      }
    })();

    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === LIVE_MODE_KEY) {
        setIsLiveModeState(e.newValue === 'true');
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => {
      isMounted = false;
      window.removeEventListener('storage', handleStorageChange);
    };
  }, []);

  return (
    <TradingModeContext.Provider value={{ isLiveMode, setLiveMode }}>
      {children}
    </TradingModeContext.Provider>
  );
};

export const useTradingMode = () => {
  const context = useContext(TradingModeContext);
  if (context === undefined) {
    throw new Error('useTradingMode must be used within a TradingModeProvider');
  }
  return context;
};
