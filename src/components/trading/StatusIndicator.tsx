import { cn } from '@/lib/utils';

interface StatusIndicatorProps {
  status: 'connected' | 'disconnected' | 'mock' | 'loading';
  label?: string;
  showLabel?: boolean;
}

const statusConfig = {
  connected: {
    color: 'bg-success',
    pulse: true,
    text: 'Connected'
  },
  disconnected: {
    color: 'bg-destructive',
    pulse: false,
    text: 'Disconnected'
  },
  mock: {
    color: 'bg-warning',
    pulse: true,
    text: 'Mock Mode'
  },
  loading: {
    color: 'bg-muted-foreground',
    pulse: true,
    text: 'Connecting...'
  }
};

export const StatusIndicator = ({ status, label, showLabel = true }: StatusIndicatorProps) => {
  const config = statusConfig[status];
  
  return (
    <div className="flex items-center gap-2">
      <div className="relative">
        <div className={cn(
          "w-3 h-3 rounded-full",
          config.color
        )} />
        {config.pulse && (
          <div className={cn(
            "absolute inset-0 w-3 h-3 rounded-full animate-ping opacity-75",
            config.color
          )} />
        )}
      </div>
      {showLabel && (
        <span className="text-sm text-muted-foreground">
          {label || config.text}
        </span>
      )}
    </div>
  );
};
