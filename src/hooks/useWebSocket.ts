import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Base URL for WebSocket connections.
 * Uses VITE_API_URL environment variable if set, otherwise defaults to ws://localhost:8000
 */
const getWebSocketURL = (path: string): string => {
  const apiURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  // Convert http:// to ws:// and https:// to wss://
  const wsURL = apiURL.replace(/^http/, 'ws');
  return `${wsURL}${path}`;
};

interface WebSocketConfig {
  url: string;
  onMessage?: (data: unknown) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
  reconnectAttempts?: number;
  reconnectInterval?: number;
  autoConnect?: boolean;
}

interface WebSocketState {
  isConnected: boolean;
  isConnecting: boolean;
  lastMessage: unknown | null;
  error: string | null;
}

/**
 * WebSocket hook for real-time data connection.
 * 
 * Creates real WebSocket connections to backend endpoints.
 * Handles reconnection logic, message parsing, and connection lifecycle.
 * 
 * @example
 * ```ts
 * const { isConnected, lastMessage, connect, disconnect } = useWebSocket({
 *   url: '/ws/ticks/EURUSD',
 *   onMessage: (data) => console.log('Tick:', data),
 *   onOpen: () => console.log('Connected'),
 *   autoConnect: true
 * });
 * ```
 */
export const useWebSocket = (config: WebSocketConfig) => {
  const {
    url,
    onMessage,
    onOpen,
    onClose,
    onError,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
    autoConnect = false,
  } = config;

  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isConnecting: false,
    lastMessage: null,
    error: null,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const shouldReconnectRef = useRef(true);
  const onMessageRef = useRef<WebSocketConfig['onMessage']>(onMessage);
  const onOpenRef = useRef<WebSocketConfig['onOpen']>(onOpen);
  const onCloseRef = useRef<WebSocketConfig['onClose']>(onClose);
  const onErrorRef = useRef<WebSocketConfig['onError']>(onError);
  const fullURL = getWebSocketURL(url);

  useEffect(() => {
    onMessageRef.current = onMessage;
    onOpenRef.current = onOpen;
    onCloseRef.current = onClose;
    onErrorRef.current = onError;
  }, [onMessage, onOpen, onClose, onError]);

  const connect = useCallback(() => {
    // Don't connect if already connected or connecting
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }
    if (wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    // Close existing connection if any
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setState((prev) => ({ ...prev, isConnecting: true, error: null }));
    shouldReconnectRef.current = true;

    try {
      const ws = new WebSocket(fullURL);

      ws.onopen = () => {
        console.log('[WebSocket] Connected to:', fullURL);
        reconnectCountRef.current = 0;
        setState((prev) => ({
          ...prev,
          isConnected: true,
          isConnecting: false,
          error: null,
        }));
        onOpenRef.current?.();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setState((prev) => ({ ...prev, lastMessage: data }));
          onMessageRef.current?.(data);
        } catch (error) {
          console.error('[WebSocket] Failed to parse message:', error);
          // Still call onMessage with raw data
          setState((prev) => ({ ...prev, lastMessage: event.data }));
          onMessageRef.current?.(event.data);
        }
      };

      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        const errorMessage = 'WebSocket connection error';
        setState((prev) => ({
          ...prev,
          error: errorMessage,
          isConnecting: false,
        }));
        onErrorRef.current?.(error);
      };

      ws.onclose = (event) => {
        console.log('[WebSocket] Disconnected:', event.code, event.reason);
        setState((prev) => ({
          ...prev,
          isConnected: false,
          isConnecting: false,
        }));

        // Attempt reconnection if should reconnect and haven't exceeded attempts
        if (
          shouldReconnectRef.current &&
          reconnectCountRef.current < reconnectAttempts
        ) {
          reconnectCountRef.current += 1;
          console.log(
            `[WebSocket] Reconnecting (${reconnectCountRef.current}/${reconnectAttempts})...`
          );

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        } else if (reconnectCountRef.current >= reconnectAttempts) {
          setState((prev) => ({
            ...prev,
            error: 'Failed to reconnect after maximum attempts',
          }));
        }

        onCloseRef.current?.();
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('[WebSocket] Failed to create connection:', error);
      setState((prev) => ({
        ...prev,
        isConnecting: false,
        error: 'Failed to create WebSocket connection',
      }));
      onErrorRef.current?.(error as Event);
    }
  }, [fullURL, reconnectAttempts, reconnectInterval]);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = undefined;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setState({
      isConnected: false,
      isConnecting: false,
      lastMessage: null,
      error: null,
    });

    onCloseRef.current?.();
    console.log('[WebSocket] Disconnected');
  }, []);

  const sendMessage = useCallback(
    (message: unknown) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        try {
          wsRef.current.send(JSON.stringify(message));
          return true;
        } catch (error) {
          console.error('[WebSocket] Failed to send message:', error);
          return false;
        }
      }
      console.warn('[WebSocket] Cannot send - not connected');
      return false;
    },
    []
  );

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    ...state,
    connect,
    disconnect,
    sendMessage,
  };
};
