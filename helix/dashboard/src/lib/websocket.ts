'use client';

/**
 * HELIX Dashboard WebSocket Client
 * Production-grade WebSocket connection management with automatic reconnection,
 * message queuing, channel subscriptions, and comprehensive error handling.
 */

// ============================================================================
// Types
// ============================================================================

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';

export interface WebSocketConfig {
    url: string;
    reconnectAttempts?: number;
    reconnectDelay?: number;
    reconnectBackoffMultiplier?: number;
    maxReconnectDelay?: number;
    heartbeatInterval?: number;
    heartbeatTimeout?: number;
    messageTimeout?: number;
    debug?: boolean;
}

export interface WebSocketMessage<T = unknown> {
    id?: string;
    type: string;
    channel?: string;
    timestamp: number;
    data: T;
}

export interface SubscriptionOptions {
    onMessage?: <T = unknown>(message: WebSocketMessage<T>) => void;
    onError?: (error: Error) => void;
    onSubscribed?: () => void;
    onUnsubscribed?: () => void;
}

export interface PendingRequest {
    resolve: (value: unknown) => void;
    reject: (reason: Error) => void;
    timeout: ReturnType<typeof setTimeout>;
}

type MessageHandler<T = unknown> = (message: WebSocketMessage<T>) => void;
type ConnectionHandler = (state: ConnectionState) => void;
type ErrorHandler = (error: Error) => void;

// ============================================================================
// WebSocket Client Implementation
// ============================================================================

export class HelixWebSocketClient {
    private ws: WebSocket | null = null;
    private config: Required<WebSocketConfig>;
    private connectionState: ConnectionState = 'disconnected';
    private reconnectAttempts = 0;
    private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
    private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
    private heartbeatTimeout: ReturnType<typeof setTimeout> | null = null;
    private lastHeartbeat: number = 0;

    // Message handling
    private messageHandlers: Map<string, Set<MessageHandler>> = new Map();
    private channelHandlers: Map<string, Set<MessageHandler>> = new Map();
    private connectionHandlers: Set<ConnectionHandler> = new Set();
    private errorHandlers: Set<ErrorHandler> = new Set();

    // Subscriptions and queuing
    private subscriptions: Map<string, SubscriptionOptions> = new Map();
    private messageQueue: WebSocketMessage[] = [];
    private pendingRequests: Map<string, PendingRequest> = new Map();
    private messageIdCounter = 0;

    // Connection management
    private isIntentionalClose = false;
    private connectPromise: Promise<void> | null = null;
    private connectResolve: (() => void) | null = null;
    private connectReject: ((err: Error) => void) | null = null;

    constructor(config: WebSocketConfig) {
        this.config = {
            url: config.url,
            reconnectAttempts: config.reconnectAttempts ?? 10,
            reconnectDelay: config.reconnectDelay ?? 1000,
            reconnectBackoffMultiplier: config.reconnectBackoffMultiplier ?? 1.5,
            maxReconnectDelay: config.maxReconnectDelay ?? 30000,
            heartbeatInterval: config.heartbeatInterval ?? 30000,
            heartbeatTimeout: config.heartbeatTimeout ?? 10000,
            messageTimeout: config.messageTimeout ?? 30000,
            debug: config.debug ?? false,
        };
    }

    // ========================================================================
    // Connection Management
    // ========================================================================

    async connect(): Promise<void> {
        // If already connected, return immediately
        if (this.ws?.readyState === WebSocket.OPEN) {
            return Promise.resolve();
        }

        // If already connecting, wait for the existing connection attempt
        if (this.connectPromise) {
            return this.connectPromise;
        }

        this.isIntentionalClose = false;
        this.setConnectionState('connecting');

        this.connectPromise = new Promise<void>((resolve, reject) => {
            this.connectResolve = resolve;
            this.connectReject = reject;

            try {
                this.ws = new WebSocket(this.config.url);

                this.ws.onopen = () => {
                    this.log('WebSocket connected');
                    this.reconnectAttempts = 0;
                    this.setConnectionState('connected');
                    this.startHeartbeat();
                    this.flushMessageQueue();
                    this.resubscribeAll();

                    this.connectResolve?.();
                    this.connectPromise = null;
                    this.connectResolve = null;
                    this.connectReject = null;
                };

                this.ws.onclose = (event) => {
                    this.log(`WebSocket closed: ${event.code} - ${event.reason}`);
                    this.stopHeartbeat();

                    if (!this.isIntentionalClose) {
                        this.setConnectionState('reconnecting');
                        this.scheduleReconnect();
                    } else {
                        this.setConnectionState('disconnected');
                    }
                };

                this.ws.onerror = (event) => {
                    const error = new Error('WebSocket error');
                    this.log('WebSocket error:', event);
                    this.notifyErrorHandlers(error);

                    if (this.connectionState === 'connecting') {
                        this.connectReject?.(error);
                        this.connectPromise = null;
                        this.connectResolve = null;
                        this.connectReject = null;
                    }
                };

                this.ws.onmessage = (event) => {
                    this.handleMessage(event.data);
                };
            } catch (err) {
                const error = err instanceof Error ? err : new Error(String(err));
                this.setConnectionState('error');
                this.notifyErrorHandlers(error);
                reject(error);
                this.connectPromise = null;
                this.connectResolve = null;
                this.connectReject = null;
            }
        });

        return this.connectPromise;
    }

    disconnect(): void {
        this.isIntentionalClose = true;
        this.stopHeartbeat();
        this.clearReconnectTimeout();

        if (this.ws) {
            this.ws.close(1000, 'Client disconnect');
            this.ws = null;
        }

        this.setConnectionState('disconnected');
        this.subscriptions.clear();
        this.messageQueue = [];
        this.pendingRequests.forEach((req) => {
            clearTimeout(req.timeout);
            req.reject(new Error('Connection closed'));
        });
        this.pendingRequests.clear();
    }

    get isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }

    get state(): ConnectionState {
        return this.connectionState;
    }

    // ========================================================================
    // Subscription Management
    // ========================================================================

    subscribe(channel: string, options: SubscriptionOptions = {}): () => void {
        this.subscriptions.set(channel, options);

        if (this.isConnected) {
            this.sendSubscription(channel);
        }

        return () => this.unsubscribe(channel);
    }

    unsubscribe(channel: string): void {
        const sub = this.subscriptions.get(channel);
        this.subscriptions.delete(channel);
        this.channelHandlers.delete(channel);

        if (this.isConnected) {
            this.send({
                type: 'unsubscribe',
                timestamp: Date.now(),
                data: { channel },
            });
        }

        sub?.onUnsubscribed?.();
    }

    private sendSubscription(channel: string): void {
        const sub = this.subscriptions.get(channel);
        this.send({
            type: 'subscribe',
            timestamp: Date.now(),
            data: { channel },
        });
        sub?.onSubscribed?.();
    }

    private resubscribeAll(): void {
        this.subscriptions.forEach((_, channel) => {
            this.sendSubscription(channel);
        });
    }

    // ========================================================================
    // Message Handling
    // ========================================================================

    on<T = unknown>(type: string, handler: MessageHandler<T>): () => void {
        if (!this.messageHandlers.has(type)) {
            this.messageHandlers.set(type, new Set());
        }
        this.messageHandlers.get(type)!.add(handler as MessageHandler);

        return () => {
            this.messageHandlers.get(type)?.delete(handler as MessageHandler);
        };
    }

    onChannel<T = unknown>(channel: string, handler: MessageHandler<T>): () => void {
        if (!this.channelHandlers.has(channel)) {
            this.channelHandlers.set(channel, new Set());
        }
        this.channelHandlers.get(channel)!.add(handler as MessageHandler);

        return () => {
            this.channelHandlers.get(channel)?.delete(handler as MessageHandler);
        };
    }

    onConnectionChange(handler: ConnectionHandler): () => void {
        this.connectionHandlers.add(handler);
        return () => {
            this.connectionHandlers.delete(handler);
        };
    }

    onError(handler: ErrorHandler): () => void {
        this.errorHandlers.add(handler);
        return () => {
            this.errorHandlers.delete(handler);
        };
    }

    private handleMessage(data: string): void {
        try {
            const message: WebSocketMessage = JSON.parse(data);

            // Handle heartbeat response
            if (message.type === 'pong' || message.type === 'heartbeat_ack') {
                this.handleHeartbeatResponse();
                return;
            }

            // Handle request responses
            if (message.id && this.pendingRequests.has(message.id)) {
                const pending = this.pendingRequests.get(message.id)!;
                clearTimeout(pending.timeout);
                this.pendingRequests.delete(message.id);
                pending.resolve(message.data);
                return;
            }

            // Notify type handlers
            const typeHandlers = this.messageHandlers.get(message.type);
            if (typeHandlers) {
                typeHandlers.forEach((handler) => {
                    try {
                        handler(message);
                    } catch (err) {
                        this.log('Handler error:', err);
                    }
                });
            }

            // Notify channel handlers
            if (message.channel) {
                const channelHandler = this.channelHandlers.get(message.channel);
                if (channelHandler) {
                    channelHandler.forEach((handler) => {
                        try {
                            handler(message);
                        } catch (err) {
                            this.log('Channel handler error:', err);
                        }
                    });
                }

                // Notify subscription-specific handler
                const sub = this.subscriptions.get(message.channel);
                if (sub?.onMessage) {
                    try {
                        sub.onMessage(message);
                    } catch (err) {
                        sub.onError?.(err instanceof Error ? err : new Error(String(err)));
                    }
                }
            }

            // Notify all handlers (wildcard)
            const allHandlers = this.messageHandlers.get('*');
            if (allHandlers) {
                allHandlers.forEach((handler) => {
                    try {
                        handler(message);
                    } catch (err) {
                        this.log('Wildcard handler error:', err);
                    }
                });
            }
        } catch (err) {
            this.log('Failed to parse message:', err);
        }
    }

    // ========================================================================
    // Sending Messages
    // ========================================================================

    send(message: WebSocketMessage): void {
        if (this.isConnected) {
            this.ws!.send(JSON.stringify(message));
        } else {
            this.messageQueue.push(message);
        }
    }

    async request<T = unknown>(type: string, data: unknown, timeout?: number): Promise<T> {
        const id = `req-${++this.messageIdCounter}-${Date.now()}`;

        return new Promise<T>((resolve, reject) => {
            const timeoutId = setTimeout(() => {
                this.pendingRequests.delete(id);
                reject(new Error(`Request timeout: ${type}`));
            }, timeout ?? this.config.messageTimeout);

            this.pendingRequests.set(id, {
                resolve: resolve as (value: unknown) => void,
                reject,
                timeout: timeoutId,
            });

            this.send({
                id,
                type,
                timestamp: Date.now(),
                data,
            });
        });
    }

    private flushMessageQueue(): void {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            if (message && this.isConnected) {
                this.ws!.send(JSON.stringify(message));
            }
        }
    }

    // ========================================================================
    // Heartbeat Management
    // ========================================================================

    private startHeartbeat(): void {
        this.stopHeartbeat();
        this.lastHeartbeat = Date.now();

        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected) {
                this.send({
                    type: 'ping',
                    timestamp: Date.now(),
                    data: {},
                });

                // Set timeout for heartbeat response
                this.heartbeatTimeout = setTimeout(() => {
                    this.log('Heartbeat timeout - reconnecting');
                    this.ws?.close(4000, 'Heartbeat timeout');
                }, this.config.heartbeatTimeout);
            }
        }, this.config.heartbeatInterval);
    }

    private stopHeartbeat(): void {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
        if (this.heartbeatTimeout) {
            clearTimeout(this.heartbeatTimeout);
            this.heartbeatTimeout = null;
        }
    }

    private handleHeartbeatResponse(): void {
        this.lastHeartbeat = Date.now();
        if (this.heartbeatTimeout) {
            clearTimeout(this.heartbeatTimeout);
            this.heartbeatTimeout = null;
        }
    }

    // ========================================================================
    // Reconnection Logic
    // ========================================================================

    private scheduleReconnect(): void {
        if (this.reconnectAttempts >= this.config.reconnectAttempts) {
            this.log('Max reconnect attempts reached');
            this.setConnectionState('error');
            this.notifyErrorHandlers(new Error('Max reconnect attempts reached'));
            return;
        }

        this.reconnectAttempts++;
        const delay = Math.min(
            this.config.reconnectDelay * Math.pow(this.config.reconnectBackoffMultiplier, this.reconnectAttempts - 1),
            this.config.maxReconnectDelay
        );

        this.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

        this.reconnectTimeout = setTimeout(() => {
            this.connect().catch((err) => {
                this.log('Reconnect failed:', err);
            });
        }, delay);
    }

    private clearReconnectTimeout(): void {
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }
    }

    // ========================================================================
    // State Management
    // ========================================================================

    private setConnectionState(state: ConnectionState): void {
        if (this.connectionState !== state) {
            this.connectionState = state;
            this.connectionHandlers.forEach((handler) => {
                try {
                    handler(state);
                } catch (err) {
                    this.log('Connection handler error:', err);
                }
            });
        }
    }

    private notifyErrorHandlers(error: Error): void {
        this.errorHandlers.forEach((handler) => {
            try {
                handler(error);
            } catch (err) {
                this.log('Error handler error:', err);
            }
        });
    }

    private log(...args: unknown[]): void {
        if (this.config.debug) {
            console.log('[HelixWS]', ...args);
        }
    }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let wsClientInstance: HelixWebSocketClient | null = null;

export function getWebSocketClient(config?: Partial<WebSocketConfig>): HelixWebSocketClient {
    if (!wsClientInstance) {
        const wsUrl = config?.url || process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001/ws';
        wsClientInstance = new HelixWebSocketClient({
            url: wsUrl,
            ...config,
        });
    }
    return wsClientInstance;
}

export function resetWebSocketClient(): void {
    if (wsClientInstance) {
        wsClientInstance.disconnect();
        wsClientInstance = null;
    }
}

// ============================================================================
// React Hook for WebSocket
// ============================================================================

import { useState, useEffect, useCallback, useRef } from 'react';

export interface UseWebSocketOptions {
    url?: string;
    autoConnect?: boolean;
    channels?: string[];
    onMessage?: <T = unknown>(message: WebSocketMessage<T>) => void;
    onConnectionChange?: (state: ConnectionState) => void;
    onError?: (error: Error) => void;
}

export interface UseWebSocketReturn {
    isConnected: boolean;
    connectionState: ConnectionState;
    connect: () => Promise<void>;
    disconnect: () => void;
    send: (message: WebSocketMessage) => void;
    subscribe: (channel: string, options?: SubscriptionOptions) => () => void;
    unsubscribe: (channel: string) => void;
    on: <T = unknown>(type: string, handler: MessageHandler<T>) => () => void;
    request: <T = unknown>(type: string, data: unknown, timeout?: number) => Promise<T>;
}

export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketReturn {
    const {
        url,
        autoConnect = true,
        channels = [],
        onMessage,
        onConnectionChange,
        onError,
    } = options;

    const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
    const clientRef = useRef<HelixWebSocketClient | null>(null);
    const cleanupRef = useRef<(() => void)[]>([]);

    // Initialize client
    useEffect(() => {
        clientRef.current = getWebSocketClient(url ? { url } : undefined);

        // Set up connection state handler
        const unsubConnection = clientRef.current.onConnectionChange((state) => {
            setConnectionState(state);
            onConnectionChange?.(state);
        });
        cleanupRef.current.push(unsubConnection);

        // Set up error handler
        if (onError) {
            const unsubError = clientRef.current.onError(onError);
            cleanupRef.current.push(unsubError);
        }

        // Set up message handler
        if (onMessage) {
            const unsubMessage = clientRef.current.on('*', onMessage);
            cleanupRef.current.push(unsubMessage);
        }

        // Auto-connect if enabled
        if (autoConnect) {
            clientRef.current.connect().catch(console.error);
        }

        return () => {
            cleanupRef.current.forEach((cleanup) => cleanup());
            cleanupRef.current = [];
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [url, autoConnect]);

    // Subscribe to channels
    const channelsKey = channels.join(',');
    useEffect(() => {
        if (!clientRef.current) return;

        const unsubscribes = channels.map((channel) =>
            clientRef.current!.subscribe(channel)
        );

        return () => {
            unsubscribes.forEach((unsub) => unsub());
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [channelsKey]);

    const connect = useCallback(async () => {
        await clientRef.current?.connect();
    }, []);

    const disconnect = useCallback(() => {
        clientRef.current?.disconnect();
    }, []);

    const send = useCallback((message: WebSocketMessage) => {
        clientRef.current?.send(message);
    }, []);

    const subscribe = useCallback((channel: string, opts?: SubscriptionOptions) => {
        return clientRef.current?.subscribe(channel, opts) ?? (() => {});
    }, []);

    const unsubscribe = useCallback((channel: string) => {
        clientRef.current?.unsubscribe(channel);
    }, []);

    const on = useCallback(<T = unknown>(type: string, handler: MessageHandler<T>) => {
        return clientRef.current?.on(type, handler) ?? (() => {});
    }, []);

    const request = useCallback(<T = unknown>(type: string, data: unknown, timeout?: number) => {
        return clientRef.current?.request<T>(type, data, timeout) ?? Promise.reject(new Error('Not connected'));
    }, []);

    return {
        isConnected: connectionState === 'connected',
        connectionState,
        connect,
        disconnect,
        send,
        subscribe,
        unsubscribe,
        on,
        request,
    };
}

export default HelixWebSocketClient;
