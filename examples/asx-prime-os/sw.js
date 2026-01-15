// sw.js - ASX Prime OS Service Worker
const CACHE_NAME = 'asx-prime-v4.2';
const RUNTIME = 'runtime';
const SERVER_PORT = 7777;
const WS_PORT = 7778;

// DNS resolution mapping
const DNS_MAP = new Map([
  ['xjson.local', '127.0.0.1'],
  ['rig.local', '127.0.0.1'],
  ['hive.local', '127.0.0.1'],
  ['prime.local', '127.0.0.1'],
  ['kuhul.local', '127.0.0.1'],
  ['scx.local', '127.0.0.1'],
  ['localhost.local', '127.0.0.1']
]);

// Tunneling system
class TunnelManager {
  constructor() {
    this.tunnels = new Map();
    this.tunnelCounter = 0;
  }
  
  createTunnel(config) {
    const tunnelId = `tunnel_${Date.now()}_${this.tunnelCounter++}`;
    const tunnel = {
      id: tunnelId,
      localPort: config.localPort || 3000,
      protocol: config.protocol || 'http',
      subdomain: config.subdomain || tunnelId.slice(0, 8),
      publicUrl: `https://${config.subdomain || tunnelId}.asxprime.app`,
      target: `http://localhost:${config.localPort}`,
      createdAt: Date.now(),
      active: true,
      stats: { requests: 0, bytes: 0 }
    };
    
    this.tunnels.set(tunnelId, tunnel);
    return tunnel;
  }
  
  async proxyRequest(tunnelId, request) {
    const tunnel = this.tunnels.get(tunnelId);
    if (!tunnel) return null;
    
    // Rewrite URL to target
    const targetUrl = new URL(request.url);
    targetUrl.hostname = 'localhost';
    targetUrl.port = tunnel.localPort;
    
    const proxyRequest = new Request(targetUrl, {
      method: request.method,
      headers: request.headers,
      body: request.body,
      redirect: 'manual'
    });
    
    try {
      const response = await fetch(proxyRequest);
      tunnel.stats.requests++;
      tunnel.stats.bytes += Number(response.headers.get('content-length')) || 0;
      return response;
    } catch (error) {
      console.error('Tunnel proxy error:', error);
      return new Response('Tunnel proxy error', { status: 502 });
    }
  }
}

// XJSON Server implementation
class XJSONServer {
  constructor() {
    this.port = SERVER_PORT;
    this.routes = new Map();
    this.setupRoutes();
    this.connections = new Set();
  }
  
  setupRoutes() {
    // Core endpoints
    this.routes.set('/xjson/ping', this.handlePing.bind(this));
    this.routes.set('/xjson/status', this.handleStatus.bind(this));
    this.routes.set('/xjson/execute', this.handleExecute.bind(this));
    this.routes.set('/xjson/compile', this.handleCompile.bind(this));
    this.routes.set('/dns/resolve', this.handleDNS.bind(this));
    this.routes.set('/tunnel/create', this.handleTunnelCreate.bind(this));
    this.routes.set('/tunnel/list', this.handleTunnelList.bind(this));
    this.routes.set('/tunnel/close', this.handleTunnelClose.bind(this));
    this.routes.set('/fs/list', this.handleFSList.bind(this));
    this.routes.set('/fs/read', this.handleFSRead.bind(this));
    this.routes.set('/fs/write', this.handleFSWrite.bind(this));
  }
  
  async handleRequest(request) {
    const url = new URL(request.url);
    const handler = this.routes.get(url.pathname);
    
    if (handler) {
      return handler(request);
    }
    
    // Dynamic tunnel routing
    if (url.hostname.includes('.asxprime.app')) {
      const tunnelId = url.hostname.split('.')[0];
      return tunnelManager.proxyRequest(tunnelId, request);
    }
    
    return new Response('Not Found', { status: 404 });
  }
  
  // Route handlers
  async handlePing() {
    return this.jsonResponse({
      status: 'ok',
      timestamp: Date.now(),
      runtime: 'asx-prime-v4.2'
    });
  }
  
  async handleStatus() {
    return this.jsonResponse({
      server: {
        status: 'running',
        uptime: Date.now() - serverStartTime,
        connections: this.connections.size
      },
      runtime: {
        kuhul: 'active',
        scx: 'ready',
        xjson: 'serving'
      },
      resources: {
        tunnels: tunnelManager.tunnels.size,
        cache: await caches.keys()
      }
    });
  }
  
  async handleExecute(request) {
    const { code, language = 'kuhul' } = await request.json();
    
    // Execute in appropriate runtime
    let result;
    switch (language) {
      case 'kuhul':
        result = await this.executeKuhul(code);
        break;
      case 'scx':
        result = await this.executeSCX(code);
        break;
      case 'xjson':
        result = await this.executeXJSON(code);
        break;
      default:
        result = { error: 'Unsupported language' };
    }
    
    return this.jsonResponse(result);
  }

  async handleCompile(request) {
    const { code, language = 'kuhul' } = await request.json();
    return this.jsonResponse({
      status: 'compiled',
      language,
      length: code?.length || 0
    });
  }
  
  async executeKuhul(code) {
    // K'uhul Virtual Machine
    const glyphs = code.match(/\[[^\]]+\]/g) || [];
    const stack = [];
    const memory = new Map();
    
    for (const glyph of glyphs) {
      const clean = glyph.slice(1, -1).trim();
      const [op, ...args] = clean.split(' ');
      
      switch (op.toLowerCase()) {
        case 'pop':
          stack.push(args.join(' '));
          break;
        case 'push':
          memory.set(args[0], stack.pop());
          break;
        case 'add':
          const b = parseInt(stack.pop());
          const a = parseInt(stack.pop());
          stack.push(a + b);
          break;
        case 'wo':
          // Output operation
          return { output: args.join(' '), stack: [...stack] };
        default:
          stack.push(op);
      }
    }
    
    return { result: stack.pop(), stack: [...stack], memory: Object.fromEntries(memory) };
  }
  
  async handleDNS(request) {
    const { hostname } = await request.json();
    const ip = DNS_MAP.get(hostname) || await this.resolveExternal(hostname);
    
    return this.jsonResponse({
      hostname,
      ip,
      internal: DNS_MAP.has(hostname),
      timestamp: Date.now()
    });
  }
  
  async resolveExternal(hostname) {
    // Use DNS-over-HTTPS
    try {
      const response = await fetch(`https://cloudflare-dns.com/dns-query?name=${hostname}`, {
        headers: { 'Accept': 'application/dns-json' }
      });
      const data = await response.json();
      return data.Answer?.[0]?.data || '8.8.8.8';
    } catch {
      return '8.8.8.8';
    }
  }
  
  async handleTunnelCreate(request) {
    const config = await request.json();
    const tunnel = tunnelManager.createTunnel(config);
    
    return this.jsonResponse({
      id: tunnel.id,
      publicUrl: tunnel.publicUrl,
      target: tunnel.target,
      status: 'active'
    });
  }

  async handleTunnelList() {
    const tunnels = Array.from(tunnelManager.tunnels.values());
    return this.jsonResponse({ tunnels });
  }

  async handleTunnelClose(request) {
    const { id } = await request.json();
    const tunnel = tunnelManager.tunnels.get(id);
    if (!tunnel) {
      return this.jsonResponse({ error: 'Tunnel not found' }, 404);
    }
    tunnel.active = false;
    tunnelManager.tunnels.delete(id);
    return this.jsonResponse({ status: 'closed', id });
  }
  
  async handleFSList(request) {
    // Virtual filesystem
    const { path = '/' } = await request.json();
    const files = [
      { name: 'system.kuhul', type: 'file', size: 1024 },
      { name: 'config.xjson', type: 'file', size: 512 },
      { name: 'logs', type: 'directory' },
      { name: 'cache', type: 'directory' }
    ];
    
    return this.jsonResponse({ path, files });
  }

  async handleFSRead(request) {
    const { path = '/' } = await request.json();
    return this.jsonResponse({ path, content: '// virtual file' });
  }

  async handleFSWrite(request) {
    const { path = '/', content = '' } = await request.json();
    return this.jsonResponse({ path, bytes: content.length, status: 'written' });
  }
  
  jsonResponse(data, status = 200) {
    return new Response(JSON.stringify(data, null, 2), {
      status,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'X-ASX-Runtime': 'v4.2'
      }
    });
  }
}

// Initialize systems
const tunnelManager = new TunnelManager();
const xjsonServer = new XJSONServer();
const serverStartTime = Date.now();

// Service Worker Events
self.addEventListener('install', (event) => {
  console.log('🔧 ASX Prime OS installing...');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll([
        './',
        './index.html',
        './manifest.json'
      ]))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  console.log('🚀 ASX Prime OS activated');
  event.waitUntil(
    Promise.all([
      caches.keys().then(keys => 
        Promise.all(keys.map(key => 
          key !== CACHE_NAME && key !== RUNTIME ? caches.delete(key) : null
        ))
      ),
      self.clients.claim()
    ])
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  
  // API requests
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(handleAPI(event.request));
    return;
  }
  
  // Tunnel requests
  if (url.hostname.includes('.asxprime.app')) {
    event.respondWith(handleTunnel(event.request));
    return;
  }
  
  // DNS resolution for .local
  if (url.hostname.endsWith('.local')) {
    event.respondWith(handleLocalDNS(event.request));
    return;
  }
  
  // Cache with network fallback
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request)
        .then(response => {
          // Cache successful GET requests
          if (response.ok && event.request.method === 'GET') {
            const clone = response.clone();
            caches.open(RUNTIME).then(cache => cache.put(event.request, clone));
          }
          return response;
        })
      )
  );
});

async function handleAPI(request) {
  const url = new URL(request.url);
  // Strip /api prefix for internal routing
  url.pathname = url.pathname.replace('/api', '');
  return xjsonServer.handleRequest(new Request(url, request));
}

async function handleTunnel(request) {
  const url = new URL(request.url);
  const tunnelId = url.hostname.split('.')[0];
  return tunnelManager.proxyRequest(tunnelId, request);
}

async function handleLocalDNS(request) {
  const url = new URL(request.url);
  const ip = DNS_MAP.get(url.hostname);
  
  if (ip) {
    // Rewrite to localhost
    const newUrl = new URL(request.url);
    newUrl.hostname = ip;
    return fetch(new Request(newUrl, request));
  }
  
  return new Response('DNS resolution failed', { status: 404 });
}

// Background sync for offline operations
self.addEventListener('sync', (event) => {
  if (event.tag === 'asx-sync') {
    event.waitUntil(syncOperations());
  }
});

async function syncOperations() {
  // Sync pending operations
  const cache = await caches.open('asx-sync');
  const keys = await cache.keys();
  
  for (const request of keys) {
    try {
      await fetch(request);
      await cache.delete(request);
    } catch (error) {
      console.log('Sync failed for:', request.url);
    }
  }
}

// Periodic cleanup
setInterval(() => {
  // Clean old tunnels (older than 24 hours)
  const now = Date.now();
  for (const [id, tunnel] of tunnelManager.tunnels) {
    if (now - tunnel.createdAt > 24 * 60 * 60 * 1000) {
      tunnelManager.tunnels.delete(id);
    }
  }
}, 60 * 60 * 1000);

console.log('🌟 ASX Prime OS Service Worker ready');
