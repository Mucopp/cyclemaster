const CACHE_NAME = 'cyclemaster-v1';
const STATIC_ASSETS = [
  './',
  './index.html',
  './manifest.json',
  './icon-192.png',
  './icon-512.png',
  'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap'
];

// Install: cache static assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(STATIC_ASSETS).catch(() => {});
    })
  );
  self.skipWaiting();
});

// Activate: clean old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch strategy:
// - Binance API: network first, no cache (always live data)
// - Everything else: cache first, fallback to network
self.addEventListener('fetch', event => {
  const url = event.request.url;

  // Binance API — always network, never cache
  if (url.includes('binance.com')) {
    event.respondWith(fetch(event.request).catch(() => new Response('{}', { headers: { 'Content-Type': 'application/json' } })));
    return;
  }

  // Fonts & static — cache first
  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(response => {
        if (response && response.status === 200 && response.type !== 'opaque') {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        }
        return response;
      }).catch(() => caches.match('./index.html'));
    })
  );
});

// Background sync placeholder (for future alert system)
self.addEventListener('sync', event => {
  if (event.tag === 'check-signals') {
    // future: background signal check
  }
});

// Push notification support (for future use)
self.addEventListener('push', event => {
  if (!event.data) return;
  const data = event.data.json();
  self.registration.showNotification(data.title || 'CycleMaster Pro', {
    body: data.body || 'Yeni sinyal!',
    icon: '/icon-192.png',
    badge: '/icon-192.png',
    tag: 'signal',
    renotify: true,
    vibrate: [200, 100, 200]
  });
});
