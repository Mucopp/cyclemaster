const CACHE_NAME = 'cyclemaster-v2';
const STATIC_ASSETS = [
  './',
  './index.html',
  './manifest.json',
  './icon-192.png',
  './icon-512.png'
];

// ── INSTALL ──
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(STATIC_ASSETS).catch(() => {});
    })
  );
  self.skipWaiting();
});

// ── ACTIVATE ──
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// ── FETCH ──
self.addEventListener('fetch', event => {
  const url = event.request.url;

  // Binance API — her zaman canlı, cache'leme
  if (url.includes('binance.com') || url.includes('fonts.googleapis') || url.includes('fonts.gstatic')) {
    event.respondWith(
      fetch(event.request).catch(() => new Response('{}', {
        headers: { 'Content-Type': 'application/json' }
      }))
    );
    return;
  }

  // Statik dosyalar — cache first
  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(response => {
        if (response && response.status === 200) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        }
        return response;
      }).catch(() => caches.match('./index.html'));
    })
  );
});

// ── BİLDİRİM: Push (sunucudan gelen) ──
self.addEventListener('push', event => {
  if (!event.data) return;
  let data = { title: 'CycleMaster Pro', body: 'Yeni sinyal!' };
  try { data = event.data.json(); } catch(e) {}

  event.waitUntil(
    self.registration.showNotification(data.title, {
      body: data.body,
      icon: './icon-192.png',
      badge: './icon-192.png',
      tag: 'cyclemaster-signal',
      renotify: true,
      vibrate: [200, 100, 200, 100, 200],
      requireInteraction: false
    })
  );
});

// ── BİLDİRİM: Tıklanınca uygulamayı aç ──
self.addEventListener('notificationclick', event => {
  event.notification.close();
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then(clientList => {
      // Uygulama zaten açıksa öne getir
      for (const client of clientList) {
        if ('focus' in client) return client.focus();
      }
      // Değilse yeni sekme aç
      if (clients.openWindow) return clients.openWindow('./');
    })
  );
});
