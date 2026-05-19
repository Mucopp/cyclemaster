importScripts('https://www.gstatic.com/firebasejs/10.12.0/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/10.12.0/firebase-messaging-compat.js');

firebase.initializeApp({
    apiKey: "AIzaSyCflGd5zuToK5QDkvONTelVwUqHvW0gKnw",
    authDomain: "cyclemaster-pro-cc443.firebaseapp.com",
    projectId: "cyclemaster-pro-cc443",
    storageBucket: "cyclemaster-pro-cc443.firebasestorage.app",
    messagingSenderId: "965620537440",
    appId: "1:965620537440:web:1d5c923b620afa7cb48b05"
});

const messaging = firebase.messaging();

// Arka planda gelen bildirimleri göster
messaging.onBackgroundMessage(payload => {
    console.log('Arka plan bildirimi:', payload);
    const { title, body } = payload.notification;
    // Bu zaten sadece arka planda çalışır, kontrol gerek yok
    self.registration.showNotification(title, {
        body: body,
        icon: './icon-192.png',
        badge: './icon-192.png',
        tag: 'cyclemaster-signal',
        renotify: true,
        vibrate: [200, 100, 200, 100, 200]
    });
});

// Bildirime tıklanınca uygulamayı aç
self.addEventListener('notificationclick', event => {
    event.notification.close();
    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true }).then(clientList => {
            for (const client of clientList) {
                if ('focus' in client) return client.focus();
            }
            if (clients.openWindow) return clients.openWindow('./');
        })
    );
});

// Statik cache
const CACHE_NAME = 'cyclemaster-v4';
const STATIC_ASSETS = ['./', './index.html', './manifest.json', './icon-192.png', './icon-512.png'];

self.addEventListener('install', event => {
    event.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(STATIC_ASSETS).catch(()=>{})));
    self.skipWaiting();
});

self.addEventListener('activate', event => {
    event.waitUntil(caches.keys().then(keys =>
        Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ));
    self.clients.claim();
});

self.addEventListener('fetch', event => {
    const url = event.request.url;
    if (url.includes('binance.com') || url.includes('googleapis') || url.includes('gstatic')) {
        event.respondWith(fetch(event.request).catch(() => new Response('{}')));
        return;
    }
    event.respondWith(
        caches.match(event.request).then(cached => {
            if (cached) return cached;
            return fetch(event.request).then(response => {
                if (response && response.status === 200) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then(c => c.put(event.request, clone));
                }
                return response;
            }).catch(() => caches.match('./index.html'));
        })
    );
});
