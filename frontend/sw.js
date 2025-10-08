// Service Worker for Tamil MovieRec PWA
const CACHE_NAME = 'tamil-movierec-v1';
const API_CACHE_NAME = 'tamil-movierec-api-v1';

// Assets to cache immediately
const STATIC_ASSETS = [
    '/',
    '/static/manifest.json',
    '/static/icons/icon-192.png',
    '/static/icons/icon-512.png'
];

// API endpoints to cache
const API_ENDPOINTS = [
    '/api/health',
    '/api/genres',
    '/api/moods',
    '/api/directors',
    '/api/actors'
];

// Install Event - Cache static assets
self.addEventListener('install', (event) => {
    console.log('🎬 Service Worker: Installing...');
    
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('📦 Service Worker: Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            })
            .then(() => {
                console.log('✅ Service Worker: Installation complete');
                return self.skipWaiting(); // Activate immediately
            })
            .catch((error) => {
                console.error('❌ Service Worker: Installation failed', error);
            })
    );
});

// Activate Event - Clean up old caches
self.addEventListener('activate', (event) => {
    console.log('🚀 Service Worker: Activating...');
    
    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames.map((cacheName) => {
                        if (cacheName !== CACHE_NAME && cacheName !== API_CACHE_NAME) {
                            console.log('🗑️ Service Worker: Deleting old cache:', cacheName);
                            return caches.delete(cacheName);
                        }
                    })
                );
            })
            .then(() => {
                console.log('✅ Service Worker: Activated');
                return self.clients.claim(); // Take control immediately
            })
    );
});

// Fetch Event - Network First, fallback to Cache
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);
    
    // Skip cross-origin requests
    if (url.origin !== location.origin) {
        return;
    }
    
    // API requests - Network First with Cache Fallback
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(
            networkFirstStrategy(request, API_CACHE_NAME)
        );
        return;
    }
    
    // Static assets - Cache First with Network Fallback
    event.respondWith(
        cacheFirstStrategy(request, CACHE_NAME)
    );
});

// Network First Strategy (for API calls)
async function networkFirstStrategy(request, cacheName) {
    try {
        // Try network first
        const networkResponse = await fetch(request);
        
        // If successful, cache the response
        if (networkResponse.ok) {
            const cache = await caches.open(cacheName);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        // Network failed, try cache
        console.log('📡 Network failed, trying cache for:', request.url);
        const cachedResponse = await caches.match(request);
        
        if (cachedResponse) {
            return cachedResponse;
        }
        
        // Return offline response
        return new Response(
            JSON.stringify({ 
                error: 'Offline', 
                message: 'You are currently offline. Please check your internet connection.' 
            }),
            { 
                status: 503,
                statusText: 'Service Unavailable',
                headers: { 'Content-Type': 'application/json' }
            }
        );
    }
}

// Cache First Strategy (for static assets)
async function cacheFirstStrategy(request, cacheName) {
    // Try cache first
    const cachedResponse = await caches.match(request);
    
    if (cachedResponse) {
        return cachedResponse;
    }
    
    // Cache miss, try network
    try {
        const networkResponse = await fetch(request);
        
        // Cache the response for future use
        if (networkResponse.ok) {
            const cache = await caches.open(cacheName);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        console.error('❌ Failed to fetch:', request.url, error);
        
        // Return a fallback response
        if (request.destination === 'image') {
            // Return placeholder image for failed image requests
            return new Response(
                '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200"><rect fill="#1e293b" width="200" height="200"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#64748b" font-size="16">Image Unavailable</text></svg>',
                { headers: { 'Content-Type': 'image/svg+xml' } }
            );
        }
        
        // For HTML pages, return offline page
        if (request.destination === 'document') {
            return caches.match('/') || new Response(
                '<!DOCTYPE html><html><head><title>Offline</title></head><body><h1>You are offline</h1><p>Please check your internet connection.</p></body></html>',
                { headers: { 'Content-Type': 'text/html' } }
            );
        }
        
        return new Response('Network error', { status: 408 });
    }
}

// Background Sync - for offline actions
self.addEventListener('sync', (event) => {
    console.log('🔄 Background Sync:', event.tag);
    
    if (event.tag === 'sync-recommendations') {
        event.waitUntil(syncRecommendations());
    }
});

async function syncRecommendations() {
    // Implement background sync for recommendations
    console.log('🎬 Syncing recommendations...');
    // Add your sync logic here
}

// Push Notifications (optional)
self.addEventListener('push', (event) => {
    console.log('🔔 Push notification received');
    
    const data = event.data ? event.data.json() : {};
    const title = data.title || 'Tamil MovieRec';
    const options = {
        body: data.body || 'New Tamil movies added!',
        icon: '/static/icons/icon-192.png',
        badge: '/static/icons/badge-72.png',
        vibrate: [200, 100, 200],
        data: data.url || '/',
        actions: [
            {
                action: 'explore',
                title: 'Explore Movies',
                icon: '/static/icons/explore-icon.png'
            },
            {
                action: 'close',
                title: 'Close',
                icon: '/static/icons/close-icon.png'
            }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification(title, options)
    );
});

// Notification Click Handler
self.addEventListener('notificationclick', (event) => {
    console.log('🔔 Notification clicked:', event.action);
    
    event.notification.close();
    
    if (event.action === 'explore') {
        event.waitUntil(
            clients.openWindow(event.notification.data || '/')
        );
    }
});

// Message Handler - for communication with main app
self.addEventListener('message', (event) => {
    console.log('💬 Message received:', event.data);
    
    if (event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
    
    if (event.data.type === 'CLEAR_CACHE') {
        event.waitUntil(
            caches.keys().then((cacheNames) => {
                return Promise.all(
                    cacheNames.map((cacheName) => caches.delete(cacheName))
                );
            }).then(() => {
                console.log('✅ All caches cleared');
            })
        );
    }
});

// Periodic Background Sync (Chrome 80+)
self.addEventListener('periodicsync', (event) => {
    console.log('⏰ Periodic sync:', event.tag);
    
    if (event.tag === 'update-movies') {
        event.waitUntil(updateMovieCache());
    }
});

async function updateMovieCache() {
    console.log('🔄 Updating movie cache...');
    // Refresh API cache in background
    const cache = await caches.open(API_CACHE_NAME);
    
    for (const endpoint of API_ENDPOINTS) {
        try {
            const response = await fetch(endpoint);
            if (response.ok) {
                await cache.put(endpoint, response);
                console.log('✅ Cached:', endpoint);
            }
        } catch (error) {
            console.error('❌ Failed to cache:', endpoint, error);
        }
    }
}

console.log('🎬 Tamil MovieRec Service Worker Loaded');
