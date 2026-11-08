const V="mp-v1",SHELL=["./","index.html","manifest.webmanifest"];
self.addEventListener("install",e=>e.waitUntil(caches.open(V).then(c=>c.addAll(SHELL))));
self.addEventListener("fetch",e=>{
  const u=new URL(e.request.url);if(u.origin!==location.origin)return;
  const net=u.pathname.includes("/data/");
  e.respondWith(fetch(e.request).then(r=>{const k=r.clone();caches.open(V).then(c=>c.put(e.request,k));return r}).catch(()=>caches.match(e.request)));
});
