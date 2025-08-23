<script>
  import { onMount } from 'svelte';
  
  let zip = '';
  let size = '20ft';
  let result = null;
  let error = '';
  let loading = false;

  const API_BASE = 'http://localhost:8000';

  async function getQuote() {
    if (!/^\d{5}$/.test(zip)) {
      error = 'Please enter a valid 5-digit ZIP code';
      return;
    }
    
    error = '';
    loading = true;
    
    try {
      const response = await fetch(`${API_BASE}/quote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          origin_zip: '40507', // Lexington, KY
          dest_zip: zip,
          container_size: size
        })
      });
      
      if (!response.ok) throw new Error('Failed to get quote');
      result = await response.json();
    } catch (err) {
      error = err.message;
      result = null;
    } finally {
      loading = false;
    }
  }

  function handleZipInput(event) {
    zip = event.target.value.replace(/\D/g, '').slice(0, 5);
    error = '';
    result = null;
  }

  $: selectedPrice = result ? result.price : 0;
</script>

<svelte:head>
  <title>McDaniels — The Strongest Name in Shipping Containers</title>
  <meta name="description" content="AI-powered shipping container delivery pricing. Get instant quotes with accurate delivery estimates." />
</svelte:head>

<!-- Header -->
<header class="bg-navy text-white sticky top-0 z-50 shadow-lg">
  <div class="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
    <div class="text-xl font-black uppercase tracking-wide">McDaniels</div>
    <nav class="hidden md:flex space-x-6">
      <a href="#calculator" class="hover:text-teal-300 transition-colors">Calculator</a>
      <a href="#about" class="hover:text-teal-300 transition-colors">About</a>
    </nav>
    <a href="tel:+18594367304" class="bg-gold text-navy px-4 py-2 rounded-lg font-bold hover:bg-yellow-500 transition-all">
      (859) 436-7304
    </a>
  </div>
</header>

<!-- Hero with Calculator -->
<section class="relative min-h-screen flex items-center justify-center bg-gradient-to-br from-navy via-gray-800 to-navy">
  <div class="absolute inset-0 bg-black/40"></div>
  <div 
    class="absolute inset-0 bg-cover bg-center opacity-30"
    style="background-image: url('https://images.unsplash.com/photo-1579034963892-388c821d1d9f?q=80&w=1600&auto=format&fit=crop')"
  ></div>
  
  <div class="relative z-10 text-center text-white max-w-2xl mx-auto px-4">
    <h1 class="text-6xl md:text-8xl font-black mb-6 text-shadow-lg">
      McDaniels
    </h1>
    <p class="text-xl md:text-2xl font-semibold text-gold mb-12">
      The Strongest Name in Shipping Containers
    </p>
    
    <!-- Calculator -->
    <div class="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
      <!-- Container Size Selector -->
      <div class="flex gap-4 mb-6 justify-center">
        <button 
          class="px-6 py-3 rounded-xl font-bold transition-all {size === '20ft' ? 'bg-gold text-navy' : 'bg-white/20 text-white hover:bg-white/30'}"
          on:click={() => size = '20ft'}
        >
          20' Container
        </button>
        <button 
          class="px-6 py-3 rounded-xl font-bold transition-all {size === '40ft' ? 'bg-gold text-navy' : 'bg-white/20 text-white hover:bg-white/30'}"
          on:click={() => size = '40ft'}
        >
          40' Container
        </button>
      </div>
      
      <!-- ZIP Input -->
      <div class="flex gap-4 mb-4">
        <input
          type="text"
          placeholder="Enter ZIP Code"
          bind:value={zip}
          on:input={handleZipInput}
          on:keydown={(e) => e.key === 'Enter' && getQuote()}
          maxlength="5"
          class="flex-1 px-4 py-3 rounded-xl text-navy font-bold text-center text-lg border-2 border-transparent focus:border-gold focus:outline-none"
        />
        <button 
          on:click={getQuote}
          disabled={loading || zip.length !== 5}
          class="px-8 py-3 bg-gold text-navy rounded-xl font-bold disabled:bg-gray-400 hover:bg-yellow-500 transition-all"
        >
          {loading ? 'Getting Quote...' : 'Get Quote'}
        </button>
      </div>
      
      <!-- Error -->
      {#if error}
        <div class="bg-red-500/20 border border-red-500 text-red-200 px-4 py-3 rounded-xl mb-4">
          {error}
        </div>
      {/if}
      
      <!-- Result -->
      {#if result}
        <div class="bg-navy/80 backdrop-blur-lg rounded-xl p-6 border-2 border-gold">
          <div class="text-4xl font-black text-gold mb-2">
            ${result.price}
          </div>
          <div class="text-lg font-semibold mb-4">
            {size} to {result.dest_state} • {Math.round(result.distance_miles)} miles
          </div>
          <div class="text-sm text-gray-300 space-y-1">
            <div>Estimated delivery: {result.eta_window}</div>
            <div>Quote generated: {new Date(result.timestamp).toLocaleString()}</div>
            {#if result.model_used}
              <div class="text-teal-300">✨ AI-optimized pricing</div>
            {:else}
              <div>📐 Formula-based pricing</div>
            {/if}
          </div>
        </div>
      {/if}
    </div>
  </div>
</section>

<!-- About Section -->
<section id="about" class="py-20 bg-white">
  <div class="max-w-7xl mx-auto px-4 text-center">
    <h2 class="text-4xl font-black text-navy mb-8">Why Choose McDaniels?</h2>
    <div class="grid md:grid-cols-3 gap-8">
      <div class="p-6 rounded-xl border-2 border-gray-200 hover:border-gold transition-all hover:shadow-xl">
        <div class="text-4xl mb-4">🤖</div>
        <h3 class="text-xl font-bold text-navy mb-4">AI-Powered Pricing</h3>
        <p class="text-gray-600">Our machine learning model considers real-time factors for the most accurate quotes.</p>
      </div>
      <div class="p-6 rounded-xl border-2 border-gray-200 hover:border-gold transition-all hover:shadow-xl">
        <div class="text-4xl mb-4">🚛</div>
        <h3 class="text-xl font-bold text-navy mb-4">Fast Delivery</h3>
        <p class="text-gray-600">Reliable delivery times with real-time tracking and professional drivers.</p>
      </div>
      <div class="p-6 rounded-xl border-2 border-gray-200 hover:border-gold transition-all hover:shadow-xl">
        <div class="text-4xl mb-4">📱</div>
        <h3 class="text-xl font-bold text-navy mb-4">Instant Quotes</h3>
        <p class="text-gray-600">Get accurate pricing in seconds with our advanced quoting system.</p>
      </div>
    </div>
  </div>
</section>

<!-- Footer -->
<footer class="bg-navy text-white py-12">
  <div class="max-w-7xl mx-auto px-4 text-center">
    <div class="text-2xl font-black uppercase mb-4">McDaniels</div>
    <p class="text-gray-300 mb-8">Quality storage & cargo containers with AI-powered delivery.</p>
    <div class="border-t border-gray-700 pt-6 text-sm text-gray-400">
      © 2025 McDaniels. All rights reserved.
    </div>
  </div>
</footer>

<style>
  .text-shadow-lg {
    text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
  }
</style>
