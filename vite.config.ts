import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  publicDir: 'public',
  define: {
    __PWA_VERSION__: JSON.stringify(process.env.npm_package_version || '1.0.0'),
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    assetsDir: 'assets',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          icons: ['lucide-react'],
          pwa: ['./src/hooks/usePWA.ts', './src/components/PWAInstallButton.tsx']
        }
      }
    }
  },
  server: {
    host: true,
    port: 5173,
    headers: {
      'Service-Worker-Allowed': '/'
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        ws: true
      }
    }
  }
})