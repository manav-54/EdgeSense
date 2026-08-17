import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API is served by the sink service. In dev it is proxied so the browser
// talks to one origin and no CORS preflight sits in front of the live socket.
const API = process.env.VITE_API_TARGET ?? 'http://localhost:8099'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': { target: API, changeOrigin: true, ws: true },
    },
  },
  build: { outDir: 'dist', sourcemap: true },
})
