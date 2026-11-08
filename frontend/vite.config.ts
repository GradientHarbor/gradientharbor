import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: proxy the API to the collector. Prod: the collector serves dist/ itself (one URL).
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/v1': process.env.GH_COLLECTOR_URL ?? 'http://localhost:8000' } },
})
