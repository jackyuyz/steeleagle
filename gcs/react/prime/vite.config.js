import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { env } from 'node:process'

const gcsBackend = env.STEELEAGLE_GCS_TARGET || 'http://127.0.0.1:8002'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': gcsBackend,
      '/ws': { target: gcsBackend, ws: true },
    },
  },
  test: {
    environment: 'node',
  },
})
