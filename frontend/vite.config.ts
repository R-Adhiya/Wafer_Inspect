import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

import fs from 'fs'

if (fs.existsSync(process.cwd())) {
  const realCwd = fs.realpathSync(process.cwd())
  if (process.cwd() !== realCwd) {
    process.chdir(realCwd)
  }
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
