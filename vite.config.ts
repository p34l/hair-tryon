import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// HTTPS-подобные требования (камера) работают на localhost без сертификата.
// host: true — чтобы можно было открыть с телефона по локальной сети.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
  },
  worker: {
    // Классический (iife) воркер для билда. Сам воркер не импортит модули
    // (MediaPipe грузится через importScripts), поэтому совместим с classic.
    format: 'iife',
  },
});
