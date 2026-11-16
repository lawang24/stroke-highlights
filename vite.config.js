import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  optimizeDeps: { exclude: ['@ffmpeg/ffmpeg'] },
  server: { fs: { deny: ['.env', '.env.*', '*.{crt,pem}', '**/.git/**', '**/inputs/**', '**/outputs/**', '**/*.{mov,MOV,mp4,mkv,webm}'] } },
});
