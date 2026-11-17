import { defineConfig } from 'vite';
import { devTestClips } from './scripts/dev-test-clips.mjs';

export default defineConfig({
  base: './',
  plugins: [devTestClips()],
  optimizeDeps: { exclude: ['@ffmpeg/ffmpeg'] },
  server: { fs: { deny: ['.env', '.env.*', '*.{crt,pem}', '**/.git/**', '**/inputs/**', '**/outputs/**', '**/*.{mov,MOV,mp4,mkv,webm}'] } },
});
