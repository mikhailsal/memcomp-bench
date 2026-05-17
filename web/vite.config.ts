import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { resolve } from 'path'
import { readFile } from 'fs/promises'
import type { Plugin } from 'vite'

function serveOutputPlugin(): Plugin {
  const outputDir = resolve(__dirname, '..', 'output');
  return {
    name: 'serve-local-output',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const prefix = '/memcomp-bench/output/';
        if (!req.url?.startsWith(prefix)) return next();

        const filePath = resolve(outputDir, req.url.slice(prefix.length));
        if (!filePath.startsWith(outputDir)) {
          res.statusCode = 403;
          res.end('Forbidden');
          return;
        }
        try {
          const content = await readFile(filePath, 'utf-8');
          const ext = filePath.split('.').pop();
          const mime = ext === 'jsonl' ? 'application/jsonlines' : 'application/json';
          res.setHeader('Content-Type', mime);
          res.end(content);
        } catch {
          res.statusCode = 404;
          res.end('Not found');
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), serveOutputPlugin()],
  base: '/memcomp-bench/',
})
