import { readdir, realpath, stat } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { resolve, basename, extname } from 'node:path';
import { pipeline } from 'node:stream/promises';

const endpoint = '/__dev/test-clips';
const sources = ['small-clip.mov', 'inputs/long-demo-last-5-minutes.mp4', 'outputs/adaspot-comparison-input.mp4'];
const reviewDirectory = 'outputs/regression-review-v1/clips';

async function listClips(root) {
  let names = [];
  try { names = await readdir(resolve(root, reviewDirectory)); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  const paths = [...sources, ...names.filter(name => /^clip-\d+\.mp4$/.test(name)).sort().map(name => `${reviewDirectory}/${name}`)];
  const clips = [];
  for (const relative of paths) {
    const path = resolve(root, relative);
    try {
      // Only serve these local recordings, never symlink targets or arbitrary paths.
      if (await realpath(path) !== path) continue;
      const info = await stat(path);
      if (!info.isFile() || !info.size || info.size > 500 * 1024 * 1024) continue;
      const id = Buffer.from(relative).toString('base64url');
      clips.push({ id, path, name: basename(path), group: relative.startsWith(reviewDirectory) ? 'Regression clips' : 'Recordings',
        size: info.size, lastModified: Math.trunc(info.mtimeMs), type: extname(path) === '.mov' ? 'video/quicktime' : 'video/mp4' });
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  return clips;
}

export function devTestClips() {
  return {
    name: 'local-test-clips',
    apply: 'serve',
    async configureServer(server) {
      const root = await realpath(server.config.root);
      server.middlewares.use(async (req, res, next) => {
        const pathname = new URL(req.url, 'http://localhost').pathname;
        if (pathname !== endpoint && !pathname.startsWith(`${endpoint}/`)) return next();
        res.setHeader('Cache-Control', 'no-store');
        res.setHeader('X-Content-Type-Options', 'nosniff');
        // Private fixtures are available only to this local dev page.
        const local = ['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(req.socket.remoteAddress);
        if (!local || req.headers['sec-fetch-site'] === 'cross-site' ||
            (req.headers.origin && req.headers.origin !== `http://${req.headers.host}`)) {
          res.writeHead(403); res.end('Local development only.'); return;
        }
        if (!['GET', 'HEAD'].includes(req.method)) {
          res.writeHead(405, { Allow: 'GET, HEAD' }); res.end(); return;
        }
        try {
          const clips = await listClips(root);
          if (pathname === endpoint) {
            res.setHeader('Content-Type', 'application/json');
            res.end(req.method === 'HEAD' ? undefined : JSON.stringify(clips.map(({ path, ...clip }) => clip)));
            return;
          }
          const clip = clips.find(item => pathname === `${endpoint}/${item.id}`);
          if (!clip) { res.writeHead(404); res.end('Test clip not found.'); return; }
          res.writeHead(200, { 'Content-Type': clip.type, 'Content-Length': clip.size });
          if (req.method === 'HEAD') { res.end(); return; }
          await pipeline(createReadStream(clip.path), res);
        } catch (error) {
          if (res.headersSent) res.destroy();
          else { res.writeHead(500); res.end('Could not read local test clips.'); }
          if (error.code !== 'ERR_STREAM_PREMATURE_CLOSE') server.config.logger.error(error.message);
        }
      });
    },
  };
}
