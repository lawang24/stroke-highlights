import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp, mkdir, writeFile, symlink, rm, realpath } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { devTestClips } from '../scripts/dev-test-clips.mjs';

async function fixture(t) {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'test-clip-picker-')));
  let middleware;
  await devTestClips().configureServer({ config: { root, logger: { error: message => assert.fail(message) } },
    middlewares: { use(handler) { middleware = handler; } } });
  const server = createServer((req, res) => middleware(req, res, () => { res.writeHead(404); res.end(); }));
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => {
    await new Promise(resolve => server.close(resolve));
    await rm(root, { recursive: true, force: true });
  });
  return { root, request: (path = '', options) => fetch(`http://127.0.0.1:${server.address().port}/__dev/test-clips${path}`, options) };
}

test('dev picker serves only enumerated clips, preserving their file bytes and metadata', async t => {
  const { root, request } = await fixture(t);
  await mkdir(join(root, 'outputs/regression-review-v1/clips'), { recursive: true });
  await writeFile(join(root, 'small-clip.mov'), 'original recording');
  await writeFile(join(root, 'outputs/regression-review-v1/clips/clip-001.mp4'), 'regression recording');
  await writeFile(join(root, 'outputs/private.mp4'), 'not a test input');
  await symlink(join(root, 'outputs/private.mp4'), join(root, 'outputs/regression-review-v1/clips/clip-002.mp4'));
  const response = await request();
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('cache-control'), 'no-store');
  const clips = await response.json();
  assert.deepEqual(clips.map(clip => clip.name), ['small-clip.mov', 'clip-001.mp4']);
  assert.deepEqual(clips.map(clip => clip.group), ['Recordings', 'Regression clips']);
  assert.ok(clips.every(clip => !('path' in clip) && clip.lastModified > 0));
  const media = await request(`/${clips[0].id}`);
  assert.equal(media.headers.get('content-type'), 'video/quicktime');
  assert.equal(Number(media.headers.get('content-length')), clips[0].size);
  assert.equal(await media.text(), 'original recording');
  const head = await request(`/${clips[1].id}`, { method: 'HEAD' });
  assert.equal(head.status, 200);
  assert.equal(await head.text(), '');
  const privateId = Buffer.from('outputs/private.mp4').toString('base64url');
  assert.equal((await request(`/${privateId}`)).status, 404);
  assert.equal((await request('/unknown')).status, 404);
  assert.equal((await request('', { method: 'POST' })).status, 405);
  assert.equal((await request('', { headers: { Origin: 'https://example.com' } })).status, 403);
  assert.equal((await request('', { headers: { 'Sec-Fetch-Site': 'cross-site' } })).status, 403);
});

test('a checkout without local test media returns an empty picker', async t => {
  const { request } = await fixture(t);
  assert.deepEqual(await (await request()).json(), []);
  assert.equal(devTestClips().apply, 'serve');
  assert.equal(devTestClips().configurePreviewServer, undefined);
});
