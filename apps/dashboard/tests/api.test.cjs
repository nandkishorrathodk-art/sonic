const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const ts = require('typescript');

function loadClient(fetch, authOverrides = {}) {
  const exports = {};
  const source = fs.readFileSync(require.resolve('../lib/api.ts'), 'utf8');
  vm.runInNewContext(ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText, {
    exports, process: { env: {} }, fetch, AbortController, Headers, setTimeout, clearTimeout,
    require: (name) => {
      if (name === './auth') return {
        getAuthToken: () => 'old-token', clearAuth() {},
        ensureAuthToken: async () => 'new-token', ...authOverrides,
      };
      if (name === '../types/workstation') return {};
      throw new Error(`Unexpected import: ${name}`);
    },
  });
  return exports.apiClient;
}

function untilAborted(signal) {
  return new Promise((resolve, reject) => {
    const abort = () => reject(new DOMException('Aborted', 'AbortError'));
    if (signal?.aborted) abort();
    else signal?.addEventListener('abort', abort, { once: true });
  });
}

// A bounded watchdog also makes regressions fail instead of hanging the suite.
async function bounded(promise) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error('request never settled')), 500);
    })]);
  } finally { clearTimeout(timer); }
}

test('timeout covers the response body after headers arrive', async () => {
  const client = loadClient(async (_url, options) => ({
    ok: true, status: 200, text: () => untilAborted(options.signal),
  }));
  await assert.rejects(bounded(client('/slow', { timeout: 20 })), /Request timeout/);
});

test('timeout applies to the authenticated retry', async () => {
  let calls = 0;
  const client = loadClient(async (_url, options) => {
    if (++calls === 1) return { status: 401 };
    return untilAborted(options.signal);
  });
  await assert.rejects(bounded(client('/retry', { timeout: 20 })), /Request timeout/);
  assert.equal(calls, 2);
});

test('retry uses normal parsing for empty responses and real API errors', async () => {
  for (const response of [new Response(null, { status: 204 }), new Response('{"detail":"Permission denied"}', { status: 403 })]) {
    let calls = 0;
    const client = loadClient(async () => ++calls === 1 ? { status: 401 } : response);
    if (response.ok) assert.equal(JSON.stringify(await client('/retry')), '{}');
    else await assert.rejects(client('/retry'), /Permission denied/);
  }
});

test('caller cancellation is preserved and is not reported as a timeout', async () => {
  const controller = new AbortController();
  const client = loadClient(async (_url, options) => untilAborted(options.signal));
  const result = client('/cancel', { signal: controller.signal, timeout: 50 });
  controller.abort();
  await assert.rejects(bounded(result), error => error.name === 'AbortError');
});

test('Headers objects and authentication reach fetch', async () => {
  const client = loadClient(async (_url, options) => {
    const headers = new Headers(options.headers);
    assert.equal(headers.get('X-Test'), 'present');
    assert.equal(headers.get('Authorization'), 'Bearer old-token');
    return new Response('{"ok":true}');
  });
  assert.equal((await client('/headers', { headers: new Headers({ 'X-Test': 'present' }) })).ok, true);
});

test('deadline also covers token refresh', async () => {
  const client = loadClient(async () => ({ status: 401 }), {
    ensureAuthToken: async (_base, _force, signal) => untilAborted(signal),
  });
  await assert.rejects(bounded(client('/refresh', { timeout: 20 })), /Request timeout/);
});

test('an already cancelled request never reaches fetch', async () => {
  let calls = 0;
  const client = loadClient(async () => { calls++; return new Response('{}'); });
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(client('/cancelled', { signal: controller.signal }), error => error.name === 'AbortError');
  assert.equal(calls, 0);
});
