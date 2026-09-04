// UniqToken playground frontend (no build step, no dependencies).
//
// Tokenizer pills, live metrics, and shareable URL hashes are all computed
// client-side through the `uniqtoken_core` WebAssembly module. Build it with:
//   wasm-pack build crates/uniqtoken_core --target web \
//     --out-dir ../../docs/playground/pkg --no-pack \
//     -- --no-default-features --features wasm
// then serve this directory over HTTP (ES modules require http(s), not file://).

const DEBOUNCE_MS = 50;
const MAX_HASH_CHARS = 4000;

// Bounds against decompression bombs (CWE-409): a crafted share fragment must
// not inflate into an unbounded in-memory buffer. Both caps comfortably exceed
// anything the link writer emits (encoded payloads are capped at
// MAX_HASH_CHARS), so legitimate links are unaffected. The writer enforces the
// same byte bound up front, so every link it emits is also readable.
const MAX_PAYLOAD_CHARS = MAX_HASH_CHARS;
const MAX_STREAM_BYTES = 262144; // 256 KiB of share-link input or streamed codec output

const inputEl = document.getElementById("input");
const chipsEl = document.getElementById("chips");
const errorEl = document.getElementById("error");
const shareNoteEl = document.getElementById("share-note");

const metricEls = {
  tokens: document.getElementById("m-tokens"),
  bpt: document.getElementById("m-bpt"),
  fallback: document.getElementById("m-fallback"),
  logprob: document.getElementById("m-logprob"),
};

let tokenizer = null;
let debounceTimer = 0;

function showError(message) {
  errorEl.textContent = message;
}

// wasm-bindgen errors surface as opaque JsValue objects, so coerce them into
// a readable message instead of "[object Object]".
function formatError(err) {
  if (typeof err === "string") {
    return err;
  }
  return err?.message ?? String(err);
}

// Versioned share payloads: "v1." marks deflate-compressed UTF-8 bytes,
// while a bare payload keeps the legacy uncompressed encoding so links shared
// by older versions keep working.
const HASH_VERSION = "v1.";

function base64UrlEncode(bytes) {
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64UrlDecode(payload) {
  const binary = atob(payload.replace(/-/g, "+").replace(/_/g, "/"));
  return Uint8Array.from(binary, (ch) => ch.charCodeAt(0));
}

async function readStream(stream, limit) {
  const chunks = [];
  const reader = stream.getReader();
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    total += value.length;
    if (total > limit) {
      reader.cancel().catch(() => {});
      throw new Error("stream exceeded size limit");
    }
    chunks.push(value);
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

async function deflateBytes(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("deflate"));
  return readStream(stream, MAX_STREAM_BYTES);
}

async function inflateBytes(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new TextDecoder().decode(await readStream(stream, MAX_STREAM_BYTES));
}

async function encodeHashPayload(text) {
  const bytes = new TextEncoder().encode(text);
  // Writer/reader consistency: the reader caps inflated output at
  // MAX_STREAM_BYTES, so refuse larger inputs here instead of emitting a link
  // that fails to open.
  if (bytes.length > MAX_STREAM_BYTES) {
    const err = new Error("input too long for a shareable link");
    err.code = "SHARE_LINK_TOO_LONG";
    throw err;
  }
  if (typeof CompressionStream !== "undefined") {
    try {
      return HASH_VERSION + base64UrlEncode(await deflateBytes(bytes));
    } catch {
      // Fall through to the legacy uncompressed encoding below.
    }
  }
  return base64UrlEncode(bytes);
}

async function decodeHashPayload(payload) {
  // The link writer caps encoded payloads at MAX_HASH_CHARS, so anything
  // longer is hand-crafted: reject it before allocating decode buffers.
  if (payload.length > MAX_PAYLOAD_CHARS) {
    throw new Error("share link is too long");
  }
  if (payload.startsWith(HASH_VERSION)) {
    if (typeof DecompressionStream === "undefined") {
      throw new Error("this share link needs DecompressionStream support");
    }
    return inflateBytes(base64UrlDecode(payload.slice(HASH_VERSION.length)));
  }
  return new TextDecoder().decode(base64UrlDecode(payload));
}

async function decodeHash() {
  const match = location.hash.match(/^#t=([A-Za-z0-9\-_.]+)$/);
  if (!match) {
    return "";
  }
  return decodeHashPayload(match[1]);
}

let hashSeq = 0;

async function updateHash(text) {
  const seq = ++hashSeq;
  let payload;
  try {
    payload = await encodeHashPayload(text);
  } catch (err) {
    // Keep the previous link, but say so: a silent failure would strand
    // anyone holding the stale URL without explanation.
    shareNoteEl.textContent =
      err?.code === "SHARE_LINK_TOO_LONG"
        ? "Input too long for a shareable link — the URL was left unchanged."
        : "Could not update the share link.";
    return;
  }
  if (seq !== hashSeq) {
    return; // a newer keystroke already superseded this render
  }
  if (payload.length > MAX_HASH_CHARS) {
    // Leave the URL untouched and say so: silently dropping share state
    // would strand anyone holding the stale link.
    shareNoteEl.textContent = "Input too long for a shareable link — the URL was left unchanged.";
    return;
  }
  shareNoteEl.textContent = "";
  history.replaceState(null, "", `#t=${payload}`);
}

function render() {
  if (!tokenizer) {
    return;
  }
  const text = inputEl.value;
  // One encode per render: the result carries tokens, ids, and every metric,
  // so a keystroke never re-runs the normalize-to-Viterbi pipeline per value.
  let encoded;
  try {
    encoded = tokenizer.encode(text);
  } catch (err) {
    showError(`Tokenization failed: ${formatError(err)}`);
    return;
  }
  const tokens = Array.from(encoded.tokens);
  const ids = Array.from(encoded.ids);

  chipsEl.replaceChildren();
  tokens.forEach((token, index) => {
    const pill = document.createElement("span");
    // textContent (never innerHTML): token bytes cannot inject markup or scripts.
    pill.textContent = token === "" ? "∅" : token;
    pill.className = `chip c${index % 8}`;
    pill.title = `id=${ids[index]}`;
    chipsEl.appendChild(pill);
  });

  metricEls.tokens.textContent = String(tokens.length);
  metricEls.bpt.textContent = tokens.length === 0 ? "0.00" : encoded.bytesPerToken.toFixed(2);
  metricEls.fallback.textContent = String(encoded.fallbackCount);
  metricEls.logprob.textContent = tokens.length === 0 ? "0.00" : encoded.avgLogprob.toFixed(3);

  updateHash(text);
}

function scheduleRender() {
  window.clearTimeout(debounceTimer);
  debounceTimer = window.setTimeout(render, DEBOUNCE_MS);
}

async function boot() {
  let module;
  try {
    module = await import("./pkg/uniqtoken_core.js");
  } catch {
    showError(
      "WebAssembly bundle not found at ./pkg/. Build it with " +
        "`wasm-pack build crates/uniqtoken_core --target web --out-dir ../../docs/playground/pkg " +
        "--no-pack -- --no-default-features --features wasm`, then serve this directory over HTTP."
    );
    return;
  }
  try {
    await module.default();
    tokenizer = new module.PlaygroundTokenizer();
  } catch (err) {
    showError(`Failed to start the tokenizer: ${err}`);
    return;
  }

  document.getElementById("vocab-line").textContent = `Demo vocabulary: ${tokenizer.vocab_size()} entries.`;

  const shared = await decodeHash().catch((err) => {
    showError(`Could not open the shared link: ${formatError(err)}`);
    return "";
  });
  inputEl.value = shared !== "" ? shared : "def calculate_fibonacci(n: int) -> int:\nprint('hello world 🌍')";
  inputEl.addEventListener("input", scheduleRender);
  render();
}

boot();
