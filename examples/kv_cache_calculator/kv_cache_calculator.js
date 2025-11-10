// kv_cache_calculator.js — verbose debug version with browser tokenizer attempt.
// - No HTML changes needed.
// - Logs network fetches, tokenizer load progress, and errors.
// - Uses Transformers.js to load tokenizer; falls back to 10 tokens if it fails.

(() => {
  const $ = (id) => document.getElementById(id);

  const els = {
    model: $("model"),
    dtype: $("dtype"),
    tokens: $("tokens"),
    textInput: $("textInput"),
    tokensMode: $("tokensMode"),
    textMode: $("textMode"),
    toggleBtn: $("toggleInputMode"),
    calcBtn: $("calculateButton"),
    status: $("tokenizeStatus"),
    result: $("result"),
  };

  // ------------------------ Debug helpers ------------------------
  function uiStatus(msg) {
    console.log("[KVDBG]", msg);
    if (els.status) els.status.textContent = String(msg);
  }
  function uiAppend(msg) {
    console.log("[KVDBG]", msg);
    if (!els.status) return;
    const prev = els.status.textContent || "";
    els.status.textContent = prev ? prev + " | " + msg : msg;
  }

  // Monkey-patch fetch once to see network calls
  if (!window.__kvdbg_fetch_patched__) {
    const origFetch = window.fetch;
    window.fetch = async function (...args) {
      const url = args[0];
      console.log("[KVDBG][FETCH →]", url, args[1] || {});
      try {
        const res = await origFetch.apply(this, args);
        console.log("[KVDBG][FETCH ←]", res.status, res.url);
        return res;
      } catch (e) {
        console.error("[KVDBG][FETCH ✖]", url, e?.name, e?.message);
        throw e;
      }
    };
    window.__kvdbg_fetch_patched__ = true;
  }

  // ------------------------ Model family & templates ------------------------
  function getFamily(modelId) {
    const m = (modelId || "").toLowerCase();
    if (m.startsWith("mistralai/")) return "mistral";
    if (m.startsWith("qwen/")) return "qwen";
    if (m.startsWith("deepseek-ai/")) return "deepseek";
    if (m.startsWith("lmsys/longchat")) return "llama";
    if (m.startsWith("meta-llama/") || m.includes("llama-3") || m.includes("llama3")) return "llama";
    if (m.startsWith("sao10k/")) return "llama";
    return "llama";
  }

  function llamaTemplate(messages) {
    let out = "";
    const sys = messages.find((m) => m.role === "system");
    if (sys) {
      out += "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" +
             sys.content + "<|eot_id|>";
    }
    for (const m of messages) {
      if (m.role === "user") {
        out += "<|start_header_id|>user<|end_header_id|>\n" + m.content + "<|eot_id|>";
      } else if (m.role === "assistant") {
        out += "<|start_header_id|>assistant<|end_header_id|>\n" + m.content + "<|eot_id|>";
      }
    }
    out += "<|start_header_id|>assistant<|end_header_id|>\n";
    return out;
  }

  function mistralTemplate(messages) {
    let out = "";
    const sys = messages.find((m) => m.role === "system");
    if (sys) out += `<s>[SYSTEM] ${sys.content}[/SYSTEM]\n`;
    let open = false;
    for (const m of messages) {
      if (m.role === "user") {
        out += (open ? "" : "<s>") + `[INST] ${m.content} [/INST]`;
        open = true;
      } else if (m.role === "assistant") {
        out += ` ${m.content}</s>`;
        open = false;
      }
    }
    if (open) out += " ";
    return out;
  }

  function qwenTemplate(messages) {
    let s = "";
    for (const m of messages) {
      s += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
    }
    s += `<|im_start|>assistant\n`;
    return s;
  }

  function deepseekTemplate(messages) {
    let s = "";
    for (const m of messages) {
      s += `<<<${m.role}>>>\n${m.content}\n<<<end>>>\n`;
    }
    s += `<<<assistant>>>\n`;
    return s;
  }

  function applyTemplate(family, msgs) {
    switch (family) {
      case "mistral": return mistralTemplate(msgs);
      case "qwen": return qwenTemplate(msgs);
      case "deepseek": return deepseekTemplate(msgs);
      case "llama":
      default: return llamaTemplate(msgs);
    }
  }

  // ------------------------ Transformers.js setup ------------------------
  const tokenizerCache = new Map();

  // Allow setting HF token from console:
  //   setHfToken("hf_XXXXXXXX")
  window.setHfToken = function setHfToken(token) {
    try {
      localStorage.setItem("HF_TOKEN", token || "");
      if (window.transformers?.env) {
        window.transformers.env.HF_TOKEN = token || "";
      }
      console.log("[KVDBG] HF token set.");
    } catch (e) {
      console.warn("[KVDBG] Failed to persist HF token:", e?.message || e);
    }
  };

  function initTransformersEnv() {
    if (!window.transformers) {
      console.warn("[KVDBG] transformers.js not found. Did the script load?");
      return;
    }
    const env = window.transformers.env;
    env.useBrowserCache = true;           // Cache files in IndexedDB
    env.allowLocalModels = false;         // Don’t look for local FS
    env.backends.onnx.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/"; // not strictly needed for tokenizer
    // Pick up HF token if provided
    try {
      const saved = localStorage.getItem("HF_TOKEN");
      if (saved) {
        env.HF_TOKEN = saved;
        console.log("[KVDBG] Loaded HF token from localStorage");
      }
    } catch {}
  }

  async function getTokenizer(modelId) {
    if (!window.transformers) throw new Error("Transformers.js is not loaded");
    initTransformersEnv();

    if (tokenizerCache.has(modelId)) {
      uiAppend(`cached:${modelId}`);
      return tokenizerCache.get(modelId);
    }

    uiStatus(`Loading tokenizer for ${modelId}…`);
    console.time(`[KVDBG] load tokenizer: ${modelId}`);

    try {
      const tok = await window.transformers.AutoTokenizer.from_pretrained(
        modelId,
        // progress callback
        {
          progress_callback: (p) => {
            // { status, file, loaded, total }
            const msg = `${p.status || "downloading"} ${p.file || ""} ` +
                        `${p.loaded || 0}/${p.total || "?"}`;
            console.log("[KVDBG][HF]", msg);
            uiStatus(`Tokenizer: ${msg}`);
          },
        }
      );
      console.timeEnd(`[KVDBG] load tokenizer: ${modelId}`);
      console.log("[KVDBG] Tokenizer loaded:", modelId, tok?._processor_config || {});
      tokenizerCache.set(modelId, tok);
      uiStatus(`Tokenizer ready: ${modelId}`);
      return tok;
    } catch (err) {
      console.timeEnd(`[KVDBG] load tokenizer: ${modelId}`);
      console.error("[KVDBG] Tokenizer load failed:", {
        modelId,
        name: err?.name,
        message: err?.message,
        cause: err?.cause,
      });
      uiStatus(`Tokenizer failed for ${modelId} (${err?.message || "unknown"})`);

      // Heuristic fallback (public model per family)
      const family = getFamily(modelId);
      const fallbackRepo = ({
        mistral: "mistralai/Mistral-7B-Instruct-v0.2",
        qwen: "Qwen/Qwen2.5-7B-Instruct",
        deepseek: "deepseek-ai/DeepSeek-V3",
        llama: "lmsys/longchat-7b-16k",
      })[family] || "lmsys/longchat-7b-16k";

      uiAppend(`fallback:${fallbackRepo}`);
      try {
        const tok = await window.transformers.AutoTokenizer.from_pretrained(
          fallbackRepo,
          {
            progress_callback: (p) => {
              const msg = `FB ${p.status || "downloading"} ${p.file || ""} ${p.loaded || 0}/${p.total || "?"}`;
              console.log("[KVDBG][HF][FB]", msg);
              uiStatus(`Tokenizer (fallback): ${msg}`);
            },
          }
        );
        tok.__approximate__ = true;
        tok.__fallback_repo__ = fallbackRepo;
        tokenizerCache.set(modelId, tok); // cache under requested id
        uiStatus(`Tokenizer fallback ready: ${fallbackRepo}`);
        return tok;
      } catch (fbErr) {
        console.error("[KVDBG] Fallback tokenizer failed:", {
          fallbackRepo,
          name: fbErr?.name,
          message: fbErr?.message,
          cause: fbErr?.cause,
        });
        uiStatus(`Tokenizer fallback failed (${fallbackRepo})`);
        throw err; // rethrow to trigger debug path
      }
    }
  }

  // ------------------------ Token counting ------------------------
    async function textToTokenCount(text, modelId) {
        try {
            const tok = await getTokenizer(modelId);

            // Prefer tokenize() — it returns plain tokens without specials.
            if (typeof tok.tokenize === "function") {
                const toks = tok.tokenize(text || "");
                const n = Array.isArray(toks) ? toks.length : 0;
                uiStatus(`Tokens: ${n}`);
                els.status.style.color = "#555"; // reset to default
                return n;
            }

            // Fallback: encode(). Some builds don't accept options.
            const enc = tok.encode(text || "");
            // enc may be an array-like or an object — try common shapes.
            const n =
            (typeof enc.length === "number" && enc.length) ||
            (Array.isArray(enc) && enc.length) ||
            (enc && typeof enc.getIds === "function" && enc.getIds().length) ||
            (enc && Array.isArray(enc.input_ids) && enc.input_ids.length) ||
            0;

            uiStatus(`Tokens: ${n}`);
            return n;
        } catch (e) {
            if (els.status) {
                els.status.textContent = "Sorry, this model is not supported yet.";
                els.status.style.color = "red";
                els.status.style.fontSize = "12px"; // same as before
            }
            return 0;
        }
    }

  // ------------------------ UI flow ------------------------
  function inTextMode() {
    return els.textMode.style.display !== "none";
  }
  function showTokensMode() {
    els.tokensMode.style.display = "";
    els.textMode.style.display = "none";
    els.toggleBtn.textContent = "Enter text instead";
    uiStatus("Tokens mode");
  }
  function showTextMode() {
    els.tokensMode.style.display = "none";
    els.textMode.style.display = "";
    els.toggleBtn.textContent = "Enter token count instead";
    uiStatus("Text mode");
    void updateTextTokenCount();
  }
  function toggleMode() {
    if (inTextMode()) showTokensMode(); else showTextMode();
  }

  async function updateTextTokenCount() {
    const txt = els.textInput.value || "";
    const modelId = els.model.value;
    uiStatus("Counting tokens…");
    const n = await textToTokenCount(txt, modelId);   // RAW (no template)
    els.tokens.value = String(n);                     // sync numeric field for calculateKVCache()
    return n;
  }

  async function handleCalculate() {
    let tokenCount;
    if (inTextMode()) {
      tokenCount = await updateTextTokenCount();
      els.result.textContent = `DEBUG: Using text→tokens count = ${tokenCount}`;
    } else {
      const parsed = parseInt(els.tokens.value, 10);
      tokenCount = Number.isFinite(parsed) ? parsed : 0;
      uiStatus(`Using tokens field = ${tokenCount}`);
    }

    if (typeof window.calculateKVCache === "function") {
      console.log("[KVDBG] Calling calculateKVCache()");
      window.calculateKVCache();
    } else {
      console.warn("[KVDBG] calculateKVCache() not found on window.");
    }
  }

  // ------------------------ Public entry point ------------------------
  window.setupTokenizer = function setupTokenizer() {
    console.log("[KVDBG] setupTokenizer() init");
    // Wire
    els.toggleBtn.addEventListener("click", toggleMode);
    els.calcBtn.addEventListener("click", (e) => { e.preventDefault(); void handleCalculate(); });
    els.textInput.addEventListener("input", () => { if (inTextMode()) void updateTextTokenCount(); });
    els.model.addEventListener("change", () => { if (inTextMode()) void updateTextTokenCount(); });

    // Default to Tokens mode (your HTML default)
    showTokensMode();

    // Print environment info
    if (window.transformers) {
      console.log("[KVDBG] transformers.js detected. env:", window.transformers.env);
    } else {
      console.warn("[KVDBG] transformers.js NOT detected. Check the CDN script tag.");
      uiStatus("transformers.js missing");
    }
  };
})();
