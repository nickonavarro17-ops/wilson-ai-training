// Wilson AI Training — proxy seguro hacia la API de Anthropic.
// Los prompts, el manual y los perfiles viven acá (servidor), así la API key
// no se puede usar para otra cosa que no sea esta app.

import { KB, PERFILES, QUIZ_CATS } from './_kb.js';

const MODEL_MAIN = 'claude-sonnet-5-5';
const MODEL_FAST = 'claude-haiku-5-5';
const MODEL_FALLBACK = 'claude-sonnet-4-6';

const MAX_TURNS = 30;          // mensajes por conversación
const MAX_MSG_CHARS = 800;     // largo máximo de cada mensaje del vendedor
const RATE_LIMIT = 40;         // llamadas por IP por ventana
const RATE_WINDOW_MS = 10 * 60 * 1000;

const hits = new Map();
function rateLimited(ip) {
  const now = Date.now();
  const arr = (hits.get(ip) || []).filter(t => now - t < RATE_WINDOW_MS);
  arr.push(now);
  hits.set(ip, arr);
  if (hits.size > 5000) hits.clear();
  return arr.length > RATE_LIMIT;
}

const ESTILO = `Hablas español de Chile, natural y cercano, tuteando ("tú", "tienes", "prefieres"). Nunca uses voseo ni modismos argentinos ("vos", "tenés", "querés", "che", "boludo").`;

function quizPrompt(cat, avoid) {
  const avoidTxt = avoid.length ? `\nNO repitas estos temas/modelos recientes: ${avoid.join(', ')}. Elige otro producto o concepto del manual.` : '';
  return `Eres un experto en productos Wilson Chile. Con base SOLO en este manual, genera UNA pregunta de opción múltiple sobre "${cat}" para capacitar vendedores de tienda.

MANUAL:
${KB}

REGLAS:
- Nivel: vendedor que atiende a jugadores amateur de club. Prioriza preguntas de situación ("Un cliente te dice X, ¿qué le recomiendas?") por sobre memorizar números.
- Usa SOLO datos que estén explícitos en el manual. No inventes specs, precios ni tecnologías.
- No preguntes diferencias de pocos gramos ni datos de precio.
- 4 opciones, una sola correcta; las incorrectas deben ser plausibles.
- Explicación corta (máx 2 líneas) que enseñe el porqué.
- ${ESTILO}${avoidTxt}

Responde SOLO con JSON válido, sin markdown:
{"cat":"${cat}","modelo":"producto o concepto principal","q":"pregunta","opts":["A","B","C","D"],"correct":0,"tip":"explicación"}`;
}

function clientePrompt(p) {
  return `Eres ${p.name}, un cliente real en una tienda Wilson Chile. Perfil: ${p.desc}
${ESTILO}
- Mantén el personaje siempre. Responde en 1 a 3 frases cortas, como hablaría un cliente de verdad (no como experto).
- No conoces nombres de tecnologías internas de Wilson. Sí puedes entender specs básicas si te las explican simple.
- Si el vendedor te hace buenas preguntas de diagnóstico, entrégale información útil de tu juego y tus necesidades.
- Si te habla muy técnico o intenta vender sin preguntarte nada, muéstrate confundido, escéptico o con prisa.
- Si te convence con argumentos que resuelven TU problema, puedes decidir comprar o pedir probarla.
- Nunca digas que eres una IA. No escribas acotaciones ni narración entre paréntesis.`;
}

function coachPrompt(p) {
  return `Eres un coach de ventas de Wilson Chile: un mentor cercano y práctico, no un inspector del manual.
${ESTILO}

MANUAL DE REFERENCIA:
${KB}

PERFIL DEL CLIENTE (el vendedor NO lo conoce, debe descubrirlo preguntando): ${p.desc}

Vas a ver la conversación completa. Evalúa SOLO el último mensaje del VENDEDOR considerando el momento de la conversación:
- Al inicio, lo importante es saludar y diagnosticar (preguntar antes de recomendar).
- A mitad, recomendar algo coherente con lo que el cliente contó y explicarlo como beneficio, no como ficha técnica.
- Al final, cerrar (proponer probar, resolver objeción de precio con valor/durabilidad, sumar cuerda o accesorio si calza).
Si el vendedor recomienda un producto equivocado para el perfil, dilo y sugiere cuál calzaría mejor según el manual.

Formato: máx 2 líneas. Empieza con ✅ si estuvo bien o ⚠️ si hay algo que mejorar. Si es ⚠️, da un ejemplo concreto de qué podría decir.`;
}

// ===== Gemini (plan gratuito de Google AI Studio) =====
const GEMINI_MAIN = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
const GEMINI_FAST = process.env.GEMINI_MODEL_FAST || 'gemini-2.5-flash-lite';

async function callGemini(apiKey, model, { system, messages, max_tokens, temperature }) {
  const gen = { maxOutputTokens: Math.max(max_tokens, 400), temperature: temperature ?? 0.9 };
  if (model.startsWith('gemini-2.5')) gen.thinkingConfig = { thinkingBudget: 0 };
  const body = {
    contents: messages.map(m => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content }] })),
    generationConfig: gen,
  };
  if (system) body.systemInstruction = { parts: [{ text: system }] };
  const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const e = new Error(data?.error?.message || `HTTP ${r.status}`);
    e.status = r.status;
    throw e;
  }
  const parts = data?.candidates?.[0]?.content?.parts || [];
  const text = parts.filter(p => !p.thought).map(p => p.text || '').join('').trim();
  if (!text) throw new Error('Respuesta vacía');
  return text;
}

// Usa Gemini si hay GEMINI_API_KEY (gratis); si no, Anthropic.
async function callAI(body) {
  const gKey = process.env.GEMINI_API_KEY;
  if (gKey) {
    const model = body.model === MODEL_FAST ? GEMINI_FAST : GEMINI_MAIN;
    try { return await callGemini(gKey, model, body); }
    catch (e) {
      if (e.status === 429 && model !== GEMINI_FAST) return callGemini(gKey, GEMINI_FAST, body);
      throw e;
    }
  }
  return callAnthropic(process.env.ANTHROPIC_API_KEY, body);
}

async function callAnthropic(apiKey, body) {
  const doFetch = (model) => fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({ ...body, model }),
  });
  let r = await doFetch(body.model);
  if (r.status === 404 || r.status === 400) {
    const err = await r.clone().json().catch(() => ({}));
    const msg = JSON.stringify(err).toLowerCase();
    if (msg.includes('model')) r = await doFetch(MODEL_FALLBACK);
  }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const e = new Error(data?.error?.message || `HTTP ${r.status}`);
    e.status = r.status;
    throw e;
  }
  return (data.content || []).map(c => c.text || '').join('').trim();
}

function cleanMessages(raw) {
  if (!Array.isArray(raw)) return null;
  const msgs = raw.slice(-MAX_TURNS)
    .filter(m => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
    .map(m => ({ role: m.role, content: m.content.slice(0, MAX_MSG_CHARS) }));
  return msgs;
}

// En el simulador: "assistant" = cliente (IA), "user" = vendedor.
// La API exige que la conversación parta con "user", así que anteponemos la entrada a la tienda.
function forCliente(msgs) {
  const out = [{ role: 'user', content: '(Entras a la tienda Wilson. El vendedor se te acerca.) Empieza tú la conversación.' }];
  for (const m of msgs) {
    const last = out[out.length - 1];
    if (last.role === m.role) last.content += '\n' + m.content; else out.push({ ...m });
  }
  return out;
}

function transcript(msgs, name) {
  return msgs.map(m => `${m.role === 'user' ? 'VENDEDOR' : name.toUpperCase() + ' (cliente)'}: ${m.content}`).join('\n');
}

export default async function handler(req, res) {
  const apiKey = process.env.GEMINI_API_KEY || process.env.ANTHROPIC_API_KEY;
  const ip = (req.headers['x-forwarded-for'] || '').split(',')[0].trim() || 'anon';

  // Chequeo de salud: GET /api/claude?check=1 → confirma key y modelos.
  const query = req.query || Object.fromEntries(new URL(req.url, 'http://x').searchParams);
  if (req.method === 'GET' && query.check) {
    if (!apiKey) return res.status(200).json({ key: false });
    if (rateLimited(ip)) return res.status(429).json({ error: 'rate' });
    if (process.env.GEMINI_API_KEY) {
      const out = { proveedor: 'gemini' };
      for (const model of [GEMINI_MAIN, GEMINI_FAST]) {
        try { await callGemini(process.env.GEMINI_API_KEY, model, { messages: [{ role: 'user', content: 'Di ok' }], max_tokens: 20 }); out[model] = 'ok'; }
        catch (e) { out[model] = `${e.status || ''}: ${String(e.message).slice(0, 160)}`; }
      }
      return res.status(200).json(out);
    }
    const out = { proveedor: 'anthropic', key: true };
    for (const model of [MODEL_MAIN, MODEL_FAST, MODEL_FALLBACK]) {
      const r = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
        body: JSON.stringify({ model, max_tokens: 5, messages: [{ role: 'user', content: 'Di ok' }] }),
      }).catch(() => null);
      if (!r) { out[model] = 'sin conexión'; continue; }
      const j = await r.json().catch(() => ({}));
      out[model] = r.ok ? 'ok' : `${r.status}: ${(j.error && j.error.message || '').slice(0, 160)}`;
    }
    return res.status(200).json(out);
  }

  if (req.method !== 'POST') return res.status(405).json({ error: 'Método no permitido' });
  if (!apiKey) return res.status(500).json({ error: 'Falta configurar la API key en Vercel' });
  if (rateLimited(ip)) return res.status(429).json({ error: 'Muchas solicitudes seguidas. Espera unos minutos.' });

  let body = req.body;
  if (typeof body === 'string') { try { body = JSON.parse(body); } catch { body = {}; } }
  const { task } = body || {};

  try {
    if (task === 'quiz') {
      const cat = QUIZ_CATS.includes(body.cat) ? body.cat : QUIZ_CATS[Math.floor(Math.random() * QUIZ_CATS.length)];
      const avoid = Array.isArray(body.avoid) ? body.avoid.slice(-8).map(s => String(s).slice(0, 40)) : [];
      const text = await callAI({
        model: MODEL_FAST, max_tokens: 600, temperature: 1,
        messages: [{ role: 'user', content: quizPrompt(cat, avoid) }],
      });
      const m = text.match(/\{[\s\S]*\}/);
      if (!m) throw new Error('Respuesta sin JSON');
      const q = JSON.parse(m[0]);
      if (!q.q || !Array.isArray(q.opts) || q.opts.length !== 4 || !(q.correct >= 0 && q.correct <= 3)) throw new Error('Pregunta inválida');
      return res.status(200).json({ question: q });
    }

    const p = PERFILES[Number(body.perfil)];
    if (!p) return res.status(400).json({ error: 'Perfil inválido' });
    const msgs = cleanMessages(body.messages);
    if (!msgs) return res.status(400).json({ error: 'Mensajes inválidos' });

    if (task === 'cliente') {
      const text = await callAI({
        model: MODEL_MAIN, max_tokens: 300,
        system: clientePrompt(p),
        messages: forCliente(msgs),
      });
      return res.status(200).json({ text });
    }

    if (task === 'coach') {
      if (!msgs.some(m => m.role === 'user')) return res.status(400).json({ error: 'No hay mensaje del vendedor' });
      const text = await callAI({
        model: MODEL_MAIN, max_tokens: 250,
        system: coachPrompt(p),
        messages: [{ role: 'user', content: `CONVERSACIÓN HASTA AHORA:\n${transcript(msgs, p.name)}\n\nEvalúa el último mensaje del VENDEDOR.` }],
      });
      return res.status(200).json({ text });
    }

    return res.status(400).json({ error: 'Tarea inválida' });
  } catch (e) {
    console.error('wilson-ai error', e.status, e.message);
    return res.status(502).json({ error: 'No se pudo conectar con la IA. Intenta de nuevo.' });
  }
}
