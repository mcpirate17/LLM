const PLOT_LAYOUT = {
  paper_bgcolor: "#ffffff",
  plot_bgcolor: "#fbfaf6",
  font: { color: "#2b2b33", size: 12 },
  margin: { l: 50, r: 20, t: 30, b: 45 },
  xaxis: { gridcolor: "#ece8df", zerolinecolor: "#ddd7ca" },
  yaxis: { gridcolor: "#ece8df", zerolinecolor: "#ddd7ca" },
};
// light-theme sequential scale (pale → saturated), reads well on paper bg
const HEAT_SCALE = [
  [0, "#fbfaf6"],
  [0.25, "#d9e9ff"],
  [0.5, "#8fb6f5"],
  [0.75, "#5b6cf0"],
  [1, "#3b2fb0"],
];
// resolve the live per-family accent so Plotly lines match the page tint
function accent() {
  return getComputedStyle(content).getPropertyValue("--accent").trim() || "#6c4cf1";
}

const content = document.getElementById("content");
const laneNav = document.getElementById("lane-nav");
let LANES = [];

async function getJSON(url) {
  // no-store: the probe endpoints are recomputed per request; a cached response
  // can otherwise leave a stale value on screen after the server restarts.
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

function el(tag, attrs = {}, children = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return n;
}

function setActive(view) {
  document.querySelectorAll(".nav-item").forEach((a) => {
    a.classList.toggle("active", a.dataset.view === view);
  });
}

// ---------- plain-English layer ----------

let PLAIN = localStorage.getItem("plain") !== "0"; // default ON
let rerender = null; // re-runs the current view when the toggle flips

function plainBox(text, label = "🔰 In plain English") {
  if (!PLAIN || !text) return null;
  return el("div", { class: "plainbox" }, [
    el("span", { class: "plainbox-label" }, label),
    el("div", { class: "plainbox-body", html: text }),
  ]);
}

function add(parent, node) {
  if (node) parent.appendChild(node);
}

// Always-on plain caption under a plot: what you're looking at + what "good" looks like.
// (Independent of the 🔰 toggle, which only gates the deeper analogy boxes.)
function caption(whatItIs, whatGood) {
  const box = el("div", { class: "caption" });
  box.appendChild(el("span", {}, `${whatItIs} `));
  if (whatGood) {
    box.appendChild(el("b", {}, "What good looks like: "));
    box.appendChild(el("span", {}, whatGood));
  }
  return box;
}

// "How it works, in plain words" — the four faithful facts from the invention record.
function factsCard(facts) {
  if (!facts?.length) return null;
  const card = el("div", { class: "card" });
  card.appendChild(el("h3", {}, "How it works — the four big ideas"));
  const grid = el("div", { class: "facts-grid" });
  for (const f of facts) {
    grid.appendChild(
      el("div", { class: "fact" }, [
        el("div", { class: "fact-label" }, `${f.emoji} ${f.label}`),
        el("div", { class: "fact-text" }, f.text),
      ]),
    );
  }
  card.appendChild(grid);
  return card;
}

function togglePlain() {
  PLAIN = !PLAIN;
  localStorage.setItem("plain", PLAIN ? "1" : "0");
  syncPlainBtn();
  if (rerender) rerender();
}

function syncPlainBtn() {
  const b = document.getElementById("plain-toggle");
  if (!b) return;
  b.textContent = PLAIN ? "🔰 Plain English: ON" : "🔰 Plain English: OFF";
  b.classList.toggle("on", PLAIN);
}

const GUIDE_MECH =
  "👉 These are the actual math recipes the lane follows. <br><br><b>'Write'</b> is how it " +
  "jots down notes in its memory after reading a word. <br><b>'Read'</b> is how it " +
  "flips back through those notes to find an answer. <br><br>Don't sweat the symbols! " +
  "The animated guide above and the plots below show you the 'vibe' of this math in action.";
const GUIDE_MIXING =
  "🎯 <b>Who affects whom?</b> Each row is an 'input poke' at a certain position. " +
  "The bright spots to the right show which future words 'felt' that poke.<br><br>" +
  "• <b>Diagonal strip:</b> The word only affects its neighbors (short-term memory).<br>" +
  "• <b>Wide bright block:</b> This word's influence carries far into the future (long-term memory).<br>" +
  "• <b>Empty top-left:</b> The model can't see the future—it only looks at the past!";
const GUIDE_TRACE =
  "🧠 <b>Watch a memory being born!</b> The left grid is the model's 'scratchpad.' " +
  "It starts blank and fills up as it reads. <br><br>Hit <b>▶ Play</b> and watch t=0 to t=15. " +
  "The red spikes on the right show <b>Surprise</b>: tall spikes mean 'I didn't see " +
  "that word coming!' <br><br>Notice when a word <b>repeats</b> (orange lines): a good " +
  "memory is less surprised the second time because it remembers the first!";
const GUIDE_SPECTRUM =
  "🔍 <b>One memory, three different pairs of glasses.</b> We're looking at the SAME " +
  "stored fact through different lenses:<br><br>" +
  "• <b>Grey (Mean):</b> Blends all notes together—a bit fuzzy.<br>" +
  "• <b>Red (Max):</b> Grabs only the single sharpest note—very precise.<br>" +
  "• <b>Blue (Learned):</b> This model taught ITSELF where to look. It found its " +
  "own sweet spot between fuzzy and sharp.";
const GUIDE_LIVE =
  "🏗️ <b>Watch the factory floor!</b> We're building and testing every design right now. " +
  "<br><br>• <b>Smoke:</b> Does it crash? (We want 'pass'!).<br>" +
  "• <b>Half-life:</b> How many words until it forgets a fact?<br>" +
  "• <b>Mixing:</b> Does it share info across the whole sentence, or stay local?";
const GUIDE_LEDGER =
  "🏆 <b>The Architecture Hall of Fame.</b> Every design here was invented, built from " +
  "scratch, and put through the same set of tests. <br><br>Each gets a <b>Score (0 to 1)</b> " +
  "that blends four things: does it run without crashing, does it actually learn, can it " +
  "pin one fact to one cue, and how often it wins a hide-and-seek recall test. <br><br>" +
  "• <b>Promoted ⭐:</b> cleared the bar and graduated to bigger, longer training runs.<br>" +
  "• <b>Archived:</b> a good try that didn't quite make the cut this round.<br><br>" +
  "Click any card that says <i>“Open the explainer”</i> to see exactly how it works.";
const GUIDE_FLOW =
  "🚀 <b>Token Journey:</b> Watch the tokens (words) fly through the architecture! " +
  "<br><br>As they pass each numbered stage, they get transformed, compared to " +
  "the past, or written into memory. The <b>highlighted box</b> shows which stage " +
  "is working on the tokens right now.";

const FLOW_TOKENS = ["the", "key", "turns", "again", "key", "answer"];

const ICONS = {
  token: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="9" x2="15" y2="9"/><line x1="9" y1="13" x2="15" y2="13"/><line x1="9" y1="17" x2="11" y2="17"/></svg>`,
  projection: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg>`,
  mixer: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>`,
  memory: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>`,
  output: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8L22 12L18 16"/><path d="M2 12H22"/><path d="M12 2L12 22"/></svg>`,
  surprise: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
  router: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>`,
  compress: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14h6v6H4z"/><path d="M14 4h6v6h-6z"/><path d="M2 2l20 20"/><path d="M22 2L2 22"/></svg>`,
  search: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>`,
  write: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>`,
};

function architectureProfile(card) {
  const id = card.lane_id || "";
  if (card.family === "attention") {
    return {
      title: "Attention flow",
      oneLine: "Each token looks back at the whole past and picks its favorites.",
      stages: [
        { name: "Input", icon: ICONS.token, detail: "Word turns into a vector" },
        { name: "Q/K/V", icon: ICONS.projection, detail: "Make labels & payloads" },
        { name: "Look back", icon: ICONS.search, detail: "Compare current to all past" },
        {
          name: id.includes("tropical") ? "Winner" : "Blend",
          icon: ICONS.mixer,
          detail: id.includes("sparsemax") ? "Drop zeros" : "Weighted average",
        },
        { name: "Output", icon: ICONS.output, detail: "Context-rich token" },
      ],
      callouts: [
        "No memory: it re-reads everything every time.",
        id.includes("tropical")
          ? "Winner-take-all makes it sharp."
          : "Weights decide what matters.",
        "Future words are invisible (causal).",
      ],
      tokens: FLOW_TOKENS,
    };
  }
  if (id === "causal_slot_router_memory") {
    return {
      title: "Slot-router flow",
      oneLine: "Tokens compete for a few precious memory drawers.",
      stages: [
        { name: "Input", icon: ICONS.token, detail: "Token arrives" },
        { name: "Router", icon: ICONS.router, detail: "Pick the right drawer" },
        { name: "Write", icon: ICONS.write, detail: "Update slot state" },
        { name: "Read", icon: ICONS.search, detail: "Pull from selected slots" },
        { name: "Output", icon: ICONS.output, detail: "Slot-informed token" },
      ],
      callouts: [
        "Routing separates different facts.",
        "Capacity is limited by the number of slots.",
        "Watch for routing collapse!",
      ],
      tokens: FLOW_TOKENS,
    };
  }
  if (id === "hierarchical_residual_compressor") {
    return {
      title: "Compressor flow",
      oneLine: "A pyramid of summaries at different timescales.",
      stages: [
        { name: "Input", icon: ICONS.token, detail: "Token arrives" },
        { name: "Fast", icon: ICONS.compress, detail: "Update every step" },
        { name: "Slow", icon: ICONS.compress, detail: "Update every 4+ steps" },
        { name: "Gate", icon: ICONS.mixer, detail: "Blend pyramid levels" },
        { name: "Output", icon: ICONS.output, detail: "Hierarchical context" },
      ],
      callouts: [
        "Slow levels keep the 'big picture'.",
        "Fast levels catch the details.",
        "Constant-sized memory for any length.",
      ],
      tokens: FLOW_TOKENS,
    };
  }
  if (id === "causal_fast_weight_memory") {
    return {
      title: "Fast-weight memory flow",
      oneLine: "A simple notebook that writes down everything.",
      stages: [
        { name: "Input", icon: ICONS.token, detail: "Token arrives" },
        { name: "Label", icon: ICONS.projection, detail: "Key + Value pair" },
        { name: "Write", icon: ICONS.write, detail: "Add to the sum" },
        { name: "Read", icon: ICONS.search, detail: "Retrieve with query" },
        { name: "Output", icon: ICONS.output, detail: "Smeared memory context" },
      ],
      callouts: [
        "Simple and very fast.",
        "Facts can smear together easily.",
        "The baseline for 'Surprise' versions.",
      ],
      tokens: FLOW_TOKENS,
    };
  }
  if (card.is_surprise_memory) {
    const semiring = id === "semiring_surprise_memory";
    const padic = id === "padic_surprise_memory";
    return {
      title: padic ? "p-adic surprise flow" : "Surprise-memory flow",
      oneLine: "Only writes when it sees something it didn't expect.",
      stages: [
        { name: "Input", icon: ICONS.token, detail: "Token arrives" },
        { name: "Predict", icon: ICONS.search, detail: "Check current memory" },
        { name: "Surprise", icon: ICONS.surprise, detail: "Measure error" },
        { name: "Write", icon: ICONS.write, detail: "Store the error only" },
        {
          name: semiring ? "Focus" : "Retrieve",
          icon: ICONS.mixer,
          detail: semiring ? "Learn mean vs max" : "Winner-takes-all",
        },
      ],
      callouts: [
        "Predicting well saves memory space.",
        "Repeats are less surprising (low error).",
        "Tropical read makes recall sharp.",
      ],
      tokens: FLOW_TOKENS,
    };
  }
  return {
    title: "General lane flow",
    oneLine: "Tokens transform as they pass through.",
    stages: [
      { name: "Input", icon: ICONS.token, detail: "Word arrives" },
      { name: "Project", icon: ICONS.projection, detail: "Hidden state" },
      { name: "Mix", icon: ICONS.mixer, detail: "Time info share" },
      { name: "State", icon: ICONS.memory, detail: "Update memory" },
      { name: "Output", icon: ICONS.output, detail: "Updated token" },
    ],
    callouts: [
      "Tokens move left to right.",
      "Each stage does specific math.",
      "The lane is a 'highway' for data.",
    ],
    tokens: FLOW_TOKENS,
  };
}

// ---------- sidebar ----------

async function buildSidebar() {
  const data = await getJSON("/api/lanes");
  LANES = data.lanes;
  const byFamily = {};
  for (const l of LANES) {
    byFamily[l.family] ||= [];
    byFamily[l.family].push(l);
  }
  laneNav.innerHTML = "";
  for (const [family, lanes] of Object.entries(byFamily)) {
    const sec = el("div", { class: "nav-section", "data-family": family });
    sec.appendChild(el("div", { class: "nav-head" }, family));
    for (const l of lanes) {
      const item = el("a", { class: "nav-item", "data-view": `lane:${l.lane_id}` }, [
        l.title,
        el("span", { class: "fam" }, l.class_name),
      ]);
      item.onclick = () => showLane(l.lane_id);
      sec.appendChild(item);
    }
    laneNav.appendChild(sec);
  }
  document.querySelector('[data-view="live"]').onclick = showLive;
  document.querySelector('[data-view="ledger"]').onclick = showLedger;
}

// ---------- lane explainer ----------

function badge(text, cls = "") {
  return el("span", { class: `badge ${cls}` }, text);
}

async function showLane(laneId) {
  setActive(`lane:${laneId}`);
  rerender = () => showLane(laneId);
  content.innerHTML = "";
  content.appendChild(el("div", { class: "spinner" }, "Building & probing the lane…"));

  let card;
  try {
    card = await getJSON(`/api/lanes/${laneId}`);
  } catch (e) {
    content.innerHTML = `<div class="empty">Failed: ${e.message}</div>`;
    return;
  }
  content.innerHTML = "";
  content.dataset.family = card.family || ""; // retints the page accent

  // Header card
  const head = el("div", { class: "card" });
  head.appendChild(el("h2", {}, card.title));
  const smokeOk = card.smoke?.all_finite;
  head.appendChild(
    el("div", { class: "badges" }, [
      badge(card.class_name, "k"),
      badge(card.family),
      badge(card.complexity, "k"),
      badge(`${card.params.toLocaleString()} params`, "k"),
      badge(smokeOk ? "smoke ✓ finite" : "smoke ✗", smokeOk ? "good" : "bad"),
    ]),
  );
  add(head, plainBox(card.plain));
  head.appendChild(el("p", { class: "summary" }, card.summary));
  for (const n of card.notes || []) head.appendChild(el("span", { class: "badge" }, n));
  content.appendChild(head);

  renderArchitectureFlow(card);

  // How it works — the four faithful "big idea" facts (skipped if undescribed)
  add(content, factsCard(card.facts));

  // Watch it remember — the headline faithful recall demo (memory lanes only)
  if (card.supports_recall) renderRecall(laneId, card.family);

  // Equations
  const eq = el("div", { class: "card" });
  eq.appendChild(el("h3", {}, "Mechanism"));
  add(eq, plainBox(GUIDE_MECH));
  eq.appendChild(el("div", { class: "eq-label" }, "Write (state update)"));
  eq.appendChild(el("pre", { class: "eq" }, card.write_eq));
  eq.appendChild(el("div", { class: "eq-label" }, "Read (retrieval algebra)"));
  eq.appendChild(el("pre", { class: "eq" }, card.read_eq));
  content.appendChild(eq);

  // Token mixing
  renderMixing(laneId);

  // Surprise trace
  if (card.supports_trace) renderTrace(laneId);

  // Algebra spectrum
  if (card.supports_spectrum) renderSpectrum(laneId);
}

function renderArchitectureFlow(card) {
  const profile = architectureProfile(card);
  const cardEl = el("div", { class: "card flow-card" });
  cardEl.appendChild(el("h3", {}, profile.title));

  // Narrative box for "cartoon" style teaching
  const narrative = el("div", { class: "narrative-box" }, profile.oneLine);
  cardEl.appendChild(narrative);

  add(cardEl, plainBox(GUIDE_FLOW, "Animated teaching aid"));

  const stageRail = el("div", { class: "flow-rail" });
  const track = el("div", { class: "flow-track", "aria-hidden": "true" });

  // Particles are now little token chips with text
  for (let i = 0; i < profile.tokens.length; i += 1) {
    track.appendChild(
      el(
        "div",
        {
          class: `flow-particle p${i % 6}`,
          style: `--delay:${(i * 0.96).toFixed(2)}s; --lane:${i % 3};`,
        },
        profile.tokens[i],
      ),
    );
  }
  stageRail.appendChild(track);

  const stages = el("div", { class: "flow-stages" });
  profile.stages.forEach((stage, idx) => {
    const stageEl = el(
      "div",
      {
        class: "flow-stage",
        style: `--stage-delay:${(idx * 0.55).toFixed(2)}s;`,
        title: `Step ${idx + 1}: ${stage.name} — ${stage.detail}`,
      },
      [
        el("div", { class: "stage-index" }, String(idx + 1)),
        el("div", { class: "stage-icon", html: stage.icon }),
        el("div", { class: "stage-name" }, stage.name),
        el("div", { class: "stage-detail" }, stage.detail),
      ],
    );
    stages.appendChild(stageEl);
  });
  stageRail.appendChild(stages);
  cardEl.appendChild(stageRail);

  const lower = el("div", { class: "flow-lower" });
  const tokenStrip = el("div", { class: "token-strip" });
  profile.tokens.forEach((token, idx) => {
    tokenStrip.appendChild(
      el(
        "span",
        {
          class: `token-chip ${idx === 1 || idx === 4 ? "repeat" : ""}`,
          style: `--token-delay:${(idx * 0.28).toFixed(2)}s;`,
        },
        token,
      ),
    );
  });
  lower.appendChild(tokenStrip);

  const callouts = el("div", { class: "flow-callouts" });
  profile.callouts.forEach((text, idx) => {
    callouts.appendChild(
      el(
        "div",
        {
          class: "flow-callout",
          style: `--callout-delay:${(idx * 0.9 + 0.3).toFixed(2)}s;`,
        },
        text,
      ),
    );
  });
  lower.appendChild(callouts);
  cardEl.appendChild(lower);
  content.appendChild(cardEl);
}

async function renderRecall(laneId) {
  const card = el("div", { class: "card" });
  card.appendChild(el("h3", {}, "Watch it remember — can it keep facts straight?"));
  card.appendChild(
    caption(
      "We file a handful of facts into the memory, then ask for each one back while the " +
        "others are still crowded in there. Each bar is how cleanly a fact survived the crowd.",
      "tall green bars — the model pulled back the right fact without the others smudging it.",
    ),
  );
  const body = el("div", {});
  card.appendChild(body);
  content.appendChild(card);

  let s;
  try {
    s = await getJSON(`/api/lanes/${laneId}/recall`);
  } catch (e) {
    body.appendChild(el("div", { class: "empty" }, `recall demo failed: ${e.message}`));
    return;
  }

  const pct = Math.round(s.mean_retention * 100);
  body.appendChild(
    el("div", { class: "recall-head" }, [
      el("div", { class: "recall-stat" }, [
        el("div", { class: "recall-pct" }, `${pct}%`),
        el("div", { class: "recall-pct-label" }, "of each memory stays clean in a crowd"),
      ]),
      el("div", { class: "badges" }, [
        badge(s.read_kind, "good"),
        badge(
          `${s.clean_count} / ${s.n_facts} facts stayed clean`,
          s.clean_count === s.n_facts ? "good" : "k",
        ),
      ]),
    ]),
  );

  body.appendChild(el("div", { class: "recall-step" }, "① Filed into the memory"));
  const notebook = el("div", { class: "notebook" });
  s.facts.forEach((f, i) => {
    notebook.appendChild(
      el("div", { class: "note", style: `--d:${(i * 0.16).toFixed(2)}s` }, f.label),
    );
  });
  body.appendChild(notebook);

  body.appendChild(el("div", { class: "recall-step" }, "② Asked for each one back"));
  const rows = el("div", { class: "recall-rows" });
  s.results.forEach((r, i) => {
    const p = Math.max(0, Math.min(100, Math.round(r.retention * 100)));
    rows.appendChild(
      el("div", { class: "recall-row", style: `--d:${(i * 0.2 + 0.3).toFixed(2)}s` }, [
        el("div", { class: "recall-label" }, r.label),
        el(
          "div",
          { class: "bar-track" },
          el("div", { class: `bar-fill ${r.clean ? "clean" : "muddied"}`, style: `--w:${p}%` }),
        ),
        el(
          "div",
          { class: `recall-verdict ${r.clean ? "ok" : "no"}` },
          r.clean ? `✓ clean ${p}%` : `~ muddied ${p}%`,
        ),
      ]),
    );
  });
  body.appendChild(rows);

  if (laneId !== "causal_fast_weight_memory") {
    try {
      const base = await getJSON("/api/lanes/causal_fast_weight_memory/recall");
      const bp = Math.round(base.mean_retention * 100);
      body.appendChild(
        el("div", {
          class: "caption",
          html:
            `<b>Side by side:</b> this design keeps <b>${pct}%</b> clean (${s.read_kind}); ` +
            `the plain fuzzy-notebook baseline keeps <b>${bp}%</b>. ` +
            (pct >= bp
              ? "A sharper read means less smudging between facts."
              : "Here the simpler memory held up — worth a longer look."),
        }),
      );
    } catch (_e) {
      /* baseline comparison is optional */
    }
  }
}

async function renderMixing(laneId) {
  const card = el("div", { class: "card" });
  card.appendChild(el("h3", {}, "Token mixing — how far a word's influence reaches"));
  card.appendChild(
    caption(
      "We nudge one word and watch which other words change. Each row is the nudged word; " +
        "bright cells to its right are the later words that felt it.",
      "a clear lower-left triangle (it only affects the future, never the past) — and how " +
        "far the brightness spreads tells you short- vs long-term memory.",
    ),
  );
  add(card, plainBox(GUIDE_MIXING));
  const stats = el("div", { class: "badges" });
  card.appendChild(stats);
  const row = el("div", { class: "row" });
  const heat = el("div", { class: "col" });
  const heatPlot = el("div", { class: "plot" });
  heat.appendChild(heatPlot);
  const line = el("div", { class: "col" });
  const linePlot = el("div", { class: "plot" });
  line.appendChild(linePlot);
  row.appendChild(heat);
  row.appendChild(line);
  card.appendChild(row);
  content.appendChild(card);

  let m;
  try {
    m = await getJSON(`/api/lanes/${laneId}/mixing`);
  } catch (e) {
    card.appendChild(el("div", { class: "empty" }, `mixing failed: ${e.message}`));
    return;
  }
  stats.appendChild(
    badge(
      m.mix_half_life === null ? "half-life: ∞ (no decay)" : `half-life: ${m.mix_half_life}`,
      "k",
    ),
  );
  stats.appendChild(
    badge(m.mixes_globally ? "mixes globally" : "local mixing", m.mixes_globally ? "good" : ""),
  );
  if (m.is_pure_local) stats.appendChild(badge("pure-local", "warn"));

  Plotly.newPlot(
    heatPlot,
    [
      {
        z: m.matrix,
        type: "heatmap",
        colorscale: HEAT_SCALE,
        colorbar: { title: "Δ" },
      },
    ],
    {
      ...PLOT_LAYOUT,
      title: "How much an early word changes a later one",
      xaxis: { ...PLOT_LAYOUT.xaxis, title: "a later word in the sentence →" },
      yaxis: { ...PLOT_LAYOUT.yaxis, title: "↓ an earlier word", autorange: "reversed" },
    },
    { displayModeBar: false, responsive: true },
  );

  Plotly.newPlot(
    linePlot,
    [
      {
        y: m.decay,
        x: m.decay.map((_, i) => i),
        type: "scatter",
        mode: "lines+markers",
        line: { color: accent() },
        fill: "tozeroy",
        fillcolor: "rgba(108,76,241,.10)",
      },
    ],
    {
      ...PLOT_LAYOUT,
      title: "How fast the influence fades with distance",
      xaxis: { ...PLOT_LAYOUT.xaxis, title: "words away from the nudge" },
      yaxis: { ...PLOT_LAYOUT.yaxis, title: "how much it still changes" },
    },
    { displayModeBar: false, responsive: true },
  );
}

async function renderTrace(laneId) {
  const card = el("div", { class: "card" });
  card.appendChild(el("h3", {}, "Memory trace — watch the scratchpad fill, word by word"));
  card.appendChild(
    caption(
      "Hit ▶ Play. The left grid is the model's memory, filling in as it reads. The right " +
        "chart is its surprise — how wrong its guess was — at each word. One word is repeated " +
        "on purpose (orange line).",
      "the surprise spike should be smaller the second time the word appears — proof the " +
        "model remembered it.",
    ),
  );
  add(card, plainBox(GUIDE_TRACE));
  const ctrl = el("div", { class: "controls" });
  const playBtn = el("button", {}, "▶ Play");
  const slider = el("input", { type: "range", min: "0", max: "0", value: "0" });
  const label = el("span", { class: "badge k" }, "t = 0");
  ctrl.appendChild(playBtn);
  ctrl.appendChild(slider);
  ctrl.appendChild(label);
  card.appendChild(ctrl);
  const row = el("div", { class: "row" });
  const memCol = el("div", { class: "col" });
  const memPlot = el("div", { class: "plot" });
  memCol.appendChild(memPlot);
  const errCol = el("div", { class: "col" });
  const errPlot = el("div", { class: "plot" });
  errCol.appendChild(errPlot);
  row.appendChild(memCol);
  row.appendChild(errCol);
  card.appendChild(row);
  content.appendChild(card);

  let tr;
  try {
    tr = await getJSON(`/api/lanes/${laneId}/trace`);
  } catch (e) {
    card.appendChild(el("div", { class: "empty" }, `trace failed: ${e.message}`));
    return;
  }
  const frames = tr.frames;
  slider.max = String(frames.length - 1);

  const errSeries = frames.map((f) => f.error_norm);
  const memSeries = frames.map((f) => f.memory_norm);
  const xs = frames.map((f) => f.t);
  const markers = [tr.repeat_src, tr.repeat_dst];

  function drawMem(t) {
    Plotly.react(
      memPlot,
      [
        {
          z: frames[t].memory,
          type: "heatmap",
          colorscale: HEAT_SCALE,
          zmin: -1,
          zmax: 1,
          colorbar: { title: "M" },
        },
      ],
      {
        ...PLOT_LAYOUT,
        title: `the memory grid after reading word ${t}`,
        xaxis: { ...PLOT_LAYOUT.xaxis, title: "what is stored (value cell)" },
        yaxis: {
          ...PLOT_LAYOUT.yaxis,
          title: "where it's filed (key cell)",
          autorange: "reversed",
        },
      },
      { displayModeBar: false, responsive: true },
    );
  }

  Plotly.newPlot(
    errPlot,
    [
      {
        y: errSeries,
        x: xs,
        name: "surprise (how wrong the guess was)",
        type: "scatter",
        mode: "lines+markers",
        line: { color: "#e05656" },
      },
      {
        y: memSeries,
        x: xs,
        name: "memory fullness",
        type: "scatter",
        mode: "lines",
        line: { color: "#11a36b", dash: "dot" },
        yaxis: "y2",
      },
    ],
    {
      ...PLOT_LAYOUT,
      title: "surprise (red) and how full the memory is (green)",
      xaxis: { ...PLOT_LAYOUT.xaxis, title: "word number as it reads left → right" },
      yaxis: { ...PLOT_LAYOUT.yaxis, title: "surprise (how wrong the guess was)" },
      yaxis2: { title: "memory fullness", overlaying: "y", side: "right", showgrid: false },
      legend: { orientation: "h", y: 1.15 },
      shapes: markers.map((mk) => ({
        type: "line",
        x0: mk,
        x1: mk,
        yref: "paper",
        y0: 0,
        y1: 1,
        line: { color: "#f0a020", width: 1, dash: "dash" },
      })),
      annotations: [
        {
          x: tr.repeat_src,
          yref: "paper",
          y: 1.02,
          text: "token first seen",
          showarrow: false,
          font: { color: "#f0a020", size: 10 },
        },
        {
          x: tr.repeat_dst,
          yref: "paper",
          y: 1.02,
          text: "same token repeats",
          showarrow: false,
          font: { color: "#f0a020", size: 10 },
        },
      ],
    },
    { displayModeBar: false, responsive: true },
  );

  let cur = 0,
    timer = null;
  function go(t) {
    cur = t;
    slider.value = String(t);
    label.textContent = `t = ${t}`;
    drawMem(t);
  }
  slider.oninput = () => go(Number(slider.value));
  playBtn.onclick = () => {
    if (timer) {
      clearInterval(timer);
      timer = null;
      playBtn.textContent = "▶ Play";
      return;
    }
    playBtn.textContent = "⏸ Pause";
    timer = setInterval(() => {
      let t = cur + 1;
      if (t > frames.length - 1) {
        t = 0;
      }
      go(t);
    }, 450);
  };
  go(0);
}

async function renderSpectrum(laneId) {
  const card = el("div", { class: "card" });
  card.appendChild(el("h3", {}, "Focus knob — how sharply this model reads its memory"));
  card.appendChild(
    caption(
      "The same stored memory, read three ways: a fuzzy average (grey), a sharp single-best " +
        "pick (red), and where this model actually set its own focus knob (colored line).",
      "the learned line sitting wherever the data wanted it — proof the model tuned its own " +
        "sharpness instead of being hard-wired.",
    ),
  );
  add(card, plainBox(GUIDE_SPECTRUM));
  const plot = el("div", { class: "plot short" });
  card.appendChild(plot);
  const note = el("div", { class: "badges" });
  card.appendChild(note);
  content.appendChild(card);

  let s;
  try {
    s = await getJSON(`/api/lanes/${laneId}/spectrum`);
  } catch (e) {
    card.appendChild(el("div", { class: "empty" }, `spectrum failed: ${e.message}`));
    return;
  }
  if (s.beta !== null && s.beta !== undefined) {
    note.appendChild(badge(`focus knob β = ${s.beta.toFixed(2)}`, "good"));
    note.appendChild(
      badge(
        s.beta > 8
          ? "turned up sharp (picks the single best note)"
          : s.beta < 0.5
            ? "turned down fuzzy (blends notes)"
            : "balanced between sharp and fuzzy",
        "k",
      ),
    );
  }
  Plotly.newPlot(
    plot,
    [
      {
        y: s.mean,
        x: s.dims,
        name: "fuzzy average (blends every note)",
        type: "scatter",
        mode: "lines+markers",
        line: { color: "#9b9486" },
      },
      {
        y: s.learned,
        x: s.dims,
        name: "this model's learned focus",
        type: "scatter",
        mode: "lines+markers",
        line: { color: accent(), width: 3 },
      },
      {
        y: s.max,
        x: s.dims,
        name: "sharp pick (single best note)",
        type: "scatter",
        mode: "lines+markers",
        line: { color: "#e05656" },
      },
    ],
    {
      ...PLOT_LAYOUT,
      title: "what comes out of memory, read three different ways",
      xaxis: { ...PLOT_LAYOUT.xaxis, title: "memory cell" },
      yaxis: { ...PLOT_LAYOUT.yaxis, title: "value read out" },
      legend: { orientation: "h", y: 1.18 },
    },
    { displayModeBar: false, responsive: true },
  );
}

// ---------- live grading ----------

function showLive() {
  setActive("live");
  rerender = showLive;
  content.innerHTML = "";
  content.dataset.family = "";
  const card = el("div", { class: "card" });
  card.appendChild(el("h3", {}, "Live grading — every lane built & tested in real time"));
  card.appendChild(
    el(
      "div",
      { class: "hint" },
      "Streams over Server-Sent Events. Each lane is instantiated, smoke-tested " +
        "(forward/backward/finite) and probed for mixing — the same intrinsic checks the fab grader runs.",
    ),
  );
  add(card, plainBox(GUIDE_LIVE));
  const btn = el("button", {}, "▶ Run live grading");
  card.appendChild(btn);
  const status = el("span", { class: "badge k", style: "margin-left:10px" }, "idle");
  card.appendChild(status);
  const table = el("table");
  table.appendChild(
    el(
      "thead",
      {},
      el("tr", {}, [
        el("th", {}, "Lane"),
        el("th", {}, "Family"),
        el("th", { class: "num" }, "Params"),
        el("th", {}, "Smoke"),
        el("th", { class: "num" }, "Half-life"),
        el("th", {}, "Mixing"),
      ]),
    ),
  );
  const tbody = el("tbody", {});
  table.appendChild(tbody);
  card.appendChild(table);
  content.appendChild(card);

  btn.onclick = () => {
    btn.disabled = true;
    tbody.innerHTML = "";
    status.textContent = "running…";
    const es = new EventSource("/api/run/stream");
    es.addEventListener("grading", (e) => {
      const d = JSON.parse(e.data);
      status.textContent = `grading ${d.title}…`;
    });
    es.addEventListener("graded", (e) => {
      const d = JSON.parse(e.data);
      const pass = d.smoke_pass;
      tbody.appendChild(
        el("tr", {}, [
          el("td", {}, d.title),
          el("td", {}, d.family || ""),
          el("td", { class: "num" }, (d.params || 0).toLocaleString()),
          el(
            "td",
            {},
            el("span", { class: `pill ${pass ? "pass" : "fail"}` }, pass ? "pass" : "fail"),
          ),
          el(
            "td",
            { class: "num" },
            d.mix_half_life === null || d.mix_half_life === undefined
              ? "∞"
              : String(d.mix_half_life),
          ),
          el(
            "td",
            {},
            d.error ? `err: ${d.error.slice(0, 40)}` : d.mixes_globally ? "global" : "local",
          ),
        ]),
      );
    });
    es.addEventListener("done", () => {
      status.textContent = "done ✓";
      btn.disabled = false;
      es.close();
    });
    es.onerror = () => {
      status.textContent = "stream error";
      btn.disabled = false;
      es.close();
    };
  };
}

// ---------- ledger replay ----------

const MEDALS = ["🥇", "🥈", "🥉"];

function trophyCard(p, rank) {
  const card = el("div", {
    class: `trophy ${p.lane_id ? "clickable" : ""}`,
    "data-family": p.family || "other",
    style: `--i:${(rank * 0.05).toFixed(2)}s`,
  });
  if (rank < MEDALS.length) {
    card.appendChild(el("div", { class: "trophy-rank" }, MEDALS[rank]));
  }
  card.appendChild(
    el("div", { class: "trophy-score", html: `${p.best_score.toFixed(3)} <small>/ 1.000</small>` }),
  );
  card.appendChild(el("h4", {}, p.title || p.name));
  card.appendChild(el("div", { class: "fam-tag" }, `${p.family} · ${p.category}`));

  for (const s of p.score_story || []) {
    card.appendChild(
      el("div", { class: "score-line", title: s.plain }, [
        el("span", { class: `tick ${s.ok ? "ok" : "no"}` }, s.ok ? "✓" : "✕"),
        el("span", { class: "sl-label" }, s.label),
      ]),
    );
  }

  const promoted = p.promotion_status === "promoted";
  card.appendChild(
    el(
      "span",
      { class: `promo ${promoted ? "promoted" : "archived"}` },
      promoted ? "⭐ PROMOTED" : "ARCHIVED",
    ),
  );
  if (p.lane_id) {
    card.appendChild(el("div", { class: "open-hint" }, "→ Open the explainer"));
    card.onclick = () => showLane(p.lane_id);
  }
  return card;
}

async function showLedger() {
  setActive("ledger");
  rerender = showLedger;
  content.dataset.family = "";
  content.innerHTML = "";
  content.appendChild(el("div", { class: "spinner" }, "Opening the Hall of Fame…"));
  let data;
  try {
    data = await getJSON("/api/ledger");
  } catch (e) {
    content.innerHTML = `<div class="empty">Failed to load the Hall of Fame: ${e.message}</div>`;
    return;
  }
  content.innerHTML = "";

  const head = el("div", { class: "card" });
  head.appendChild(el("h2", {}, "🏆 The Hall of Fame"));
  const summaryText =
    `These are the <b>${data.proposals.length}</b> designs that earned a place — each one ` +
    `built, stress-tested, and scored. <b>${data.promoted_count}</b> were <b>Promoted</b>: ` +
    `judged good enough to graduate to bigger, longer training runs. ` +
    `The top scorers are the same designs you can explore in the sidebar.`;
  head.appendChild(el("div", { class: "narrative-box", html: summaryText }));
  add(head, plainBox(GUIDE_LEDGER, "What the scores mean"));
  content.appendChild(head);

  const shelfCard = el("div", { class: "card" });
  shelfCard.appendChild(el("h3", {}, "The trophy shelf — ranked by score"));
  shelfCard.appendChild(
    el("div", {
      class: "caption",
      html:
        "<b>Score (0–1)</b> blends four checks: does it run, does it learn, can it bind a fact, " +
        "and how well does it pass a needle-in-a-haystack recall test. Hover a ✓/✕ for what it means.",
    }),
  );
  const shelf = el("div", { class: "shelf" });
  data.proposals.forEach((p, i) => {
    shelf.appendChild(trophyCard(p, i));
  });
  shelfCard.appendChild(shelf);
  content.appendChild(shelfCard);
}

const plainToggle = document.getElementById("plain-toggle");
if (plainToggle) plainToggle.onclick = togglePlain;
syncPlainBtn();

buildSidebar().catch((e) => {
  content.innerHTML = `<div class="empty">Failed to load lanes: ${e.message}</div>`;
});
