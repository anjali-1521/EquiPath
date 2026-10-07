const api = (path, opts) => fetch("/api" + path, opts).then(async r => {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
});

function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) k === "class" ? (n.className = v) : n.setAttribute(k, v);
  for (const kid of kids.flat()) n.append(kid instanceof Node ? kid : document.createTextNode(kid ?? ""));
  return n;
}
const $ = id => document.getElementById(id);
const fmtPct = x => Math.round(x * 100) + "%";

function debounce(fn, ms = 250) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

// ---- tabs ----
for (const [tab, view] of [["tab-researcher", "view-researcher"], ["tab-clinician", "view-clinician"]]) {
  $(tab).onclick = () => {
    for (const [t, v] of [["tab-researcher", "view-researcher"], ["tab-clinician", "view-clinician"]]) {
      $(t).classList.toggle("active", t === tab); $(v).hidden = v !== view;
    }
  };
}

// ---- searchable selects ----
function wireSearch(inputId, selectId, endpoint, label) {
  const load = async () => {
    const items = await api(`${endpoint}?q=${encodeURIComponent($(inputId).value)}&limit=50`);
    const sel = $(selectId); sel.replaceChildren();
    for (const it of items) sel.append(el("option", { value: it.id }, label(it)));
  };
  $(inputId).oninput = debounce(load); load();
}
wireSearch("disease-q", "disease-select", "/diseases",
  d => `${d.name}${d.custom ? " (rare, custom)" : ""} - ${d.degree} edges${d.untreated ? ", no known treatment" : ""}`);
wireSearch("cand-q", "cand-select", "/compounds", c => c.name + (c.ddi_covered ? "" : " (not covered by DDI model)"));
wireSearch("med-q", "med-select", "/compounds", c => c.name + (c.ddi_covered ? "" : " (not covered by DDI model)"));

$("lam").oninput = () => ($("lam-out").textContent = Number($("lam").value).toFixed(2));

// ---- researcher: repurposing ----
$("run-repurposing").onclick = async () => {
  const box = $("repurposing-result"); box.replaceChildren(el("p", {}, "Ranking..."));
  try {
    const id = $("disease-select").value;
    const r = await api(`/repurposing?disease_id=${encodeURIComponent(id)}&lam=${$("lam").value}&top_k=10`);
    const card = el("div", { class: "card" },
      el("h2", {}, `Candidates for ${r.disease.name}`),
      el("p", { class: "meta" }, `${r.disease.degree} edges in graph - ${r.disease.untreated ? "no known treatment" : "has known treatments"}`),
      r.notes.map(n => el("p", { class: "warn" }, n)),
      el("h3", {}, "Graph-evidence candidates"),
      el("p", { class: "hint" }, "Compounds with a concrete path to this disease in the knowledge graph. Specific to the disease, but only exists where the graph has the links."),
      r.evidence_candidates.length ? r.evidence_candidates.map((c, i) => el("div", { class: "cand" },
          el("h3", {}, `${i + 1}. ${c.compound_name}`),
          el("div", { class: "meta" }, `${c.num_paths} supporting path(s)`),
          el("ul", { class: "paths" }, c.evidence_paths.map(p => el("li", {}, p.text)))))
        : el("p", { class: "meta" }, "No compound has a graph path to this disease."),
      el("h3", {}, "Model-ranked candidates"),
      r.candidates.map((c, i) => el("div", { class: "cand" },
        el("h3", {}, `${i + 1}. ${c.compound_name}`),
        el("div", { class: "meta" }, `model score ${c.model_score.toFixed(2)} - raw rank #${c.raw_rank}`),
        c.has_evidence_path
          ? el("ul", { class: "paths" }, c.evidence_paths.map(p => el("li", {}, p.text)))
          : el("div", { class: "meta" }, "No supporting path in the graph - ranking rests on embedding similarity alone."))));
    box.replaceChildren(card);
  } catch (e) { box.replaceChildren(el("p", { class: "warn" }, "Error: " + e.message)); }
};

$("run-equity").onclick = async () => {
  const box = $("equity-result"); box.replaceChildren();
  try {
    const r = await api(`/equity?lam=${$("lam").value}`);
    const row = (name, a) => el("tr", {}, el("td", {}, name),
      el("td", {}, fmtPct(a.share_to_best_connected_20pct)), el("td", {}, fmtPct(a.share_to_untreated_diseases)),
      el("td", {}, String(a.distinct_diseases)));
    box.append(el("table", {},
      el("tr", {}, el("th", {}, "Ranking"), el("th", {}, "To best-connected 20% of diseases"),
        el("th", {}, "To untreated diseases"), el("th", {}, "Distinct diseases")),
      row("Raw model score", r.raw_ranking), row(`Neglect-aware (weight ${r.lam})`, r.neglect_aware_ranking)));
  } catch (e) { box.append(el("p", { class: "warn" }, "Error: " + e.message)); }
};

// ---- clinician: safety screen ----
const meds = new Map();
function renderMeds() {
  const ul = $("med-list"); ul.replaceChildren();
  for (const [id, name] of meds) {
    const rm = el("button", { type: "button" }, "remove"); rm.onclick = () => { meds.delete(id); renderMeds(); };
    ul.append(el("li", {}, name + " ", rm));
  }
}
$("add-med").onclick = () => {
  const sel = $("med-select"); if (!sel.value) return;
  meds.set(sel.value, sel.selectedOptions[0].textContent); renderMeds();
};
$("run-safety").onclick = async () => {
  const box = $("safety-result"); box.replaceChildren();
  try {
    const r = await api("/patient-safety", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ candidate: $("cand-select").value, current_medications: [...meds.keys()] }) });
    const label = { review_required: "Review required", caution: "Caution", incomplete: "Incomplete - some drugs not assessable", no_flags: "No flags" }[r.overall];
    box.append(el("div", { class: "card" },
      el("h2", {}, `${r.candidate}: ${label}`),
      r.checks.length ? el("table", {}, el("tr", {}, el("th", {}, "Current medication"), el("th", {}, "Level"),
          el("th", {}, "Score"), el("th", {}, "Shared side effects")),
        r.checks.map(c => el("tr", {}, el("td", {}, c.medication),
          el("td", {}, el("span", { class: "badge " + c.level }, c.level.replace("_", " "))),
          el("td", {}, c.interaction_score == null ? "-" : c.interaction_score.toFixed(2)),
          el("td", {}, `${c.shared_side_effects}${c.example_shared_side_effects.length ? " (e.g. " + c.example_shared_side_effects.slice(0, 3).join(", ") + ")" : ""}`))))
        : el("p", { class: "hint" }, "No current medications entered."),
      el("p", { class: "warn" }, r.disclaimer)));
  } catch (e) { box.append(el("p", { class: "warn" }, "Error: " + e.message)); }
};
