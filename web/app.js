/* SDSS Galaxy Morphology & Scaling Relations Atlas — Interactive Plotly.js Dashboard */

const PLOT_LAYOUT_BASE = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(15, 23, 42, 0.55)",
  font: { family: "Inter, sans-serif", color: "#cbd5e1", size: 12 },
  margin: { l: 58, r: 22, t: 18, b: 48 },
  legend: { orientation: "h", y: 1.12, x: 0, bgcolor: "rgba(0,0,0,0)" },
  hovermode: "closest",
};

const PLOT_CONFIG = { responsive: true, displaylogo: false };

const state = {
  data: null,
  filtered: [],
  selectedGalaxy: null,
  inspectorMode: "rgb",
  galleryPage: 0,
  galleryPageSize: 24,
  activeCurveModel: "tf",
  activeCmModel: "tf",
};

async function loadAtlasData() {
  if (window.ATLAS_DATA) {
    return window.ATLAS_DATA;
  }
  const resp = await fetch("data/atlas_data.json");
  if (!resp.ok) {
    throw new Error(`Failed to load atlas_data.json (${resp.status})`);
  }
  return resp.json();
}

function fmt(val, digits = 3) {
  return val === null || val === undefined || Number.isNaN(val) ? "—" : Number(val).toFixed(digits);
}

function pct(val) {
  return val === null || val === undefined || Number.isNaN(val) ? "—" : `${(Number(val) * 100).toFixed(1)}%`;
}

function renderHeroAndKPIs(data) {
  const s = data.summary;
  const m = data.models || {};
  const cosmo = data.project.cosmology || {};

  document.getElementById("hero-meta-badges").innerHTML = `
    <span class="meta-badge">${data.project.data_release}</span>
    <span class="meta-badge">${cosmo.model || "FlatLambdaCDM"} (H₀=${cosmo.H0_km_s_Mpc || 70}, Ωₘ₀=${cosmo.Om0 || 0.3})</span>
    <span class="meta-badge">0.02 ≤ z ≤ 0.10 · r &lt; 17.77</span>
  `;

  const rfAcc = m.rf ? pct(m.rf.test_accuracy) : "—";
  const astroAcc = m.astronn ? pct(m.astronn.test_accuracy) : "—";
  const tfAcc = m.tf ? pct(m.tf.test_accuracy) : "—";

  document.getElementById("kpi-strip").innerHTML = `
    <div class="kpi-card">
      <div class="kpi-label">Clean SDSS Galaxies</div>
      <div class="kpi-value">${s.total_galaxies.toLocaleString()}</div>
      <div class="kpi-sub">${s.early_count.toLocaleString()} early · ${s.late_count.toLocaleString()} late</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Downloaded Cutouts</div>
      <div class="kpi-value">${s.rgb_cutouts_count.toLocaleString()}</div>
      <div class="kpi-sub">128×128 RGB + ${s.fits_showcase_count} FITS Lupton</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Galaxy Zoo Matches</div>
      <div class="kpi-value">${s.zoo_matched_count.toLocaleString()}</div>
      <div class="kpi-sub">${pct(s.zoo_matched_count / Math.max(1, s.total_galaxies))} human consensus</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Random Forest (Tabular)</div>
      <div class="kpi-value">${rfAcc}</div>
      <div class="kpi-sub">ROC-AUC ${m.rf ? fmt(m.rf.roc_auc, 3) : "—"}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">astroNN Galaxy10CNN</div>
      <div class="kpi-value">${astroAcc}</div>
      <div class="kpi-sub">5-class Hubble · MC Dropout</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Custom TensorFlow CNN</div>
      <div class="kpi-value">${tfAcc}</div>
      <div class="kpi-sub">5-class Hubble · 360° Aug</div>
    </div>
  `;

  document.getElementById("footer-timestamp").textContent = `Exported ${data.generated_at}`;
}

function applyFilters() {
  const morph = document.getElementById("filter-morph").value;
  const colorPop = document.getElementById("filter-color-pop").value;
  const hubble = document.getElementById("filter-hubble").value;
  const zMax = parseFloat(document.getElementById("filter-zmax").value);
  const mMin = parseFloat(document.getElementById("filter-mmin").value);
  const hasImg = document.getElementById("filter-has-image").checked;

  document.getElementById("val-zmax").textContent = zMax.toFixed(3);
  document.getElementById("val-mmin").textContent = mMin.toFixed(1);

  state.filtered = state.data.galaxies.filter((g) => {
    if (morph !== "all" && g.mp !== morph) return false;
    if (colorPop !== "all" && g.cp !== colorPop) return false;
    if (hubble !== "all" && (!g.pred.tf || g.pred.tf.cls !== hubble)) return false;
    if (g.z > zMax) return false;
    if (g.lm < mMin) return false;
    if (hasImg && !g.img) return false;
    return true;
  });

  document.getElementById("filtered-count-pill").textContent =
    `${state.filtered.length.toLocaleString()} / ${state.data.galaxies.length.toLocaleString()} galaxies`;

  state.galleryPage = 0;
  renderExplorerPlots();
  renderGallery();
  renderCatalogTable();
}

function bindScatterClick(divId) {
  const el = document.getElementById(divId);
  el.removeAllListeners?.("plotly_click");
  el.on("plotly_click", (evt) => {
    const pt = evt.points?.[0];
    const id = pt?.customdata;
    if (!id) return;
    const found = state.data.galaxies.find((g) => g.id === String(id));
    if (found) selectGalaxy(found);
  });
}

function renderExplorerPlots() {
  const early = state.filtered.filter((g) => g.mp === "early");
  const late = state.filtered.filter((g) => g.mp === "late");

  // 1. Color-Magnitude Diagram
  const mrMin = Math.min(...state.data.galaxies.map((g) => g.mr));
  const mrMax = Math.max(...state.data.galaxies.map((g) => g.mr));
  const xCmd = [mrMin, mrMax];
  const yCmd = xCmd.map((mr) => -0.025 * (mr + 20.0) + 0.72);

  Plotly.newPlot(
    "plot-cmd",
    [
      {
        name: "Early-type",
        type: "scattergl",
        mode: "markers",
        x: early.map((g) => g.mr),
        y: early.map((g) => g.gr),
        customdata: early.map((g) => g.id),
        marker: { color: "#ef4444", size: 5.5, opacity: 0.65 },
        hovertemplate: "objID %{customdata}<br>M_r=%{x:.2f}<br>g-r=%{y:.2f}<extra>Early</extra>",
      },
      {
        name: "Late-type",
        type: "scattergl",
        mode: "markers",
        x: late.map((g) => g.mr),
        y: late.map((g) => g.gr),
        customdata: late.map((g) => g.id),
        marker: { color: "#38bdf8", size: 5.5, opacity: 0.65 },
        hovertemplate: "objID %{customdata}<br>M_r=%{x:.2f}<br>g-r=%{y:.2f}<extra>Late</extra>",
      },
      {
        name: "Red / Blue cut",
        type: "scatter",
        mode: "lines",
        x: xCmd,
        y: yCmd,
        line: { color: "#f8fafc", width: 1.8, dash: "dash" },
        hoverinfo: "skip",
      },
    ],
    {
      ...PLOT_LAYOUT_BASE,
      xaxis: { title: "Absolute Magnitude M_r (AB mag)", autorange: "reversed", gridcolor: "rgba(148,163,184,0.12)" },
      yaxis: { title: "Optical Color g − r (mag)", range: [-0.1, 1.5], gridcolor: "rgba(148,163,184,0.12)" },
    },
    PLOT_CONFIG
  );
  bindScatterClick("plot-cmd");

  // 2. Mass-Size Relation + Power-law fits
  const fits = state.data.summary.mass_size_fits || {};
  const xFit = [9.0, 9.5, 10.0, 10.5, 11.0, 11.4];
  const msTraces = [
    {
      name: "Early-type",
      type: "scattergl",
      mode: "markers",
      x: early.map((g) => g.lm),
      y: early.map((g) => g.re),
      customdata: early.map((g) => g.id),
      marker: { color: "#ef4444", size: 5.5, opacity: 0.55 },
      hovertemplate: "objID %{customdata}<br>logM=%{x:.2f}<br>R50=%{y:.2f} kpc<extra>Early</extra>",
    },
    {
      name: "Late-type",
      type: "scattergl",
      mode: "markers",
      x: late.map((g) => g.lm),
      y: late.map((g) => g.re),
      customdata: late.map((g) => g.id),
      marker: { color: "#38bdf8", size: 5.5, opacity: 0.55 },
      hovertemplate: "objID %{customdata}<br>logM=%{x:.2f}<br>R50=%{y:.2f} kpc<extra>Late</extra>",
    },
  ];
  if (fits.early) {
    msTraces.push({
      name: `Early fit (α=${fits.early.alpha_slope.toFixed(2)})`,
      type: "scatter",
      mode: "lines",
      x: xFit,
      y: xFit.map((x) => Math.pow(10, fits.early.alpha_slope * (x - 10.5) + fits.early.beta_intercept_at_10p5)),
      line: { color: "#fca5a5", width: 2.5 },
      hoverinfo: "skip",
    });
  }
  if (fits.late) {
    msTraces.push({
      name: `Late fit (α=${fits.late.alpha_slope.toFixed(2)})`,
      type: "scatter",
      mode: "lines",
      x: xFit,
      y: xFit.map((x) => Math.pow(10, fits.late.alpha_slope * (x - 10.5) + fits.late.beta_intercept_at_10p5)),
      line: { color: "#7dd3fc", width: 2.5 },
      hoverinfo: "skip",
    });
  }
  Plotly.newPlot(
    "plot-mass-size",
    msTraces,
    {
      ...PLOT_LAYOUT_BASE,
      xaxis: { title: "log₁₀(M★ / M☉)", gridcolor: "rgba(148,163,184,0.12)" },
      yaxis: { title: "Effective Radius R₅₀ (kpc)", type: "log", gridcolor: "rgba(148,163,184,0.12)" },
    },
    PLOT_CONFIG
  );
  bindScatterClick("plot-mass-size");

  // 3. Star-Formation Main Sequence
  Plotly.newPlot(
    "plot-sfms",
    [
      {
        name: "Early-type",
        type: "scattergl",
        mode: "markers",
        x: early.map((g) => g.lm),
        y: early.map((g) => g.lsfr),
        customdata: early.map((g) => g.id),
        marker: { color: "#ef4444", size: 5.5, opacity: 0.55 },
        hovertemplate: "objID %{customdata}<br>logM=%{x:.2f}<br>logSFR=%{y:.2f}<extra>Early</extra>",
      },
      {
        name: "Late-type",
        type: "scattergl",
        mode: "markers",
        x: late.map((g) => g.lm),
        y: late.map((g) => g.lsfr),
        customdata: late.map((g) => g.id),
        marker: { color: "#38bdf8", size: 5.5, opacity: 0.55 },
        hovertemplate: "objID %{customdata}<br>logM=%{x:.2f}<br>logSFR=%{y:.2f}<extra>Late</extra>",
      },
    ],
    {
      ...PLOT_LAYOUT_BASE,
      xaxis: { title: "log₁₀(M★ / M☉)", gridcolor: "rgba(148,163,184,0.12)" },
      yaxis: { title: "log₁₀(SFR / M☉ yr⁻¹)", gridcolor: "rgba(148,163,184,0.12)" },
    },
    PLOT_CONFIG
  );
  bindScatterClick("plot-sfms");

  // 4. SFR Distribution by Morphology
  Plotly.newPlot(
    "plot-sfr-dist",
    [
      {
        name: "Early-type",
        type: "box",
        y: early.map((g) => g.lsfr),
        marker: { color: "#ef4444" },
        boxpoints: "outliers",
      },
      {
        name: "Late-type",
        type: "box",
        y: late.map((g) => g.lsfr),
        marker: { color: "#38bdf8" },
        boxpoints: "outliers",
      },
    ],
    {
      ...PLOT_LAYOUT_BASE,
      yaxis: { title: "log₁₀(SFR / M☉ yr⁻¹)", gridcolor: "rgba(148,163,184,0.12)" },
    },
    PLOT_CONFIG
  );

  // 5. Morphology & Red/Blue Breakdown
  const eRed = early.filter((g) => g.cp === "red_sequence").length;
  const eBlue = early.length - eRed;
  const lRed = late.filter((g) => g.cp === "red_sequence").length;
  const lBlue = late.length - lRed;
  Plotly.newPlot(
    "plot-morph-breakdown",
    [
      {
        name: "Red Sequence",
        type: "bar",
        x: ["Early-type", "Late-type"],
        y: [eRed, lRed],
        marker: { color: "#dc2626" },
      },
      {
        name: "Blue Cloud",
        type: "bar",
        x: ["Early-type", "Late-type"],
        y: [eBlue, lBlue],
        marker: { color: "#0ea5e9" },
      },
    ],
    {
      ...PLOT_LAYOUT_BASE,
      barmode: "group",
      yaxis: { title: "Galaxy Count", gridcolor: "rgba(148,163,184,0.12)" },
    },
    PLOT_CONFIG
  );

  // 6. Pearson Correlation Heatmap
  const corr = state.data.summary.correlation_matrix;
  Plotly.newPlot(
    "plot-corr",
    [
      {
        type: "heatmap",
        z: corr.values,
        x: corr.labels,
        y: corr.labels,
        zmin: -1,
        zmax: 1,
        colorscale: "RdBu",
        reversescale: true,
        text: corr.values.map((r) => r.map((v) => v.toFixed(2))),
        texttemplate: "%{text}",
        textfont: { size: 10 },
      },
    ],
    {
      ...PLOT_LAYOUT_BASE,
      margin: { l: 86, r: 16, t: 16, b: 70 },
      xaxis: { tickangle: -35 },
      yaxis: { autorange: "reversed" },
    },
    PLOT_CONFIG
  );
}

function selectGalaxy(g) {
  state.selectedGalaxy = g;
  state.inspectorMode = "rgb";

  document.getElementById("insp-id").textContent = `objID ${g.id}`;
  const badge = document.getElementById("insp-morph-badge");
  badge.textContent = g.mp;
  badge.className = `badge ${g.mp === "early" ? "badge-early" : "badge-late"}`;

  updateInspectorImage();

  document.getElementById("insp-params").innerHTML = `
    <div class="param-item"><span>Redshift z</span><strong>${fmt(g.z, 4)}</strong></div>
    <div class="param-item"><span>Abs Mag M_r</span><strong>${fmt(g.mr, 2)}</strong></div>
    <div class="param-item"><span>Color (g − r)</span><strong>${fmt(g.gr, 3)} mag</strong></div>
    <div class="param-item"><span>log₁₀(M★/M☉)</span><strong>${fmt(g.lm, 2)}</strong></div>
    <div class="param-item"><span>log₁₀(SFR)</span><strong>${fmt(g.lsfr, 2)}</strong></div>
    <div class="param-item"><span>Radius R₅₀</span><strong>${fmt(g.re, 2)} kpc</strong></div>
    <div class="param-item"><span>Concentration C</span><strong>${fmt(g.c, 2)}</strong></div>
    <div class="param-item"><span>fracDeV_r</span><strong>${fmt(g.fdev, 2)}</strong></div>
  `;

  const zooEl = document.getElementById("insp-zoo");
  if (g.pel !== null && g.pcs !== null) {
    zooEl.innerHTML = `
      <div class="bar-row">
        <div class="bar-label"><span>Elliptical Vote (p_el_debiased)</span><strong class="mono">${pct(g.pel)}</strong></div>
        <div class="bar-track"><div class="bar-fill" style="width:${(g.pel * 100).toFixed(1)}%;background:#ef4444;"></div></div>
      </div>
      <div class="bar-row">
        <div class="bar-label"><span>Spiral Vote (p_cs_debiased)</span><strong class="mono">${pct(g.pcs)}</strong></div>
        <div class="bar-track"><div class="bar-fill" style="width:${(g.pcs * 100).toFixed(1)}%;background:#38bdf8;"></div></div>
      </div>
    `;
  } else {
    zooEl.innerHTML = `<span class="hint">No Galaxy Zoo human vote record for this object.</span>`;
  }

  const predsEl = document.getElementById("insp-predictions");
  const modelMeta = [
    ["tf", "Custom TensorFlow CNN (Hubble)"],
    ["astronn", "astroNN Galaxy10CNN (Bayesian)"],
    ["rf", "Random Forest (Photometric)"],
  ];
  const cards = modelMeta
    .filter(([k]) => g.pred && g.pred[k])
    .map(([k, label]) => {
      const p = g.pred[k];
      const extra = p.probs && p.probs._mc_uncertainty_std !== undefined
        ? `<div class="hint">MC Dropout σ = ${p.probs._mc_uncertainty_std} · Top G10: ${p.probs._galaxy10_top_class || ""}</div>`
        : "";
      return `
        <div class="pred-card">
          <div class="pred-card-top">
            <span class="pred-model-tag">${label}</span>
            <span class="mono">${pct(p.conf)}</span>
          </div>
          <div class="pred-class-val">${p.cls}</div>
          <div class="bar-track" style="margin-top:5px;"><div class="bar-fill" style="width:${((p.conf || 0) * 100).toFixed(1)}%;"></div></div>
          ${extra}
        </div>
      `;
    });
  predsEl.innerHTML = cards.length ? cards.join("") : `<span class="hint">No predictions stored for this galaxy.</span>`;
}

function updateInspectorImage() {
  const g = state.selectedGalaxy;
  if (!g) return;
  const imgEl = document.getElementById("insp-img");
  const noImgEl = document.getElementById("insp-no-img");
  const toggleRow = document.getElementById("insp-lupton-toggle-row");
  const btnRgb = document.getElementById("btn-show-rgb");
  const btnLup = document.getElementById("btn-show-lupton");

  toggleRow.style.display = g.lup ? "flex" : "none";
  btnRgb.classList.toggle("active", state.inspectorMode === "rgb");
  btnLup.classList.toggle("active", state.inspectorMode === "lupton");

  const src = state.inspectorMode === "lupton" && g.lup ? g.lup : g.img;
  if (src) {
    imgEl.src = src;
    imgEl.style.display = "block";
    noImgEl.style.display = "none";
  } else {
    imgEl.style.display = "none";
    noImgEl.style.display = "block";
  }
}

function renderFitsShowcase() {
  const container = document.getElementById("fits-showcase-grid");
  const items = state.data.showcase || [];
  container.innerHTML = items
    .map((g) => {
      const tfCls = g.pred?.tf?.cls || g.mp;
      return `
        <div class="showcase-card" data-id="${g.id}">
          <div class="showcase-pair">
            <figure>
              <img src="${g.img || g.lup}" alt="SDSS RGB ${g.id}" loading="lazy" />
              <figcaption>SDSS RGB Cutout</figcaption>
            </figure>
            <figure>
              <img src="${g.lup}" alt="Lupton RGB ${g.id}" loading="lazy" />
              <figcaption>FITS Cutout2D + Lupton RGB</figcaption>
            </figure>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
            <strong class="mono" style="font-size:0.78rem;">${g.id}</strong>
            <span class="badge ${g.mp === "early" ? "badge-early" : "badge-late"}">${g.mp}</span>
          </div>
          <div class="hint" style="margin-top:4px;">
            z=${fmt(g.z, 3)} · C=${fmt(g.c, 2)} · logM=${fmt(g.lm, 2)} · TF CNN: <strong>${tfCls}</strong>
          </div>
        </div>
      `;
    })
    .join("");

  container.querySelectorAll(".showcase-card").forEach((el) => {
    el.addEventListener("click", () => {
      const g = state.data.galaxies.find((x) => x.id === el.dataset.id);
      if (g) {
        selectGalaxy(g);
        switchTab("tab-explorer");
      }
    });
  });
}

function renderGallery() {
  const imaged = state.filtered.filter((g) => g.img);
  const totalPages = Math.max(1, Math.ceil(imaged.length / state.galleryPageSize));
  state.galleryPage = Math.min(state.galleryPage, totalPages - 1);
  const start = state.galleryPage * state.galleryPageSize;
  const pageItems = imaged.slice(start, start + state.galleryPageSize);

  document.getElementById("gallery-page-info").textContent =
    `${imaged.length} imaged · Page ${state.galleryPage + 1}/${totalPages}`;

  const grid = document.getElementById("rgb-gallery-grid");
  if (!pageItems.length) {
    grid.innerHTML = `<p class="hint">No downloaded RGB cutouts match the active filters.</p>`;
    return;
  }

  grid.innerHTML = pageItems
    .map((g) => {
      const label = g.pred?.tf?.cls || g.mp;
      return `
        <div class="cutout-tile" data-id="${g.id}">
          <img src="${g.img}" alt="Galaxy ${g.id}" loading="lazy" />
          <div class="cutout-meta">
            <div class="tile-cls" title="${label}">${label}</div>
            <div class="tile-sub">z=${fmt(g.z, 3)} · C=${fmt(g.c, 2)}</div>
          </div>
        </div>
      `;
    })
    .join("");

  grid.querySelectorAll(".cutout-tile").forEach((el) => {
    el.addEventListener("click", () => {
      const g = state.data.galaxies.find((x) => x.id === el.dataset.id);
      if (g) {
        selectGalaxy(g);
        switchTab("tab-explorer");
      }
    });
  });
}

function renderModelsTab() {
  const models = state.data.models || {};
  const cardsEl = document.getElementById("model-summary-cards");

  const entries = [
    {
      key: "rf",
      folder: "classifiers/astronn_and_astroml/tabular_baseline.py",
      title: "Tabular Photometric Random Forest",
      desc: "Classical astroML/scikit-learn baseline trained on 8 physical & photometric features (excluding fracDeV_r) against Galaxy Zoo consensus.",
      extra: models.rf ? `5-Fold CV: ${pct(models.rf.cv5_accuracy_mean)} · ROC-AUC: ${fmt(models.rf.roc_auc, 3)}` : "",
    },
    {
      key: "astronn",
      folder: "classifiers/astronn_and_astroml/astronn_classifier.py",
      title: "astroNN Galaxy10CNN (Bayesian MC Dropout)",
      desc: "4-layer Galaxy10CNN architecture using astroNN.datasets.galaxy10sdss with Monte Carlo Dropout uncertainty estimation.",
      extra: models.astronn
        ? `10-Class Acc: ${pct(models.astronn.test_accuracy_10class)} · Mean MC σ: ${fmt(models.astronn.mean_mc_epistemic_std, 4)}`
        : "",
    },
    {
      key: "tf",
      folder: "classifiers/tensorflow_cnn/cnn_model.py",
      title: "Custom TensorFlow / Keras Hubble CNN",
      desc: "4-block Conv2D + BatchNorm + MaxPool CNN built from scratch with 360° rotational & reflection invariance augmentation.",
      extra: models.tf ? `Parameters: ${(models.tf.n_parameters || 0).toLocaleString()} · Epochs: ${models.tf.epochs_run}` : "",
    },
  ];

  cardsEl.innerHTML = entries
    .map((e) => {
      const m = models[e.key];
      return `
        <article class="card">
          <span class="eyebrow">${e.folder}</span>
          <h3>${e.title}</h3>
          <p class="card-sub">${e.desc}</p>
          <div class="model-metric-row">
            <div class="model-metric-pill"><span>Test Accuracy</span><strong>${m ? pct(m.test_accuracy) : "—"}</strong></div>
            <div class="model-metric-pill"><span>Macro F1</span><strong>${m ? pct(m.macro_f1) : "—"}</strong></div>
            <div class="model-metric-pill"><span>Catalog Scored</span><strong>${m ? (m.n_catalog_predicted || 0).toLocaleString() : "—"}</strong></div>
          </div>
          <p class="hint" style="margin-top:10px;">${e.extra}</p>
        </article>
      `;
    })
    .join("");

  // Bar comparison chart
  const compNames = ["Random Forest\n(2-Class Zoo)", "astroNN CNN\n(5-Class Hubble)", "Custom TF CNN\n(5-Class Hubble)"];
  const compKeys = ["rf", "astronn", "tf"];
  Plotly.newPlot(
    "plot-model-compare",
    [
      {
        name: "Test Accuracy",
        type: "bar",
        x: compNames,
        y: compKeys.map((k) => (models[k] ? models[k].test_accuracy * 100 : 0)),
        marker: { color: "#38bdf8" },
      },
      {
        name: "Macro F1",
        type: "bar",
        x: compNames,
        y: compKeys.map((k) => (models[k] ? models[k].macro_f1 * 100 : 0)),
        marker: { color: "#a78bfa" },
      },
    ],
    {
      ...PLOT_LAYOUT_BASE,
      barmode: "group",
      yaxis: { title: "Score (%)", range: [0, 100], gridcolor: "rgba(148,163,184,0.12)" },
    },
    PLOT_CONFIG
  );

  // RF Feature Importance
  if (models.rf && models.rf.feature_importances) {
    const pairs = Object.entries(models.rf.feature_importances).reverse();
    Plotly.newPlot(
      "plot-rf-importance",
      [
        {
          type: "bar",
          orientation: "h",
          y: pairs.map(([k]) => k),
          x: pairs.map(([, v]) => v),
          marker: { color: "#34d399" },
        },
      ],
      {
        ...PLOT_LAYOUT_BASE,
        margin: { l: 120, r: 22, t: 18, b: 48 },
        xaxis: { title: "Gini Importance", gridcolor: "rgba(148,163,184,0.12)" },
      },
      PLOT_CONFIG
    );
  }

  renderCnnCurves();
  renderConfusionMatrix();
  renderHubbleDistributionComparison();
}

function renderCnnCurves() {
  const m = state.data.models?.[state.activeCurveModel];
  if (!m || !m.history) return;
  const epochs = m.history.accuracy.map((_, i) => i + 1);
  Plotly.newPlot(
    "plot-cnn-curves",
    [
      { name: "Train Accuracy", x: epochs, y: m.history.accuracy, mode: "lines+markers", line: { color: "#38bdf8" } },
      { name: "Val Accuracy", x: epochs, y: m.history.val_accuracy, mode: "lines+markers", line: { color: "#34d399" } },
      { name: "Val Loss", x: epochs, y: m.history.val_loss, yaxis: "y2", mode: "lines+markers", line: { color: "#f97316", dash: "dot" } },
    ],
    {
      ...PLOT_LAYOUT_BASE,
      xaxis: { title: "Epoch", gridcolor: "rgba(148,163,184,0.12)" },
      yaxis: { title: "Accuracy", range: [0, 1], gridcolor: "rgba(148,163,184,0.12)" },
      yaxis2: { title: "Val Loss", overlaying: "y", side: "right", showgrid: false },
    },
    PLOT_CONFIG
  );
}

function renderConfusionMatrix() {
  const m = state.data.models?.[state.activeCmModel];
  if (!m || !m.confusion_matrix) return;
  const labels = m.hubble_classes || m.classes || [];
  const short = labels.map((s) => s.replace("Loose Spiral / Irregular (Sc/Irr)", "Loose Sc/Irr").replace("Lenticular/Edge-on", "S0/Edge-on"));
  Plotly.newPlot(
    "plot-confusion-matrix",
    [
      {
        type: "heatmap",
        z: m.confusion_matrix,
        x: short,
        y: short,
        colorscale: "Blues",
        text: m.confusion_matrix.map((r) => r.map(String)),
        texttemplate: "%{text}",
      },
    ],
    {
      ...PLOT_LAYOUT_BASE,
      margin: { l: 110, r: 22, t: 18, b: 75 },
      xaxis: { title: "Predicted", tickangle: -20 },
      yaxis: { title: "True", autorange: "reversed" },
    },
    PLOT_CONFIG
  );
}

function renderHubbleDistributionComparison() {
  const classes = state.data.models?.tf?.hubble_classes || [
    "Elliptical",
    "Lenticular/Edge-on",
    "Tight Spiral (Sa)",
    "Medium Spiral (Sb)",
    "Loose Spiral / Irregular (Sc/Irr)",
  ];
  const tfCounts = classes.map((c) => state.data.galaxies.filter((g) => g.pred?.tf?.cls === c).length);
  const astroCounts = classes.map((c) => state.data.galaxies.filter((g) => g.pred?.astronn?.cls === c).length);

  Plotly.newPlot(
    "plot-hubble-dist",
    [
      { name: "Custom TensorFlow CNN", type: "bar", x: classes, y: tfCounts, marker: { color: "#38bdf8" } },
      { name: "astroNN Galaxy10CNN", type: "bar", x: classes, y: astroCounts, marker: { color: "#a78bfa" } },
    ],
    {
      ...PLOT_LAYOUT_BASE,
      barmode: "group",
      yaxis: { title: "Predicted Galaxies (Imaged Catalog)", gridcolor: "rgba(148,163,184,0.12)" },
    },
    PLOT_CONFIG
  );
}

function renderCatalogTable() {
  const q = (document.getElementById("catalog-search").value || "").trim();
  const rows = state.filtered
    .filter((g) => (!q ? true : g.id.includes(q)))
    .slice(0, 120);

  const tbody = document.getElementById("catalog-tbody");
  tbody.innerHTML = rows
    .map(
      (g) => `
      <tr data-id="${g.id}">
        <td class="mono">${g.id}</td>
        <td class="mono">${fmt(g.z, 4)}</td>
        <td class="mono">${fmt(g.mr, 2)}</td>
        <td class="mono">${fmt(g.gr, 3)}</td>
        <td class="mono">${fmt(g.lm, 2)}</td>
        <td class="mono">${fmt(g.lsfr, 2)}</td>
        <td class="mono">${fmt(g.re, 2)}</td>
        <td class="mono">${fmt(g.c, 2)}</td>
        <td><span class="badge ${g.mp === "early" ? "badge-early" : "badge-late"}">${g.mp}</span></td>
        <td>${g.pred?.rf?.cls || "—"}</td>
        <td>${g.pred?.tf?.cls || "—"}</td>
        <td>${g.img ? "RGB" + (g.lup ? " + FITS" : "") : "—"}</td>
      </tr>
    `
    )
    .join("");

  tbody.querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => {
      const g = state.data.galaxies.find((x) => x.id === tr.dataset.id);
      if (g) {
        selectGalaxy(g);
        switchTab("tab-explorer");
      }
    });
  });
}

function switchTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === tabId));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === tabId));
  // Resize Plotly graphs after tab becomes visible
  setTimeout(() => {
    document.querySelectorAll(`#${tabId} .plot-box`).forEach((el) => {
      if (el.data) Plotly.Plots.resize(el);
    });
  }, 40);
}

function wireEvents() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  ["filter-morph", "filter-color-pop", "filter-hubble", "filter-zmax", "filter-mmin", "filter-has-image"].forEach(
    (id) => {
      document.getElementById(id).addEventListener("input", applyFilters);
      document.getElementById(id).addEventListener("change", applyFilters);
    }
  );

  document.getElementById("btn-reset-filters").addEventListener("click", () => {
    document.getElementById("filter-morph").value = "all";
    document.getElementById("filter-color-pop").value = "all";
    document.getElementById("filter-hubble").value = "all";
    document.getElementById("filter-zmax").value = "0.10";
    document.getElementById("filter-mmin").value = "8.5";
    document.getElementById("filter-has-image").checked = false;
    applyFilters();
  });

  document.getElementById("btn-show-rgb").addEventListener("click", () => {
    state.inspectorMode = "rgb";
    updateInspectorImage();
  });

  document.getElementById("btn-show-lupton").addEventListener("click", () => {
    state.inspectorMode = "lupton";
    updateInspectorImage();
  });

  document.getElementById("btn-random-imaged").addEventListener("click", () => {
    const pool = state.filtered.filter((g) => g.img);
    const list = pool.length ? pool : state.data.galaxies.filter((g) => g.img);
    if (list.length) {
      selectGalaxy(list[Math.floor(Math.random() * list.length)]);
    }
  });

  document.getElementById("btn-prev-page").addEventListener("click", () => {
    if (state.galleryPage > 0) {
      state.galleryPage -= 1;
      renderGallery();
    }
  });

  document.getElementById("btn-next-page").addEventListener("click", () => {
    const imagedCount = state.filtered.filter((g) => g.img).length;
    const maxPage = Math.max(0, Math.ceil(imagedCount / state.galleryPageSize) - 1);
    if (state.galleryPage < maxPage) {
      state.galleryPage += 1;
      renderGallery();
    }
  });

  document.querySelectorAll("[data-curve]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.activeCurveModel = btn.dataset.curve;
      document.querySelectorAll("[data-curve]").forEach((b) => b.classList.toggle("active", b === btn));
      renderCnnCurves();
    });
  });

  document.querySelectorAll("[data-cm]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.activeCmModel = btn.dataset.cm;
      document.querySelectorAll("[data-cm]").forEach((b) => b.classList.toggle("active", b === btn));
      renderConfusionMatrix();
    });
  });

  document.getElementById("catalog-search").addEventListener("input", renderCatalogTable);
}

async function init() {
  try {
    state.data = await loadAtlasData();
    wireEvents();
    renderHeroAndKPIs(state.data);
    renderFitsShowcase();
    renderModelsTab();
    applyFilters();

    const defaultGalaxy =
      state.data.showcase?.[0] ||
      state.data.galaxies.find((g) => g.img) ||
      state.data.galaxies[0];
    if (defaultGalaxy) selectGalaxy(defaultGalaxy);
  } catch (err) {
    console.error(err);
    document.body.innerHTML = `<div style="padding:40px;color:#fca5a5;">Failed to load Atlas data: ${err.message}</div>`;
  }
}

document.addEventListener("DOMContentLoaded", init);
