const $ = (s) => document.querySelector(s);
const FIELDS = ["sepal_length", "sepal_width", "petal_length", "petal_width"];
const SAMPLES = { setosa: [5.1, 3.5, 1.4, 0.2], versicolor: [6.0, 2.9, 4.5, 1.5], virginica: [6.5, 3.0, 5.5, 1.8] };
const ICONS = { "Iris Setosa": "🌸", "Iris Versicolor": "🌺", "Iris Virginica": "🌷" };
const form = $("#form");

async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch (_) { /* non-JSON response */ }
  return { ok: res.ok, status: res.status, body };
}

/* ---------- navigation ---------- */
$("#burger").addEventListener("click", () => {
  const open = $("#nav").classList.toggle("open");
  $("#burger").setAttribute("aria-expanded", open);
});
document.querySelectorAll("#nav a").forEach((a) => a.addEventListener("click", () => $("#nav").classList.remove("open")));

/* ---------- live status (real /health and /api/info responses) ---------- */
function setStatus(key, text, state) {
  const el = document.querySelector(`[data-s="${key}"]`);
  el.textContent = text;
  el.className = state;
}

async function refreshStatus() {
  const pill = $("#heroStatus");
  try {
    const [h, i] = await Promise.all([api("/health"), api("/api/info")]);
    const loaded = !!(h.body && h.body.model_loaded);
    setStatus("app", "Online", "ok");
    setStatus("model", loaded ? "Loaded" : "Not loaded", loaded ? "ok" : "bad");
    const inDocker = !!(h.body && h.body.in_container);
    setStatus("container", inDocker ? "Running (Docker)" : "Not in Docker (local run)", inDocker ? "ok" : "warn");
    const ec2 = !!(i.body && i.body.running_on_ec2);
    setStatus("cloud", ec2 ? "AWS EC2" : "EC2 not detected", ec2 ? "ok" : "warn");
    setStatus("api", i.ok ? "Available" : "Unavailable", i.ok ? "ok" : "bad");
    pill.className = "pill " + (loaded ? "ok" : "bad");
    pill.lastElementChild.textContent = loaded ? "ML Model Online" : "ML Model Offline";
    if (i.body && i.body.accuracy !== undefined) {
      $("#acc").textContent = (i.body.accuracy * 100).toFixed(1) + "%";
      $("#split").textContent = `Trained on ${i.body.train_samples} samples, evaluated on ${i.body.test_samples} unseen samples (scikit-learn ${i.body.sklearn_version}).`;
    } else {
      $("#acc").textContent = "Unavailable";
      $("#split").textContent = "The model is not loaded, so no accuracy can be reported.";
    }
  } catch (_) {
    ["app", "model", "container", "cloud", "api"].forEach((k) => setStatus(k, k === "app" ? "Offline" : "Unknown", "bad"));
    pill.className = "pill bad";
    pill.lastElementChild.textContent = "ML Model Offline";
    $("#acc").textContent = "Unavailable";
  }
}
$("#refresh").addEventListener("click", refreshStatus);

/* ---------- prediction ---------- */
function showError(msg) { const e = $("#error"); e.textContent = msg; e.hidden = !msg; }

document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () => {
  SAMPLES[b.dataset.sample].forEach((v, k) => (form.elements[FIELDS[k]].value = v.toFixed(1)));
  showError("");
}));

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const data = {};
  for (const f of FIELDS) {
    const raw = form.elements[f].value.trim();
    const v = Number(raw);
    if (raw === "" || !Number.isFinite(v)) return showError("Please enter valid numerical values for all four features.");
    if (v < 0) return showError("Negative values are not allowed.");
    if (v === 0 || v > 20) return showError("Each value must be greater than 0 and at most 20 cm.");
    data[f] = v;
  }
  showError("");
  const btn = $("#submit");
  btn.disabled = true; btn.textContent = "Predicting…";
  try {
    const r = await api("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
    if (!r.ok || !r.body || !r.body.success) return showError((r.body && r.body.error) || `Server error (${r.status}). Please try again.`);
    renderResult(r.body);
  } catch (_) {
    showError("Cannot reach the API. Check that the server or Docker container is running.");
  } finally {
    btn.disabled = false; btn.textContent = "Predict Species";
  }
});

function renderResult(d) {
  $("#rIcon").textContent = ICONS[d.prediction] || "🌼";
  $("#rName").textContent = d.prediction;
  $("#rConf").textContent = (d.confidence * 100).toFixed(1) + "%";
  const bars = $("#rBars");
  bars.replaceChildren();
  Object.entries(d.probabilities || {}).forEach(([name, p]) => {
    const row = document.createElement("div"); row.className = "bar-row";
    const label = document.createElement("span"); label.textContent = name.replace("Iris ", "");
    const track = document.createElement("div"); track.className = "track";
    const fill = document.createElement("div"); fill.className = "fill"; fill.style.width = (p * 100).toFixed(1) + "%";
    track.appendChild(fill);
    const pct = document.createElement("span"); pct.textContent = (p * 100).toFixed(1) + "%";
    row.append(label, track, pct); bars.appendChild(row);
  });
  form.hidden = true; $("#result").hidden = false;
  $(".samples").hidden = true;
}
$("#again").addEventListener("click", () => {
  form.reset(); form.hidden = false; $("#result").hidden = true; $(".samples").hidden = false; showError("");
});

/* ---------- Try API buttons (real calls) ---------- */
document.querySelectorAll("[data-try]").forEach((b) => b.addEventListener("click", async () => {
  const [method, path] = b.dataset.try.split(" ");
  const out = document.getElementById(b.dataset.out);
  const opts = { method };
  if (method === "POST") {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = $("#reqBody").value;
  }
  out.textContent = "Loading…";
  try {
    const r = await api(path, opts);
    out.textContent = `HTTP ${r.status}\n` + JSON.stringify(r.body, null, 2);
  } catch (_) {
    out.textContent = "Cannot reach the API. Is the server running?";
  }
}));

refreshStatus();
