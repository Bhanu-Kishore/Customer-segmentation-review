"use strict";

/* ------------------------------------------------------------------ setup */
const SEGMENT_COLORS = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#0ea5e9"];
const SEGMENT_SOFT = ["#eff6ff", "#ecfdf5", "#fffbeb", "#f5f3ff", "#fef2f2", "#f0f9ff"];

const currency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});
const compactCurrency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  notation: "compact",
  maximumFractionDigits: 1,
});
const decimal = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

const state = {
  view: "dashboard",
  summary: null,
  segments: [],
  customers: { items: [], total: 0, page: 1, pages: 1 },
  filters: { search: "", segment: "", page: 1, pageSize: 10 },
  charts: {},
  editingId: null,
  pendingDelete: null,
  lastPrediction: null,
};

const $ = (selector, root = document) => root.querySelector(selector);

const VIEW_META = {
  dashboard: { title: "Dashboard", subtitle: "Live overview of your customer segments.", load: loadDashboard },
  customers: { title: "Customers", subtitle: "Manage customer records and re-score segments.", load: loadCustomers },
  segmentation: { title: "Segmentation", subtitle: "Score new customers with the trained K-Means model.", load: loadSegmentation },
  analytics: { title: "Analytics", subtitle: "Distribution and behaviour insights across segments.", load: loadAnalytics },
  model: { title: "Model & Drift", subtitle: "Production model metadata and Evidently monitoring.", load: loadModel },
};

/* ---------------------------------------------------------------- helpers */
function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function segmentColor(segment) {
  return SEGMENT_COLORS[Math.abs(Number(segment)) % SEGMENT_COLORS.length];
}

function segmentSoft(segment) {
  return SEGMENT_SOFT[Math.abs(Number(segment)) % SEGMENT_SOFT.length];
}

function segmentBadge(segment, name) {
  return `<span class="segment-badge" style="--seg-color:${segmentColor(segment)};--seg-bg:${segmentSoft(segment)}">${escapeHtml(name)}</span>`;
}

function formatDate(value) {
  if (!value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatCompactNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return decimal.format(Number(value));
}

async function api(path, options = {}) {
  const config = {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  };

  let response;
  try {
    response = await fetch(path, config);
  } catch {
    throw new Error("Cannot reach the API. Check that the server is running.");
  }

  if (response.status === 204) return null;

  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }

  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    if (payload && typeof payload.detail === "string") message = payload.detail;
    else if (payload && Array.isArray(payload.detail) && payload.detail.length) {
      message = payload.detail.map((entry) => entry.msg || "Invalid input").join("; ");
    }
    throw new Error(message);
  }

  return payload;
}

function toast(message, type = "info", timeout = 4200) {
  const container = $("#toast-container");
  const element = document.createElement("div");
  element.className = `toast ${type}`;
  element.textContent = message;
  container.appendChild(element);
  window.setTimeout(() => {
    element.classList.add("is-leaving");
    window.setTimeout(() => element.remove(), 240);
  }, timeout);
}

function setStatus(online, text) {
  const pill = $("#status-pill");
  pill.classList.toggle("is-offline", !online);
  $("#status-text").textContent = text;
}

function emptyRow(colspan, message) {
  return `<tr><td colspan="${colspan}" style="text-align:center;color:var(--muted);padding:28px">${escapeHtml(message)}</td></tr>`;
}

/* ----------------------------------------------------------------- charts */
function makeChart(key, canvasId, config) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || typeof Chart === "undefined") return;
  if (state.charts[key]) state.charts[key].destroy();
  state.charts[key] = new Chart(canvas.getContext("2d"), config);
}

function baseChartOptions(extra = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 450 },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#0f172a",
        padding: 10,
        cornerRadius: 8,
        titleFont: { weight: "600" },
        ...(extra.tooltip || {}),
      },
    },
    ...extra.options,
  };
}

function applyChartDefaults() {
  if (typeof Chart === "undefined") return;
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.font.size = 11;
  Chart.defaults.color = "#64748b";
  Chart.defaults.borderColor = "#e2e8f0";
}

/* --------------------------------------------------------------- dashboard */
async function loadDashboard() {
  const [summary, analytics] = await Promise.all([api("/api/summary"), api("/api/analytics")]);
  state.summary = summary;
  state.segments = summary.segments || [];

  renderKpis(summary);
  renderSegmentCards($("#segment-cards"), state.segments);
  renderRecentCustomers(summary.recent_customers || []);
  populateSegmentFilter(state.segments);
  renderSegmentDoughnut(analytics);
  renderScatter(analytics.scatter || []);
}

function renderKpis(summary) {
  const model = summary.model || {};
  const silhouette = model.silhouette_score;
  const cards = [
    {
      label: "Total customers",
      value: decimal.format(summary.total_customers || 0),
      hint: `${summary.segment_count || 0} active segments`,
      tone: "",
      icon: '<path d="M16 11c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3Zm-8 0c1.66 0 3-1.34 3-3S9.66 5 8 5 5 6.34 5 8s1.34 3 3 3Zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5Zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5Z"/>',
    },
    {
      label: "Average income",
      value: compactCurrency.format(summary.avg_income || 0),
      hint: "Annual, per customer",
      tone: "green",
      icon: '<path d="M21 18v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v1h-9a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h9Zm-9-2h10V8H12v8Zm4-2.5a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3Z"/>',
    },
    {
      label: "Average spending",
      value: `${decimal.format(summary.avg_spending_score || 0)}/100`,
      hint: `${decimal.format(summary.avg_purchase_frequency || 0)} purchases / year`,
      tone: "amber",
      icon: '<path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2Zm1 15h-2v-2h2Zm0-4h-2V7h2Z"/>',
    },
    {
      label: "Model quality",
      value: silhouette === null || silhouette === undefined ? "n/a" : Number(silhouette).toFixed(3),
      hint: `Silhouette · inertia ${formatCompactNumber(model.inertia)}`,
      tone: "violet",
      icon: '<path d="M12 2 2 7l10 5 10-5-10-5Zm0 7.24L5.53 6 12 2.76 18.47 6 12 9.24ZM2 17l10 5 10-5-2.09-1.05L12 19.67l-7.91-3.72L2 17Zm0-5 10 5 10-5-2.09-1.05L12 14.67l-7.91-3.72L2 12Z"/>',
    },
  ];

  $("#kpi-grid").innerHTML = cards
    .map(
      (card) => `
      <article class="kpi-card">
        <div class="kpi-icon ${card.tone}"><svg viewBox="0 0 24 24" aria-hidden="true">${card.icon}</svg></div>
        <div class="kpi-body">
          <span class="kpi-label">${card.label}</span>
          <strong class="kpi-value">${escapeHtml(card.value)}</strong>
          <span class="kpi-hint">${escapeHtml(card.hint)}</span>
        </div>
      </article>`
    )
    .join("");
}

function renderSegmentCards(target, segments) {
  if (!target) return;
  if (!segments.length) {
    target.innerHTML = '<div class="empty-state"><strong>No segment data</strong><span>The model did not return any clusters.</span></div>';
    return;
  }

  target.innerHTML = segments
    .map(
      (segment) => `
      <article class="segment-card" style="--seg-color:${segmentColor(segment.segment)}">
        <div class="segment-card-head">
          <h3>${escapeHtml(segment.name)}</h3>
          <span class="segment-count">${decimal.format(segment.size)} customer${segment.size === 1 ? "" : "s"}</span>
        </div>
        <p class="segment-desc">${escapeHtml(segment.description || "Cluster discovered by K-Means.")}</p>
        <div class="segment-stats">
          <div class="segment-stat"><span>Avg income</span><strong>${compactCurrency.format(segment.avg_income || 0)}</strong></div>
          <div class="segment-stat"><span>Avg spending</span><strong>${decimal.format(segment.avg_spending_score || 0)}</strong></div>
          <div class="segment-stat"><span>Avg age</span><strong>${decimal.format(segment.avg_age || 0)} yrs</strong></div>
          <div class="segment-stat"><span>Frequency</span><strong>${decimal.format(segment.avg_purchase_frequency || 0)} / yr</strong></div>
        </div>
      </article>`
    )
    .join("");
}

function renderRecentCustomers(customers) {
  const body = $("#recent-table-body");
  if (!customers.length) {
    body.innerHTML = emptyRow(6, "No customers yet. Add your first customer to see predictions here.");
    return;
  }
  body.innerHTML = customers
    .map(
      (customer) => `
      <tr>
        <td class="cell-muted">#${customer.id}</td>
        <td class="cell-name">${escapeHtml(customer.name)}</td>
        <td>${customer.age}</td>
        <td>${currency.format(customer.annual_income)}</td>
        <td>${decimal.format(customer.spending_score)}</td>
        <td>${segmentBadge(customer.segment, customer.segment_name)}</td>
      </tr>`
    )
    .join("");
}

function renderSegmentDoughnut(analytics) {
  const segments = analytics.segments || [];
  makeChart("segments", "chart-segments", {
    type: "doughnut",
    data: {
      labels: segments.map((segment) => segment.name),
      datasets: [
        {
          data: segments.map((segment) => segment.size),
          backgroundColor: segments.map((segment) => segmentColor(segment.segment)),
          borderColor: "#ffffff",
          borderWidth: 3,
          hoverOffset: 8,
        },
      ],
    },
    options: baseChartOptions({
      options: { cutout: "62%" },
      tooltip: {
        callbacks: {
          label: (context) => ` ${context.label}: ${context.raw} customers`,
        },
      },
    }),
  });

  const chart = state.charts.segments;
  if (chart) {
    chart.options.plugins.legend = {
      display: true,
      position: "bottom",
      labels: { usePointStyle: true, pointStyle: "circle", boxWidth: 8, padding: 16 },
    };
    chart.update();
  }
}

function renderScatter(points) {
  const segments = (state.summary && state.summary.segments) || [];
  const datasets = segments.map((segment) => ({
    label: segment.name,
    data: points
      .filter((point) => point.segment === segment.segment)
      .map((point) => ({
        x: point.income,
        y: point.spending,
        name: point.name,
        id: point.id,
        frequency: point.frequency,
      })),
    backgroundColor: segmentColor(segment.segment),
    pointRadius: 5,
    pointHoverRadius: 7,
    pointBorderColor: "#ffffff",
    pointBorderWidth: 1.5,
  }));

  makeChart("scatter", "chart-scatter", {
    type: "scatter",
    data: { datasets },
    options: baseChartOptions({
      options: {
        scales: {
          x: {
            beginAtZero: true,
            title: { display: true, text: "Annual income (USD)", color: "#64748b" },
            ticks: { callback: (value) => compactCurrency.format(value) },
            grid: { color: "#eef2f7" },
          },
          y: {
            min: 0,
            max: 100,
            title: { display: true, text: "Spending score", color: "#64748b" },
            grid: { color: "#eef2f7" },
          },
        },
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const raw = context.raw;
            return `${raw.name} (#${raw.id}) · ${currency.format(raw.x)} · score ${raw.y}`;
          },
        },
      },
    }),
  });

  const chart = state.charts.scatter;
  if (chart) {
    chart.options.plugins.legend = {
      display: true,
      position: "bottom",
      labels: { usePointStyle: true, pointStyle: "circle", boxWidth: 8, padding: 14 },
    };
    chart.update();
  }
}

/* --------------------------------------------------------------- customers */
async function loadCustomers() {
  const filters = state.filters;
  const params = new URLSearchParams({ page: filters.page, page_size: filters.pageSize });
  if (filters.search) params.set("search", filters.search);
  if (filters.segment !== "") params.set("segment", filters.segment);

  const data = await api(`/api/customers?${params.toString()}`);
  state.customers = data;

  if (data.pages > 0 && filters.page > data.pages) {
    filters.page = data.pages;
    return loadCustomers();
  }

  renderCustomersTable(data);
  updatePagination(data);
}

function renderCustomersTable(data) {
  const body = $("#customers-table-body");
  const empty = $("#customers-empty");

  if (!data.items.length) {
    body.innerHTML = "";
    empty.hidden = false;
    return;
  }
  empty.hidden = true;

  body.innerHTML = data.items
    .map(
      (customer) => `
      <tr>
        <td class="cell-muted">#${customer.id}</td>
        <td class="cell-name">${escapeHtml(customer.name)}</td>
        <td>${customer.age}</td>
        <td>${currency.format(customer.annual_income)}</td>
        <td>${decimal.format(customer.spending_score)}</td>
        <td>${customer.purchase_frequency}</td>
        <td>${segmentBadge(customer.segment, customer.segment_name)}</td>
        <td class="align-right">
          <div class="row-actions">
            <button class="btn btn-ghost btn-sm" type="button" data-edit="${customer.id}">Edit</button>
            <button class="btn btn-ghost btn-sm" type="button" data-delete="${customer.id}" aria-label="Delete ${escapeHtml(customer.name)}">Delete</button>
          </div>
        </td>
      </tr>`
    )
    .join("");
}

function updatePagination(data) {
  const pages = data.pages || 1;
  $("#pagination-info").textContent = `${decimal.format(data.total)} customer${data.total === 1 ? "" : "s"} in view`;
  $("#page-indicator").textContent = `${data.page} / ${pages}`;
  $("#prev-page").disabled = data.page <= 1;
  $("#next-page").disabled = data.page >= pages;
}

function populateSegmentFilter(segments) {
  const select = $("#customer-segment-filter");
  const current = state.filters.segment;
  select.innerHTML =
    '<option value="">All segments</option>' +
    segments
      .map(
        (segment) =>
          `<option value="${segment.segment}">${escapeHtml(segment.name)} (${segment.size})</option>`
      )
      .join("");

  if (current !== "" && !segments.some((segment) => String(segment.segment) === String(current))) {
    state.filters.segment = "";
  } else {
    select.value = state.filters.segment;
  }
}

function openCustomerModal(customer = null) {
  state.editingId = customer ? customer.id : null;
  $("#modal-title").textContent = customer ? `Edit customer #${customer.id}` : "Add customer";
  $("#modal-submit").textContent = customer ? "Update customer" : "Save customer";
  $("#modal-error").hidden = true;

  $("#field-name").value = customer ? customer.name : "";
  $("#field-age").value = customer ? customer.age : 30;
  $("#field-income").value = customer ? customer.annual_income : 45000;
  $("#field-spending").value = customer ? customer.spending_score : 60;
  $("#field-frequency").value = customer ? customer.purchase_frequency : 10;

  $("#customer-modal-backdrop").hidden = false;
  window.setTimeout(() => $("#field-name").focus(), 60);
}

function closeCustomerModal() {
  $("#customer-modal-backdrop").hidden = true;
  state.editingId = null;
  $("#modal-error").hidden = true;
}

async function submitCustomerForm(event) {
  event.preventDefault();
  const error = $("#modal-error");
  error.hidden = true;

  const name = $("#field-name").value.trim();
  const age = Number($("#field-age").value);
  const annualIncome = Number($("#field-income").value);
  const spendingScore = Number($("#field-spending").value);
  const purchaseFrequency = Number($("#field-frequency").value);

  if (!Number.isFinite(age) || age < 18 || age > 100) return showFormError("Age must be between 18 and 100.");
  if (!Number.isFinite(annualIncome) || annualIncome < 1000 || annualIncome > 1000000) return showFormError("Annual income must be between 1,000 and 1,000,000.");
  if (!Number.isFinite(spendingScore) || spendingScore < 0 || spendingScore > 100) return showFormError("Spending score must be between 0 and 100.");
  if (!Number.isFinite(purchaseFrequency) || purchaseFrequency < 0 || purchaseFrequency > 365) return showFormError("Purchase frequency must be between 0 and 365.");

  const payload = {
    name: name || null,
    age: Math.round(age),
    annual_income: annualIncome,
    spending_score: spendingScore,
    purchase_frequency: Math.round(purchaseFrequency),
  };

  const submit = $("#modal-submit");
  submit.disabled = true;
  try {
    if (state.editingId) {
      const updated = await api(`/api/customers/${state.editingId}`, {
        method: "PUT",
        body: JSON.stringify(payload),
      });
      toast(`Customer #${updated.id} updated and re-scored.`, "success");
    } else {
      const created = await api("/api/customers", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      toast(`Customer #${created.id} added to ${created.segment_name}.`, "success");
    }
    closeCustomerModal();
    await refreshDataViews();
  } catch (err) {
    showFormError(err.message);
  } finally {
    submit.disabled = false;
  }

  function showFormError(message) {
    error.textContent = message;
    error.hidden = false;
    return false;
  }
}

function askDelete(customer) {
  state.pendingDelete = customer;
  $("#confirm-message").textContent = `Delete ${customer.name} (#${customer.id})? This action cannot be undone.`;
  $("#confirm-backdrop").hidden = false;
}

async function confirmDelete() {
  const customer = state.pendingDelete;
  if (!customer) return;
  const button = $("#confirm-ok");
  button.disabled = true;
  try {
    await api(`/api/customers/${customer.id}`, { method: "DELETE" });
    toast(`Customer #${customer.id} deleted.`, "success");
    $("#confirm-backdrop").hidden = true;
    state.pendingDelete = null;
    await refreshDataViews();
  } catch (err) {
    toast(err.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function refreshDataViews() {
  await Promise.allSettled([loadDashboard(), loadCustomers(), checkHealth()]);
  if (state.view === "analytics") await loadAnalytics().catch(() => {});
  if (state.view === "segmentation") await loadSegmentation().catch(() => {});
}

function exportCsv() {
  const filters = state.filters;
  const params = new URLSearchParams({ page: 1, page_size: 100 });
  if (filters.search) params.set("search", filters.search);
  if (filters.segment !== "") params.set("segment", filters.segment);

  api(`/api/customers?${params.toString()}`)
    .then((data) => {
      if (!data.items.length) {
        toast("No customers to export.", "warning");
        return;
      }
      const header = ["id", "name", "age", "annual_income", "spending_score", "purchase_frequency", "segment", "segment_name"];
      const lines = [header.join(",")];
      for (const customer of data.items) {
        lines.push(
          [
            customer.id,
            `"${String(customer.name).replace(/"/g, '""')}"`,
            customer.age,
            customer.annual_income,
            customer.spending_score,
            customer.purchase_frequency,
            customer.segment,
            `"${String(customer.segment_name).replace(/"/g, '""')}"`,
          ].join(",")
        );
      }
      const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "customers.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      toast("Customer list exported.", "success");
    })
    .catch((err) => toast(err.message, "error"));
}

/* ------------------------------------------------------------ segmentation */
async function loadSegmentation() {
  const data = await api("/api/segments");
  renderSegmentCards($("#segment-reference"), data.segments || []);
  if (!state.segments.length) {
    state.segments = data.segments || [];
    populateSegmentFilter(state.segments);
  }
}

function readPredictForm() {
  const age = Number($("#pred-age").value);
  const income = Number($("#pred-income").value);
  const spending = Number($("#pred-spending").value);
  const frequency = Number($("#pred-frequency").value);

  if (!Number.isFinite(age) || age < 18 || age > 100) throw new Error("Age must be between 18 and 100.");
  if (!Number.isFinite(income) || income < 1000 || income > 1000000) throw new Error("Income must be between 1,000 and 1,000,000.");
  if (!Number.isFinite(spending) || spending < 0 || spending > 100) throw new Error("Spending score must be between 0 and 100.");
  if (!Number.isFinite(frequency) || frequency < 0 || frequency > 365) throw new Error("Purchase frequency must be between 0 and 365.");

  return {
    age: Math.round(age),
    annual_income: income,
    spending_score: spending,
    purchase_frequency: Math.round(frequency),
  };
}

async function handlePredict(event) {
  event.preventDefault();
  const button = $("#predict-btn");
  let payload;

  try {
    payload = readPredictForm();
  } catch (err) {
    toast(err.message, "error");
    return;
  }

  button.disabled = true;
  button.textContent = "Scoring...";
  try {
    const result = await api("/api/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.lastPrediction = { ...result, input: payload };
    renderPrediction(result);
    $("#save-customer-btn").disabled = false;
  } catch (err) {
    toast(err.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Run segmentation";
  }
}

function renderPrediction(result) {
  $("#result-empty").hidden = true;
  $("#result-content").hidden = false;

  const badge = $("#result-segment-badge");
  badge.textContent = `Segment ${result.segment}`;
  badge.style.setProperty("--seg-color", segmentColor(result.segment));
  badge.style.setProperty("--seg-bg", segmentSoft(result.segment));

  $("#result-name").textContent = result.segment_name;
  $("#result-description").textContent = result.description;
  $("#result-similarity").textContent = `${(Number(result.similarity) * 100).toFixed(1)}%`;
  $("#result-distance").textContent = Number(result.distance_to_centroid).toFixed(3);

  const profile = result.profile || {};
  $("#result-profile").innerHTML = [
    ["Age", profile.avg_age !== undefined ? `${decimal.format(profile.avg_age)} yrs` : "n/a"],
    ["Income", profile.avg_income !== undefined ? compactCurrency.format(profile.avg_income) : "n/a"],
    ["Spending", profile.avg_spending_score !== undefined ? decimal.format(profile.avg_spending_score) : "n/a"],
    ["Frequency", profile.avg_purchase_frequency !== undefined ? `${decimal.format(profile.avg_purchase_frequency)}/yr` : "n/a"],
  ]
    .map(
      ([label, value]) =>
        `<div class="profile-tile"><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`
    )
    .join("");

  const distances = result.centroid_distances || [];
  makeChart("distances", "chart-distances", {
    type: "bar",
    data: {
      labels: distances.map((entry) => entry.name),
      datasets: [
        {
          data: distances.map((entry) => entry.distance),
          backgroundColor: distances.map((entry) =>
            entry.segment === result.segment ? segmentColor(entry.segment) : "#cbd5e1"
          ),
          borderRadius: 6,
          barThickness: 22,
        },
      ],
    },
    options: baseChartOptions({
      options: {
        indexAxis: "y",
        scales: {
          x: { beginAtZero: true, title: { display: true, text: "Distance to centroid (scaled)" }, grid: { color: "#eef2f7" } },
          y: { grid: { display: false } },
        },
      },
      tooltip: {
        callbacks: { label: (context) => ` distance ${Number(context.raw).toFixed(3)}` },
      },
    }),
  });
}

async function savePredictedCustomer() {
  if (!state.lastPrediction) return;
  const button = $("#save-customer-btn");
  button.disabled = true;

  const payload = {
    name: $("#pred-name").value.trim() || null,
    ...state.lastPrediction.input,
  };

  try {
    const created = await api("/api/customers", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    toast(`Saved as customer #${created.id} in ${created.segment_name}.`, "success");
    await refreshDataViews();
  } catch (err) {
    toast(err.message, "error");
    button.disabled = false;
  }
}

/* --------------------------------------------------------------- analytics */
async function loadAnalytics() {
  const data = await api("/api/analytics");
  state.segments = data.segments || state.segments;

  renderBucketChart("age", "chart-age", data.age_buckets || [], "#2563eb");
  renderBucketChart("income", "chart-income", data.income_buckets || [], "#10b981");
  renderBucketChart("frequency", "chart-frequency", data.frequency_buckets || [], "#f59e0b");
  renderSegmentSpendingChart(data.segments || []);
  renderAnalyticsTable(data.segments || []);
}

function renderBucketChart(key, canvasId, buckets, color) {
  makeChart(key, canvasId, {
    type: "bar",
    data: {
      labels: buckets.map((bucket) => bucket.label),
      datasets: [
        {
          data: buckets.map((bucket) => bucket.count),
          backgroundColor: color,
          borderRadius: 7,
          maxBarThickness: 46,
        },
      ],
    },
    options: baseChartOptions({
      options: {
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: "#eef2f7" } },
        },
      },
      tooltip: {
        callbacks: { label: (context) => ` ${context.raw} customers` },
      },
    }),
  });
}

function renderSegmentSpendingChart(segments) {
  makeChart("segmentMetrics", "chart-segment-metrics", {
    type: "bar",
    data: {
      labels: segments.map((segment) => segment.name),
      datasets: [
        {
          data: segments.map((segment) => segment.avg_spending_score),
          backgroundColor: segments.map((segment) => segmentColor(segment.segment)),
          borderRadius: 7,
          maxBarThickness: 52,
        },
      ],
    },
    options: baseChartOptions({
      options: {
        scales: {
          x: { grid: { display: false }, ticks: { maxRotation: 20, minRotation: 0 } },
          y: { beginAtZero: true, max: 100, grid: { color: "#eef2f7" } },
        },
      },
      tooltip: {
        callbacks: { label: (context) => ` avg spending ${decimal.format(context.raw)}` },
      },
    }),
  });
}

function renderAnalyticsTable(segments) {
  const body = $("#analytics-table-body");
  if (!segments.length) {
    body.innerHTML = emptyRow(7, "No segment data available.");
    return;
  }
  body.innerHTML = segments
    .map(
      (segment) => `
      <tr>
        <td>${segmentBadge(segment.segment, segment.name)}</td>
        <td>${decimal.format(segment.size)}</td>
        <td>${decimal.format(segment.avg_age)} yrs</td>
        <td>${currency.format(segment.avg_income)}</td>
        <td>${decimal.format(segment.avg_spending_score)}</td>
        <td>${decimal.format(segment.avg_purchase_frequency)}</td>
        <td>${(Number(segment.share) * 100).toFixed(1)}%</td>
      </tr>`
    )
    .join("");
}

/* ------------------------------------------------------------------- model */
async function loadModel() {
  const [model, drift] = await Promise.all([
    api("/api/model"),
    api("/api/drift-report/status").catch(() => ({ available: false })),
  ]);

  const statePill = $("#model-state");
  if (model.loaded) {
    statePill.textContent = model.fallback_trained ? "Fallback model active" : "Model ready";
    statePill.classList.toggle("is-fallback", Boolean(model.fallback_trained));
  } else {
    statePill.textContent = "Model unavailable";
    statePill.classList.add("is-fallback");
  }

  const stats = [
    ["Algorithm", model.algorithm || "KMeans", "Unsupervised clustering"],
    ["Clusters (k)", model.n_clusters ?? "n/a", "Chosen for the business case"],
    ["Inertia", formatCompactNumber(model.inertia), "Lower is tighter"],
    ["Silhouette", model.silhouette_score === null || model.silhouette_score === undefined ? "n/a" : Number(model.silhouette_score).toFixed(3), "Cluster separation"],
    ["Training rows", model.rows ?? "n/a", "Reference dataset"],
    ["Trained at", model.trained_at ? formatDate(model.trained_at) : "n/a", "UTC"],
    ["scikit-learn", model.sklearn_version || "n/a", `Pandas ${model.pandas_version || "?"}`],
    ["Features", (model.features || []).length, (model.features || []).join(", ")],
  ];

  $("#model-stats").innerHTML = stats
    .map(
      ([label, value, hint]) => `
      <div class="stat-tile">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
        <small>${escapeHtml(hint)}</small>
      </div>`
    )
    .join("");

  const segments = model.segments || [];
  $("#centers-table-body").innerHTML = segments.length
    ? segments
        .map(
          (segment) => `
          <tr>
            <td>${segmentBadge(segment.segment, segment.name)}</td>
            <td>${decimal.format(segment.avg_age)}</td>
            <td>${currency.format(segment.avg_income)}</td>
            <td>${decimal.format(segment.avg_spending_score)}</td>
            <td>${decimal.format(segment.avg_purchase_frequency)}</td>
          </tr>`
        )
        .join("")
    : emptyRow(5, "No centroid data available.");

  const driftStatus = $("#drift-status");
  const loadButton = $("#load-drift-btn");
  if (drift.available) {
    const sizeKb = drift.size_bytes ? `${(drift.size_bytes / 1024).toFixed(0)} KB` : "";
    driftStatus.textContent = `Report available · generated ${formatDate(drift.generated_at)}${sizeKb ? ` · ${sizeKb}` : ""}`;
    loadButton.disabled = false;
  } else {
    driftStatus.textContent = "No drift report has been generated yet. Run `python -m monitoring.monitor` to create one.";
    loadButton.disabled = true;
  }
}

function loadDriftReport() {
  const frame = $("#drift-frame");
  const container = $("#drift-frame-container");
  container.hidden = false;
  if (frame.getAttribute("src") !== "/api/drift-report") {
    frame.setAttribute("src", "/api/drift-report");
  }
}

/* -------------------------------------------------------------- navigation */
function switchView(name) {
  const meta = VIEW_META[name];
  if (!meta) return;

  state.view = name;
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.view === name);
  });
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === `view-${name}`);
  });

  $("#page-title").textContent = meta.title;
  $("#page-subtitle").textContent = meta.subtitle;
  closeSidebar();

  meta.load().catch((err) => {
    toast(`Could not load ${meta.title.toLowerCase()}: ${err.message}`, "error");
  });
}

function openSidebar() {
  $("#sidebar").classList.add("open");
  $("#backdrop").classList.add("show");
}

function closeSidebar() {
  $("#sidebar").classList.remove("open");
  $("#backdrop").classList.remove("show");
}

/* -------------------------------------------------------------------- init */
function setupEvents() {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => switchView(item.dataset.view));
  });

  document.querySelectorAll("[data-go]").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.go));
  });

  $("#menu-toggle").addEventListener("click", () => {
    if ($("#sidebar").classList.contains("open")) closeSidebar();
    else openSidebar();
  });
  $("#backdrop").addEventListener("click", closeSidebar);

  $("#refresh-btn").addEventListener("click", async () => {
    await checkHealth();
    VIEW_META[state.view].load().catch((err) => toast(err.message, "error"));
  });

  // customers
  let searchTimer = null;
  $("#customer-search").addEventListener("input", (event) => {
    window.clearTimeout(searchTimer);
    const value = event.target.value.trim();
    searchTimer = window.setTimeout(() => {
      state.filters.search = value;
      state.filters.page = 1;
      loadCustomers().catch((err) => toast(err.message, "error"));
    }, 300);
  });

  $("#customer-segment-filter").addEventListener("change", (event) => {
    state.filters.segment = event.target.value;
    state.filters.page = 1;
    loadCustomers().catch((err) => toast(err.message, "error"));
  });

  $("#page-size").addEventListener("change", (event) => {
    state.filters.pageSize = Number(event.target.value);
    state.filters.page = 1;
    loadCustomers().catch((err) => toast(err.message, "error"));
  });

  $("#prev-page").addEventListener("click", () => {
    if (state.filters.page > 1) {
      state.filters.page -= 1;
      loadCustomers().catch((err) => toast(err.message, "error"));
    }
  });

  $("#next-page").addEventListener("click", () => {
    if (state.filters.page < state.customers.pages) {
      state.filters.page += 1;
      loadCustomers().catch((err) => toast(err.message, "error"));
    }
  });

  $("#add-customer-btn").addEventListener("click", () => openCustomerModal());
  $("#export-csv-btn").addEventListener("click", exportCsv);

  $("#customers-table-body").addEventListener("click", (event) => {
    const editButton = event.target.closest("[data-edit]");
    if (editButton) {
      const customer = state.customers.items.find((item) => item.id === Number(editButton.dataset.edit));
      if (customer) openCustomerModal(customer);
      return;
    }
    const deleteButton = event.target.closest("[data-delete]");
    if (deleteButton) {
      const customer = state.customers.items.find((item) => item.id === Number(deleteButton.dataset.delete));
      if (customer) askDelete(customer);
    }
  });

  $("#customer-form").addEventListener("submit", submitCustomerForm);
  $("#modal-close").addEventListener("click", closeCustomerModal);
  $("#modal-cancel").addEventListener("click", closeCustomerModal);
  $("#customer-modal-backdrop").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) closeCustomerModal();
  });

  $("#confirm-cancel").addEventListener("click", () => {
    $("#confirm-backdrop").hidden = true;
    state.pendingDelete = null;
  });
  $("#confirm-ok").addEventListener("click", confirmDelete);
  $("#confirm-backdrop").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) {
      $("#confirm-backdrop").hidden = true;
      state.pendingDelete = null;
    }
  });

  // segmentation
  $("#predict-form").addEventListener("submit", handlePredict);
  $("#save-customer-btn").addEventListener("click", savePredictedCustomer);

  // model
  $("#load-drift-btn").addEventListener("click", loadDriftReport);

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (!$("#customer-modal-backdrop").hidden) closeCustomerModal();
    if (!$("#confirm-backdrop").hidden) {
      $("#confirm-backdrop").hidden = true;
      state.pendingDelete = null;
    }
    closeSidebar();
  });
}

async function checkHealth() {
  try {
    const health = await api("/health");
    setStatus(health.status === "ok", health.status === "ok" ? `API online · ${health.customers} customers` : "API degraded");
  } catch {
    setStatus(false, "API offline");
  }
}

function init() {
  applyChartDefaults();
  setupEvents();
  checkHealth();
  switchView("dashboard");
}

init();
