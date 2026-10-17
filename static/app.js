const state = {
  challenges: [],
  bankId: window.DEFAULT_BANK_ID,
  bank: window.BANK_SUMMARIES[window.DEFAULT_BANK_ID],
  unified: window.UNIFIED_SUMMARY,
};

const byId = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function optionalNumber(id) {
  const value = byId(id).value.trim();
  return value === "" ? null : Number(value);
}

function setMessage(element, text, type = "error") {
  element.textContent = text;
  element.className = `message ${type}`;
  element.hidden = !text;
}

function activateWorkspace(name) {
  document.querySelectorAll(".workspace").forEach((item) => item.classList.toggle("active", item.id === `workspace-${name}`));
  document.querySelectorAll("[data-workspace]").forEach((item) => item.classList.toggle("active", item.dataset.workspace === name));
}

function activateMode(group, name) {
  document.querySelectorAll(`[data-${group}-mode]`).forEach((item) => item.classList.toggle("active", item.dataset[`${group}Mode`] === name));
  document.querySelectorAll(`#workspace-${group === "test" ? "test" : "library"} .mode-panel`).forEach((item) => {
    item.classList.toggle("active", item.id === `${group}-${name}` || item.id === `library-${name}`);
  });
}

async function loadChallenges() {
  byId("regenerate").disabled = true;
  byId("result").hidden = true;
  setMessage(byId("test-message"), "");
  const response = await fetch("/api/challenges");
  state.challenges = (await response.json()).challenges;
  renderChallenges();
  byId("regenerate").disabled = false;
}

function renderChallenges() {
  byId("challenge-list").innerHTML = state.challenges.map((challenge, index) => `
    <article class="challenge-item">
      <div class="challenge-header">
        <strong>挑战 ${index + 1}</strong>
        <span>${challenge.expected_count} 个数字</span>
        <button type="button" data-copy="${index}">复制提示词</button>
      </div>
      <div class="challenge-columns">
        <div><label>发送给待测模型</label><pre>${escapeHtml(challenge.prompt)}</pre></div>
        <div><label for="output-${index}">粘贴完整输出</label><textarea id="output-${index}" spellcheck="false" placeholder="保留文字、标点、代码块和完整数字序列"></textarea></div>
      </div>
    </article>
  `).join("");
  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      await navigator.clipboard.writeText(state.challenges[Number(button.dataset.copy)].prompt);
      button.textContent = "已复制";
      window.setTimeout(() => { button.textContent = "复制提示词"; }, 1000);
    });
  });
}

function renderResult(payload) {
  const diagnostics = payload.diagnostics.map((item, index) => `
    <span class="diagnostic ${item.accepted ? "accepted" : "rejected"}">挑战 ${index + 1}: ${item.parsed_numbers} 个数字 · ${item.accepted ? "计入" : "忽略"}</span>
  `).join("");
  const rows = payload.results.map((item, index) => `
    <tr class="${index === 0 ? "winner" : ""}">
      <td>${index + 1}</td><td><strong>${escapeHtml(item.display_name)}</strong></td><td>${escapeHtml(item.family_name)}</td>
      <td><div class="probability-cell"><span><i style="width:${item.probability * 100}%"></i></span><strong>${percent(item.probability)}</strong></div></td>
      <td>${percent(item.profile_similarity)}</td>
    </tr>
  `).join("");
  const apiNote = payload.api_test
    ? `<span>API 获得 ${payload.api_test.received}/${payload.api_test.requested} 份有效回答，实际尝试 ${payload.api_test.attempted}/${payload.api_test.max_attempts}${payload.api_test.errors.length ? `，${payload.api_test.errors.length} 次未采用` : ""}</span>`
    : "";
  byId("result").innerHTML = `
    <div class="result-summary">
      <div><span>最可能模型</span><strong>${escapeHtml(payload.prediction_name)}</strong></div>
      <div><span>统一库概率</span><strong>${percent(payload.probability)}</strong></div>
      <div><span>自动识别家族</span><strong>${escapeHtml(payload.family_prediction_name)} · ${percent(payload.family_probability)}</strong></div>
      <div><span>有效查询</span><strong>${payload.used_outputs}/3</strong></div>
    </div>
    <div class="diagnostics">${diagnostics}</div>
    <div class="table-wrap"><table><thead><tr><th>排序</th><th>候选模型</th><th>家族</th><th>归因概率</th><th>分布相似度</th></tr></thead><tbody>${rows}</tbody></table></div>
    ${apiNote ? `<div class="result-note">${apiNote}</div>` : ""}
    <div class="result-guidance" role="note" aria-label="结果说明">
      <p>本工具仅对指纹库内的模型进行归因；若待测模型不在指纹库中，得到任何结果都有可能。</p>
      <p>Claude Code 的系统提示词会影响模型偏好，测试结果存在较大偏差，建议不要在 Claude Code 中测试。</p>
    </div>
  `;
  byId("result").hidden = false;
  byId("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function analyzeManual() {
  const button = byId("analyze");
  button.disabled = true;
  setMessage(byId("test-message"), "正在计算……", "working");
  const outputs = state.challenges.map((challenge, index) => ({
    text: byId(`output-${index}`).value,
    expected_count: challenge.expected_count,
  }));
  const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ outputs }) });
  const payload = await response.json();
  if (response.ok) {
    setMessage(byId("test-message"), "");
    renderResult(payload);
  } else {
    setMessage(byId("test-message"), payload.error || "无法完成归因。", "error");
    byId("result").hidden = true;
  }
  button.disabled = false;
}

function renderApiProgress(states, status) {
  const valid = states.filter((state) => state === "done").length;
  const attempted = states.filter((state) => ["done", "invalid", "error"].includes(state)).length;
  const target = 3;
  byId("api-test-progress").hidden = false;
  byId("api-progress-status").textContent = status;
  byId("api-progress-count").textContent = `有效 ${valid}/${target} · 已尝试 ${attempted}/${states.length}`;
  byId("api-progress-fill").style.width = `${(valid / target) * 100}%`;
  byId("api-progress-steps").innerHTML = states.map((state, index) => {
    const labels = { pending: "等待", working: "请求中", done: "有效", invalid: "数字不足", error: "接口失败", skipped: "无需调用" };
    return `<span class="progress-step ${state}"><b>${index + 1}</b>挑战 ${index + 1} · ${labels[state]}</span>`;
  }).join("");
}

async function testViaApi(event) {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  byId("result").hidden = true;
  setMessage(byId("test-message"), "");

  const challengeResponse = await fetch("/api/challenges");
  const firstBatch = (await challengeResponse.json()).challenges;
  const retryResponse = await fetch("/api/challenges");
  const challenges = firstBatch.concat((await retryResponse.json()).challenges);
  const states = challenges.map(() => "pending");
  const outputs = [];
  const errors = [];
  const target = 3;
  const configuration = {
    base_url: byId("test-api-base").value,
    api_key: byId("test-api-key").value,
    api_model: byId("test-api-model").value,
    temperature: optionalNumber("test-temperature"),
  };
  renderApiProgress(states, "已生成独立挑战，准备调用模型");

  for (let index = 0; index < challenges.length && outputs.length < target; index += 1) {
    states[index] = "working";
    renderApiProgress(states, `正在进行第 ${index + 1} 次尝试，等待模型完整输出……`);
    try {
      const response = await fetch("/api/test/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...configuration,
          prompt: challenges[index].prompt,
          expected_count: challenges[index].expected_count,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "接口请求失败");
      if (payload.accepted) {
        outputs.push({ text: payload.text, expected_count: challenges[index].expected_count });
        states[index] = "done";
      } else {
        errors.push(`尝试 ${index + 1}: 有效数字 ${payload.parsed_numbers}/${payload.minimum_numbers}`);
        states[index] = "invalid";
      }
    } catch (error) {
      errors.push(`尝试 ${index + 1}: ${error.message}`);
      states[index] = "error";
    }
    renderApiProgress(states, `当前已有 ${outputs.length}/${target} 份有效回答`);
  }

  if (outputs.length === target) {
    states.forEach((state, index) => { if (state === "pending") states[index] = "skipped"; });
  }

  if (!outputs.length) {
    renderApiProgress(states, "六次尝试后仍没有可用回答");
    setMessage(byId("test-message"), `没有获得可分析输出。${errors[0] || ""}`, "error");
    button.disabled = false;
    return;
  }

  renderApiProgress(states, "模型回答已收齐，正在计算归因概率……");
  const analysisResponse = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ outputs }),
  });
  const result = await analysisResponse.json();
  if (analysisResponse.ok) {
    const attempted = states.filter((state) => ["done", "invalid", "error"].includes(state)).length;
    result.api_test = { requested: target, attempted, max_attempts: challenges.length, received: outputs.length, errors };
    renderApiProgress(states, `测试完成：${outputs.length}/${target} 份有效回答进入归因`);
    renderResult(result);
  } else {
    setMessage(byId("test-message"), result.error || "API 自动测试失败。", "error");
  }
  button.disabled = false;
}

function updateUnifiedSummary(summary) {
  state.unified = summary;
  byId("topbar-bank-count").textContent = `${summary.model_count} 个候选模型`;
  byId("active-bank-badge").textContent = `${summary.model_count} 个候选模型`;
}

function renderInventory() {
  byId("selected-bank-name").textContent = state.bank.label;
  byId("model-options").innerHTML = state.bank.models.map((model) => `<option value="${escapeHtml(model.id)}"></option>`).join("");
  byId("bank-inventory").innerHTML = state.bank.models.length
    ? state.bank.models.map((model) => {
      const channels = Object.keys(model.channels || {});
      const suffix = channels.length > 1 ? ` · ${channels.length} 渠道` : "";
      return `<span class="fingerprint-item" title="${escapeHtml(channels.join(", "))}">${escapeHtml(model.display_name)}${suffix}</span>`;
    }).join("")
    : `<span class="empty-inventory">暂无指纹</span>`;
}

async function refreshBank() {
  const response = await fetch(`/api/bank?bank_id=${encodeURIComponent(state.bankId)}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "无法读取指纹库");
  state.bank = payload;
  renderInventory();
}

async function selectBank(bankId) {
  state.bankId = bankId;
  await refreshBank();
}

async function enrollAutomatically(event) {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  const requested = Number(byId("sample-count").value);
  const started = Date.now();
  const progressTimer = window.setInterval(() => {
    const seconds = Math.floor((Date.now() - started) / 1000);
    setMessage(byId("enrollment-message"), `正在自动识别协议并采集 ${requested} 份回答 · 已等待 ${seconds} 秒`, "working");
  }, 1000);
  setMessage(byId("enrollment-message"), `正在自动识别协议并采集 ${requested} 份回答`, "working");
  let response;
  try {
    response = await fetch("/api/enroll/auto", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        base_url: byId("api-base").value,
        api_key: byId("api-key").value,
        api_model: byId("api-model").value,
        bank_id: state.bankId,
        model_label: byId("auto-model").value,
        sample_count: requested,
        temperature: optionalNumber("temperature"),
      }),
    });
  } catch (error) {
    window.clearInterval(progressTimer);
    setMessage(byId("enrollment-message"), error.message, "error");
    button.disabled = false;
    return;
  }
  window.clearInterval(progressTimer);
  const payload = await response.json();
  if (response.ok) {
    state.bank = payload.bank;
    updateUnifiedSummary(payload.unified);
    renderInventory();
    setMessage(byId("enrollment-message"), `采集完成：收到 ${payload.received}/${payload.requested} 份，${payload.accepted} 份进入指纹库，${payload.rejected} 份无效，${payload.errors.length} 次接口错误。`, "success");
  } else {
    setMessage(byId("enrollment-message"), payload.error || "自动采集失败。", "error");
  }
  button.disabled = false;
}

function renderBankOptions(summaries, selected) {
  byId("bank-select").innerHTML = Object.entries(summaries)
    .map(([bankId, bank]) => `<option value="${escapeHtml(bankId)}"${bankId === selected ? " selected" : ""}>${escapeHtml(bank.label)}</option>`)
    .join("");
}

async function createBank(event) {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  const response = await fetch("/api/banks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label: byId("new-bank-name").value }),
  });
  const payload = await response.json();
  if (response.ok) {
    window.BANK_SUMMARIES = payload.banks;
    state.bankId = payload.bank.id;
    state.bank = payload.bank;
    updateUnifiedSummary(payload.unified);
    renderBankOptions(payload.banks, state.bankId);
    renderInventory();
    byId("new-bank-name").value = "";
    byId("create-bank-form").hidden = true;
    setMessage(byId("enrollment-message"), `已创建 ${payload.bank.label}`, "success");
  } else {
    setMessage(byId("enrollment-message"), payload.error || "创建失败。", "error");
  }
  button.disabled = false;
}

document.querySelectorAll("[data-workspace]").forEach((button) => button.addEventListener("click", () => activateWorkspace(button.dataset.workspace)));
document.querySelectorAll("[data-test-mode]").forEach((button) => button.addEventListener("click", () => activateMode("test", button.dataset.testMode)));
byId("bank-select").addEventListener("change", (event) => selectBank(event.target.value));
byId("regenerate").addEventListener("click", loadChallenges);
byId("analyze").addEventListener("click", analyzeManual);
byId("api-test-form").addEventListener("submit", testViaApi);
byId("auto-enrollment").addEventListener("submit", enrollAutomatically);
byId("show-create-bank").addEventListener("click", () => { byId("create-bank-form").hidden = !byId("create-bank-form").hidden; });
byId("create-bank-form").addEventListener("submit", createBank);

// ---------------------------------------------------------------- 多渠道整合校准

const channelState = { mode: "api", nextId: 1 };
const VERDICT_LABELS = { consistent: "一致", weak: "偏弱", suspect: "可疑", warning: "预警", unknown: "无法校验" };
const MODE_LABELS = { against_trusted: "对照可信渠道", leave_one_channel_out: "留一渠道", anchor_only: "仅库内认领" };

function channelCardHtml(id) {
  return `
    <article class="channel-card" data-channel-id="${id}">
      <div class="channel-card-header"><span>渠道 ${id}</span><button type="button" data-remove-channel="${id}">移除</button></div>
      <div class="form-grid two">
        <label><span>渠道名称</span><input data-field="name" required placeholder="例如 官方 / 中转A"></label>
        <label class="api-only"><span>Base URL</span><input data-field="base_url" placeholder="https://example.com/v1"></label>
        <label class="api-only"><span>接口模型名</span><input data-field="api_model" placeholder="该渠道实际使用的模型名"></label>
        <label class="api-only"><span>API Key</span><input data-field="api_key" type="password" autocomplete="off"></label>
        <label class="paste-only wide" style="grid-column: 1 / -1"><span>该渠道的回答（多份之间用单独一行 ===OUTPUT=== 分隔，或粘贴 jsonl）</span><textarea data-field="text" spellcheck="false"></textarea></label>
      </div>
    </article>`;
}

function addChannel() {
  const id = channelState.nextId;
  channelState.nextId += 1;
  byId("channel-list").insertAdjacentHTML("beforeend", channelCardHtml(id));
  byId("channel-list").querySelector(`[data-remove-channel="${id}"]`).addEventListener("click", () => {
    if (byId("channel-list").children.length <= 1) return;
    byId("channel-list").querySelector(`[data-channel-id="${id}"]`).remove();
  });
}

function setChannelMode(mode) {
  channelState.mode = mode;
  byId("multi-channel-form").dataset.mode = mode;
  document.querySelectorAll("[data-channel-mode]").forEach((item) => item.classList.toggle("active", item.dataset.channelMode === mode));
}

function collectChannels() {
  return [...byId("channel-list").querySelectorAll(".channel-card")].map((card) => {
    const value = (field) => card.querySelector(`[data-field="${field}"]`).value;
    return channelState.mode === "api"
      ? { name: value("name"), base_url: value("base_url"), api_model: value("api_model"), api_key: value("api_key") }
      : { name: value("name"), text: value("text") };
  });
}

function reportTableHtml(report) {
  const rows = report.entries.map((entry) => `
    <tr>
      <td><strong>${escapeHtml(entry.channel)}</strong></td>
      <td>${entry.responses}</td>
      <td>${escapeHtml(MODE_LABELS[entry.mode] || entry.mode)}</td>
      <td>${entry.target_top1_rate == null ? "—" : percent(entry.target_top1_rate)}</td>
      <td>${entry.pooled_prediction ? `${escapeHtml(entry.pooled_prediction)} · ${percent(entry.pooled_probability)}` : "—"}</td>
      <td><span class="verdict ${entry.verdict}">${VERDICT_LABELS[entry.verdict] || entry.verdict}</span></td>
      <td>${escapeHtml(entry.reason)}</td>
    </tr>`).join("");
  const pairs = report.pairwise.map((pair) => `${escapeHtml(pair.left)} ↔ ${escapeHtml(pair.right)} ${percent(pair.distribution_similarity)}`).join("；");
  return `
    <div class="channel-report">
      <h3>${escapeHtml(report.model)}${report.trusted_channels.length ? ` · 可信基准：${report.trusted_channels.map(escapeHtml).join("、")}` : ""}</h3>
      <div class="table-wrap"><table><thead><tr><th>渠道</th><th>回答</th><th>校验方式</th><th>单条命中</th><th>整体归因</th><th>结论</th><th>说明</th></tr></thead><tbody>${rows}</tbody></table></div>
      ${pairs ? `<div class="result-note">数字分布相似度：${pairs}</div>` : ""}
    </div>`;
}

function calibrationHtml(calibration) {
  if (!calibration) return "";
  const items = ["1", "2", "3"].filter((key) => calibration[key]).map((key) => {
    const item = calibration[key];
    const accuracy = item.cv_accuracy == null ? "—" : percent(item.cv_accuracy);
    return `<div><span>${key} 次查询 · β=${Number(item.beta).toFixed(2)}</span><strong>${accuracy}</strong></div>`;
  }).join("");
  return `<div class="result-summary">${items}<div><span>交叉验证准确率</span><strong>统一库</strong></div></div>`;
}

function applyBankPayload(payload) {
  if (payload.banks) window.BANK_SUMMARIES = payload.banks;
  if (payload.bank && payload.bank.id === state.bankId) {
    state.bank = payload.bank;
    renderInventory();
  }
  if (payload.unified) updateUnifiedSummary(payload.unified);
}

async function submitMultiChannel(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button[type=submit]");
  const channels = collectChannels();
  const message = byId("multi-channel-message");
  if (!channels.length) { setMessage(message, "请至少添加一个渠道"); return; }
  button.disabled = true;
  byId("multi-channel-result").hidden = true;
  const isApi = channelState.mode === "api";
  const started = Date.now();
  const label = isApi ? `正在从 ${channels.length} 个渠道并行采集并校验` : `正在校验 ${channels.length} 个渠道的回答`;
  const timer = window.setInterval(() => {
    setMessage(message, `${label} · 已等待 ${Math.floor((Date.now() - started) / 1000)} 秒（重新校准可能需要一两分钟）`, "working");
  }, 1000);
  setMessage(message, label, "working");
  try {
    const body = {
      bank_id: state.bankId,
      model_label: byId("multi-model").value,
      channels,
      keep_suspect: byId("multi-keep-suspect").checked,
      dry_run: byId("multi-dry-run").checked,
    };
    if (isApi) {
      body.sample_count = Number(byId("multi-sample-count").value);
      body.temperature = optionalNumber("multi-temperature");
    }
    const response = await fetch(isApi ? "/api/enroll/multi" : "/api/channels/import", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "整合失败");
    applyBankPayload(payload);
    const collection = (payload.collection || []).map((item) => `${escapeHtml(item.name)} ${item.received}/${item.requested}${item.errors.length ? `（${item.errors.length} 次失败）` : ""}`).join("；");
    const replaced = payload.replaced_responses ? `；替换同渠道旧回答 ${payload.replaced_responses} 条` : "";
    const excluded = payload.excluded_responses
      ? `<div class="result-note warn">其中 ${payload.excluded_responses} 份回答所在环境（${payload.excluded_conditions.map(escapeHtml).join("、")}）并非所有模型都覆盖，已写入参考数据，但不参与指纹估计。</div>`
      : "";
    byId("multi-channel-result").innerHTML = `
      ${reportTableHtml(payload.report)}
      ${calibrationHtml(payload.unified_calibration)}
      <div class="result-note"><span>${collection ? `采集：${collection}` : ""}</span><span>${payload.dry_run ? "只校验模式，未写入指纹库" : `并入渠道：${payload.accepted_channels.map(escapeHtml).join("、") || "无"}；拒绝：${payload.rejected_channels.map(escapeHtml).join("、") || "无"}${replaced}`}</span></div>
      ${excluded}`;
    byId("multi-channel-result").hidden = false;
    setMessage(message, payload.dry_run
      ? "校验完成（未写入）。"
      : `整合完成：${payload.accepted_responses} 份回答并入 ${payload.model}，指纹库已重新校准。`, payload.rejected_channels.length ? "working" : "success");
  } catch (error) {
    setMessage(message, error.message, "error");
  } finally {
    window.clearInterval(timer);
    button.disabled = false;
  }
}

async function showChannelReport() {
  const message = byId("multi-channel-message");
  setMessage(message, "正在计算渠道一致性……", "working");
  const response = await fetch(`/api/channels/report?bank_id=${encodeURIComponent(state.bankId)}`);
  const payload = await response.json();
  if (!response.ok) { setMessage(message, payload.error || "无法生成报告"); return; }
  if (!payload.reports.length) {
    setMessage(message, "当前指纹库还没有拥有两个以上渠道的模型。", "success");
    byId("multi-channel-result").hidden = true;
    return;
  }
  setMessage(message, "");
  byId("multi-channel-result").innerHTML = payload.reports.map(reportTableHtml).join("");
  byId("multi-channel-result").hidden = false;
}

async function recalibrateBank() {
  const button = byId("recalibrate");
  const message = byId("multi-channel-message");
  button.disabled = true;
  setMessage(message, "正在重建并重新校准，可能需要一两分钟……", "working");
  try {
    const response = await fetch("/api/calibrate", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ bank_id: state.bankId }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "校准失败");
    applyBankPayload(payload);
    byId("multi-channel-result").innerHTML = calibrationHtml(payload.calibration);
    byId("multi-channel-result").hidden = false;
    const channel = payload.channel_calibration || {};
    setMessage(message, `校准完成${channel.enabled ? `：已启用渠道感知（渠道干扰维度 ${channel.channel_nuisance_rank}，跨渠道留出验证）` : "：库中暂无多渠道模型，沿用原算法"}。`, "success");
  } catch (error) {
    setMessage(message, error.message, "error");
  } finally {
    button.disabled = false;
  }
}

document.querySelectorAll("[data-channel-mode]").forEach((button) => button.addEventListener("click", () => setChannelMode(button.dataset.channelMode)));
byId("add-channel").addEventListener("click", addChannel);
byId("multi-channel-form").addEventListener("submit", submitMultiChannel);
byId("channel-report").addEventListener("click", showChannelReport);
byId("recalibrate").addEventListener("click", recalibrateBank);
setChannelMode("api");
addChannel();
addChannel();

renderInventory();
loadChallenges();
