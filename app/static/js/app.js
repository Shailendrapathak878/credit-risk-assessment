const form = document.getElementById("applicant-form");
const button = document.getElementById("assess-button");
const errorBox = document.getElementById("form-error");
const statusEl = document.getElementById("api-status");
const result = document.getElementById("result");
const empty = document.getElementById("empty-state");
const currencyInputs = [...form.querySelectorAll(".currency-input")];
const ratioInput = document.getElementById("loan-percent-income-display");
const fields = [
  "person_age", "person_income", "person_home_ownership", "person_emp_length",
  "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
  "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
];
const categoricalFields = new Set(["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]);
const featureLabels = {
  loan_percent_income: "Loan-to-Income Ratio",
  person_income: "Annual Income",
  loan_amnt: "Loan Amount",
  loan_int_rate: "Interest Rate",
  person_emp_length: "Employment Length",
  person_age: "Age",
  person_home_ownership: "Home Ownership",
  loan_intent: "Loan Purpose",
  loan_grade: "Credit Grade",
  cb_person_default_on_file: "Previous Loan Default",
  cb_person_cred_hist_length: "Credit History Length",
};
const featureGroups = {
  income_to_loan_ratio: "loan_percent_income",
  loan_percent_income: "loan_percent_income",
  credit_history_age_ratio: "cb_person_cred_hist_length",
  cb_person_cred_hist_length: "cb_person_cred_hist_length",
  employment_stability_ratio: "person_emp_length",
  person_emp_length: "person_emp_length",
  person_age: "person_age",
  person_income: "person_income",
  loan_amnt: "loan_amnt",
  loan_int_rate: "loan_int_rate",
  person_home_ownership: "person_home_ownership",
  loan_intent: "loan_intent",
  loan_grade: "loan_grade",
  cb_person_default_on_file: "cb_person_default_on_file",
};
const purposeLabels = {
  PERSONAL: "personal", EDUCATION: "education", MEDICAL: "medical", VENTURE: "business venture",
  DEBTCONSOLIDATION: "debt consolidation", HOMEIMPROVEMENT: "home improvement",
};
const ownershipLabels = { RENT: "rented home", MORTGAGE: "mortgage home", OWN: "owned home", OTHER: "home ownership" };
const categoryTitles = {
  RENT: "Rent", MORTGAGE: "Mortgage", OWN: "Own", OTHER: "Other",
  PERSONAL: "Personal", EDUCATION: "Education", MEDICAL: "Medical", VENTURE: "Business venture",
  DEBTCONSOLIDATION: "Debt consolidation", HOMEIMPROVEMENT: "Home improvement",
  A: "A", B: "B", C: "C", D: "D", E: "E", F: "F", G: "G", N: "No", Y: "Yes",
};

function parseNumber(value) {
  const normalized = String(value ?? "").replace(/[₹,\s]/g, "");
  return normalized === "" ? NaN : Number(normalized);
}

function formatIndianNumber(value) {
  const number = parseNumber(value);
  return Number.isFinite(number) ? new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(number) : "";
}

function formatCurrency(value) {
  return Number.isFinite(Number(value)) ? `₹${formatIndianNumber(value)}` : "the entered amount";
}

function formatCurrencyInput(input) {
  if (input.value.trim() !== "") input.value = formatIndianNumber(input.value);
}

function unformatCurrencyInput(input) {
  input.value = input.value.replace(/[₹,\s]/g, "");
}

function validateCurrencyInputs() {
  currencyInputs.forEach((input) => {
    const number = parseNumber(input.value);
    const minimum = Number(input.dataset.min);
    input.setCustomValidity(Number.isFinite(number) && number >= minimum ? "" : `Enter an amount of at least ₹${formatIndianNumber(minimum)}.`);
  });
}

function updateLoanToIncomeRatio() {
  const annualIncome = parseNumber(form.elements.person_income.value);
  const loanAmount = parseNumber(form.elements.loan_amnt.value);
  if (!Number.isFinite(annualIncome) || annualIncome <= 0 || !Number.isFinite(loanAmount) || loanAmount < 0) {
    ratioInput.value = "—";
    ratioInput.dataset.apiValue = "";
    return;
  }
  const ratio = (loanAmount / annualIncome) * 100;
  if (!Number.isFinite(ratio)) {
    ratioInput.value = "—";
    ratioInput.dataset.apiValue = "";
    return;
  }
  ratioInput.value = `${ratio.toFixed(2)}%`;
  ratioInput.dataset.apiValue = String(ratio / 100);
}

function payload() {
  const data = new FormData(form);
  const output = {};
  fields.forEach((field) => {
    const value = data.get(field);
    if (categoricalFields.has(field)) output[field] = value;
    else if (field === "loan_percent_income") output[field] = Number(ratioInput.dataset.apiValue);
    else output[field] = currencyInputs.some((input) => input.name === field) ? parseNumber(value) : Number(value);
  });
  return output;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.classList.add("hidden");
}

function clearAssessment() {
  result.classList.add("hidden");
  document.getElementById("explanation").classList.add("hidden");
  document.getElementById("positive-factors").innerHTML = "";
  document.getElementById("negative-factors").innerHTML = "";
  document.getElementById("technical-positive").innerHTML = "";
  document.getElementById("technical-negative").innerHTML = "";
  document.getElementById("technical-details").open = false;
}

function markInvalidFields() {
  [...form.elements].forEach((element) => {
    if (element.willValidate) element.classList.toggle("invalid", !element.checkValidity());
  });
}

function clearInvalidField(event) {
  if (event.target.willValidate && event.target.checkValidity()) event.target.classList.remove("invalid");
}

function renderResult(data) {
  const isHighRisk = Boolean(data.risk_class);
  document.getElementById("probability").textContent = `${(data.risk_probability * 100).toFixed(2)}%`;
  document.getElementById("risk-label").textContent = data.risk_label.toUpperCase();
  document.getElementById("risk-label").className = `risk-pill ${isHighRisk ? "high" : "low"}`;
  const probability = (data.risk_probability * 100).toFixed(2);
  document.getElementById("risk-summary").textContent = `The model estimates a ${probability}% likelihood that this loan could result in default.`;
  const meter = document.getElementById("meter-fill");
  meter.style.width = `${data.risk_probability * 100}%`;
  meter.className = isHighRisk ? "high" : "low";
  meter.setAttribute("aria-valuenow", String((data.risk_probability * 100).toFixed(2)));
  empty.classList.add("hidden");
  result.classList.remove("hidden");
}

function parseFactor(feature) {
  let clean = feature.replace(/^(numerical|categorical)__/, "");
  const explicitCategory = clean.match(/^(.+?) = (.+)$/);
  if (explicitCategory) return { name: explicitCategory[1], category: explicitCategory[2] };
  for (const name of ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]) {
    if (clean.startsWith(`${name}_`)) return { name, category: clean.slice(name.length + 1) };
  }
  return { name: clean, category: "" };
}

function factorValue(payloadData, group) {
  if (group === "person_home_ownership") return ownershipLabels[payloadData.person_home_ownership] || "the selected home type";
  if (group === "loan_intent") return purposeLabels[payloadData.loan_intent] || "the selected purpose";
  if (group === "loan_grade") return payloadData.loan_grade;
  if (group === "cb_person_default_on_file") return payloadData.cb_person_default_on_file === "Y" ? "yes" : "no";
  if (group === "person_income" || group === "loan_amnt") return formatCurrency(payloadData[group]);
  if (group === "loan_int_rate") return `${payloadData.loan_int_rate}%`;
  if (group === "loan_percent_income") return `${(payloadData.loan_percent_income * 100).toFixed(1)}%`;
  if (group === "person_emp_length" || group === "cb_person_cred_hist_length") return `${payloadData[group]} years`;
  if (group === "person_age") return `${payloadData.person_age} years`;
  return "the submitted applicant information";
}

function factorExplanation(group, value, kind) {
  const riskPhrase = kind === "positive" ? "increased the model's estimated risk." : "contributed toward a lower estimated risk.";
  const descriptions = {
    loan_percent_income: `The requested loan represents approximately ${value} of annual income and ${riskPhrase}`,
    loan_int_rate: `The annual interest rate is ${value} and ${riskPhrase}`,
    loan_amnt: `The requested loan amount is ${value} and ${riskPhrase}`,
    person_income: `The applicant's annual income is ${value} and ${riskPhrase}`,
    person_emp_length: `Employment length of ${value} ${riskPhrase}`,
    person_home_ownership: `Your current home ownership status (${value}) ${riskPhrase}`,
    loan_intent: `The ${value} loan purpose ${riskPhrase}`,
    loan_grade: `Credit grade ${value} ${riskPhrase}`,
    cb_person_default_on_file: value === "yes" ? "A previous loan default was recorded and increased the model's estimated risk." : "No previous loan default was recorded and this contributed toward a lower estimated risk.",
    cb_person_cred_hist_length: `A recorded credit history of ${value} ${riskPhrase}`,
    person_age: `The applicant's age of ${value} ${riskPhrase}`,
  };
  return descriptions[group] || `This information contributed toward the model's estimated risk.`;
}

function aggregateFactors(positiveFactors, negativeFactors, payloadData) {
  const groups = new Map();
  [...positiveFactors, ...negativeFactors].forEach((factor) => {
    const parsed = parseFactor(factor.feature);
    const group = featureGroups[parsed.name];
    if (!group) return;
    if (!groups.has(group)) groups.set(group, { group, shapValue: 0, entries: [] });
    const current = groups.get(group);
    current.shapValue += Number(factor.shap_value);
    current.entries.push({ factor, category: parsed.category });
  });
  return [...groups.values()]
    .map((item) => {
      const value = factorValue(payloadData, item.group);
      const selectedCategory = {
        person_home_ownership: payloadData.person_home_ownership,
        loan_intent: payloadData.loan_intent,
        loan_grade: payloadData.loan_grade,
        cb_person_default_on_file: payloadData.cb_person_default_on_file,
      }[item.group];
      const category = selectedCategory ? categoryTitles[selectedCategory] || selectedCategory : "";
      const title = category ? `${featureLabels[item.group]} — ${category}` : featureLabels[item.group];
      return {
        ...item,
        title,
        value,
        description: factorExplanation(item.group, value, item.shapValue > 0 ? "positive" : "negative"),
      };
    })
    .filter((item) => item.shapValue !== 0)
    .sort((left, right) => Math.abs(right.shapValue) - Math.abs(left.shapValue));
}

function renderHumanFactors(id, factors, kind) {
  const element = document.getElementById(id);
  element.innerHTML = "";
  factors.filter((factor) => (kind === "positive" ? factor.shapValue > 0 : factor.shapValue < 0)).slice(0, 3).forEach((factor) => {
    const row = document.createElement("article");
    row.className = `human-factor ${kind}`;
    row.innerHTML = `<div><h4>${factor.title}</h4><p>${factor.description}</p></div>`;
    element.appendChild(row);
  });
}

function renderTechnicalFactors(id, factors) {
  const element = document.getElementById(id);
  element.innerHTML = "";
  factors.forEach((factor) => {
    const row = document.createElement("div");
    row.className = "technical-factor";
    row.innerHTML = `<span>${factor.feature}</span><span>SHAP: ${factor.shap_value > 0 ? "+" : ""}${factor.shap_value.toFixed(3)} · transformed: ${Number(factor.value).toFixed(3)}</span>`;
    element.appendChild(row);
  });
}

function renderTechnicalDetails(prediction, explanation) {
  document.getElementById("technical-model").textContent = prediction.model_name;
  document.getElementById("technical-threshold").textContent = prediction.threshold.toFixed(2);
  document.getElementById("technical-space").textContent = explanation.explanation_space.replaceAll("_", " / ");
  document.getElementById("technical-base").textContent = Number(explanation.base_value).toFixed(3);
  renderTechnicalFactors("technical-positive", explanation.top_positive_factors);
  renderTechnicalFactors("technical-negative", explanation.top_negative_factors);
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    statusEl.className = `status ${data.model_loaded ? "online" : "offline"}`;
    statusEl.innerHTML = `<span>●</span> ${data.model_loaded ? "API Connected" : "API Offline"}`;
  } catch (_) {
    statusEl.className = "status offline";
    statusEl.innerHTML = "<span>●</span> API Offline";
  }
}

async function submit(event) {
  event.preventDefault();
  clearError();
  clearAssessment();
  validateCurrencyInputs();
  updateLoanToIncomeRatio();
  markInvalidFields();
  if (!form.reportValidity()) {
    showError("Please correct the highlighted fields before continuing.");
    return;
  }
  [...form.elements].forEach((element) => element.classList.remove("invalid"));
  button.disabled = true;
  button.firstChild.textContent = "Assessing... ";
  try {
    const body = payload();
    const predictionResponse = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!predictionResponse.ok) throw new Error(predictionResponse.status === 422 ? "Please correct the highlighted fields before continuing." : "Prediction service is currently unavailable.");
    const prediction = await predictionResponse.json();
    renderResult(prediction);

    const explanationResponse = await fetch("/explain", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!explanationResponse.ok) throw new Error("Prediction succeeded, but explanation is currently unavailable.");
    const explanation = await explanationResponse.json();
    const groupedFactors = aggregateFactors(explanation.top_positive_factors, explanation.top_negative_factors, body);
    renderHumanFactors("positive-factors", groupedFactors, "positive");
    renderHumanFactors("negative-factors", groupedFactors, "negative");
    renderTechnicalDetails(prediction, explanation);
    document.getElementById("explanation").classList.remove("hidden");
  } catch (error) {
    showError(error.message || "Unable to process this application.");
  } finally {
    button.disabled = false;
    button.firstChild.textContent = "Assess Credit Risk ";
  }
}

currencyInputs.forEach((input) => {
  input.addEventListener("input", () => {
    validateCurrencyInputs();
    updateLoanToIncomeRatio();
  });
  input.addEventListener("focus", () => { unformatCurrencyInput(input); input.select(); });
  input.addEventListener("blur", () => {
    formatCurrencyInput(input);
    validateCurrencyInputs();
    updateLoanToIncomeRatio();
  });
});
form.addEventListener("submit", submit);
form.addEventListener("input", clearInvalidField);
form.addEventListener("change", clearInvalidField);
form.addEventListener("invalid", (event) => {
  clearAssessment();
  event.target.classList.add("invalid");
  showError("Please correct the highlighted fields before continuing.");
}, true);
currencyInputs.forEach(formatCurrencyInput);
updateLoanToIncomeRatio();
checkHealth();
fetch("/api/v1/model-info").then((response) => response.json()).catch(() => {});
