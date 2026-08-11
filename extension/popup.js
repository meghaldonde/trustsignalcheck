// Uses CONFIG.API_URL from config.js

document.getElementById("scanBtn").addEventListener("click", async () => {
  const btn = document.getElementById("scanBtn");
  const resultDiv = document.getElementById("result");

  btn.disabled = true;
  btn.textContent = "Scanning...";
  resultDiv.innerHTML = "";

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    const [{ result: pageData }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content.js"],
    });

    const response = await fetch(CONFIG.API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        url: pageData.url,
        text_snippet: pageData.text,
      }),
    });

    // Handle rate limit
    if (response.status === 429) {
      const errorData = await response.json();
      displayRateLimitError(errorData.detail);
      return;
    }

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    displayResult(data);
  } catch (error) {
    resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Scan Page";
  }
});

function displayRateLimitError(detail) {
  document.getElementById("result").innerHTML = `
    <div class="rate-limit">
      <div class="rate-limit-icon">⏱️</div>
      <div class="rate-limit-title">Daily Limit Reached</div>
      <div class="rate-limit-message">${detail.message}</div>
      <div class="rate-limit-hint">Come back tomorrow for more scans!</div>
    </div>
  `;
}

function displayResult(data) {
  const score = data.signal_trust_score;
  let scoreClass = "score-medium";
  if (score >= 70) scoreClass = "score-high";
  else if (score < 40) scoreClass = "score-low";

  const signalsList = data.ai_analysis.key_signals
    .map((s) => `<li>${s}</li>`)
    .join("");

  // Format threat warning if present
  let threatWarning = "";
  if (data.domain_signal_score.threat_type) {
    const threatLabels = {
      "MALWARE": "🦠 Malware Detected",
      "SOCIAL_ENGINEERING": "🎣 Phishing/Social Engineering",
      "UNWANTED_SOFTWARE": "⚠️ Unwanted Software",
      "POTENTIALLY_HARMFUL_APPLICATION": "⚠️ Potentially Harmful"
    };
    const label = threatLabels[data.domain_signal_score.threat_type] || `⚠️ ${data.domain_signal_score.threat_type}`;
    threatWarning = `<div style="background:#fce8e6;color:#c5221f;padding:8px;border-radius:4px;margin-bottom:8px;font-weight:bold;">${label}</div>`;
  }

  document.getElementById("result").innerHTML = `
    <div class="score-container">
      <div class="trust-score ${scoreClass}">${score}</div>
      <div>Signal Trust Score</div>
    </div>
    ${threatWarning}
    <div class="details">
      <p><strong>Domain Signal Score:</strong> ${data.domain_signal_score.reputation_score}/100 (${data.domain_signal_score.source})</p>
      <p><strong>AI Probability:</strong> ${data.ai_analysis.ai_probability_score}%</p>
      <p><strong>Analysis:</strong> ${data.ai_analysis.reasoning_flag}</p>
      <p><strong>Signals:</strong></p>
      <ul class="signals">${signalsList}</ul>
    </div>
  `;
}
