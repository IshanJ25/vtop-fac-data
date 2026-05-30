require("dotenv").config();

const puppeteer = require("puppeteer");
const fs = require("fs");
const path = require("path");
const bitmaps = require("./bitmaps");

// ─── Constants ────────────────────────────────────────────────────────────────

const VTOP_BASE = "https://vtop.vit.ac.in/vtop";

const URL = {
  login: `${VTOP_BASE}/login`,
  open: `${VTOP_BASE}/open/page`,
};

const SEL = {
  loginIndicators: '#vtop-header, #authorizedIDX, a[data-url="hrms/employeeSearchForStudent"]',
  loginInputs: "#username, #password, #captchaStr",
  username: "#username",
  password: "#password",
  captchaInput: "#captchaStr",
  captchaBlock: "#captchaBlock",
  captchaImg: "#captchaBlock img",
  loginForm: "#vtopLoginForm",
  stdForm: "#stdForm",
  primaryButton: ".btn-primary",
  googleCaptcha: "#recaptcha.g-recaptcha, div.g-recaptcha",
  loginError: ".alert-danger, .alert-warning, #loginBox .text-danger, .text-danger",
  loginErrorExt: ".alert-danger, .alert-warning, #loginBox .text-danger, .text-danger, .help-block, .error-message",
  facultyLink: 'a[data-url="hrms/employeeSearchForStudent"]',
  feedbackLink: 'a[href*="endfeedback"]',
};

const TIMEOUT = {
  short: 3_000,
  formInput: 6_000,
  nav: 12_000,
  loginSignal: 12_000,
  postLogin: 2_500,
  semUi: 15_000,
  captcha: 6_000,
};

const LIMITS = {
  loginAttempts: 15,
  captchaRetries: 3,
  submitDelayMs: 450,
};

const CAPTCHA_CHARSET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

// ─── Utilities ────────────────────────────────────────────────────────────────

const sleep = ms => new Promise(r => setTimeout(r, ms));

const compact = v => String(v ?? "").replace(/\s+/g, " ").trim();

function writeJsonAtomically(filePath, data) {
  const tmp = `${filePath}.tmp-${Date.now()}`;
  try {
    fs.writeFileSync(tmp, JSON.stringify(data, null, 2));
    fs.renameSync(tmp, filePath);
  } finally {
    if (fs.existsSync(tmp)) fs.unlinkSync(tmp);
  }
}

const log = {
  section: t => console.log(`\n${t}`),
  info: msg => console.log(`  • ${msg}`),
  ok: msg => console.log(`  ✓ ${msg}`),
  warn: msg => console.warn(`  ⚠️  ${msg}`),
};

// ─── Page helpers (run in browser context) ────────────────────────────────────

/** Returns true if the page indicates an active session. */
async function isLoggedIn(page) {
  return page.evaluate(({ loginIndicators, loginInputs }) => {
    if (document.querySelector(loginIndicators)) return true;
    const { href } = location;
    if (href.includes("/vtop/content") || href.includes("processLogin")) return true;
    return !document.querySelector(loginInputs)
      && href.includes("/vtop/")
      && !href.includes("/vtop/login")
      && !href.includes("/open/page");
  }, { loginIndicators: SEL.loginIndicators, loginInputs: SEL.loginInputs });
}

/**
 * Waits until the login form is visible or we're already authenticated.
 * Handles intermediate pages (stdForm, primary-button redirects).
 */
async function ensureLoginForm(page) {
  for (let attempt = 0; attempt < 4; attempt++) {
    if (await isLoggedIn(page)) return { ready: true, alreadyLoggedIn: true };
    const hasForm = await page.$(`${SEL.username}, ${SEL.password}`);
    if (hasForm) return { ready: true, alreadyLoggedIn: false };

    if (await page.$(SEL.stdForm)) {
      await Promise.all([
        page.waitForNavigation({ waitUntil: "networkidle2", timeout: TIMEOUT.nav }).catch(() => null),
        page.evaluate(s => document.querySelector(s)?.submit(), SEL.stdForm),
      ]);
    } else {
      await page
        .waitForSelector(`${SEL.primaryButton}, ${SEL.username}, ${SEL.stdForm}`, { timeout: TIMEOUT.short })
        .catch(() => null);
      await page.evaluate(s => document.querySelector(s)?.click(), SEL.primaryButton);
    }
    await sleep(800);
  }
  return { ready: false, alreadyLoggedIn: false };
}

/** Detects whether the page shows an image captcha, google captcha, or none. */
async function detectCaptchaMode(page) {
  return page.evaluate(({ captchaImg, captchaInput, googleCaptcha }) => {
    if (document.querySelector(captchaImg) && document.querySelector(captchaInput)) return "image";
    if (document.querySelector(googleCaptcha)) return "google";
    return "none";
  }, { captchaImg: SEL.captchaImg, captchaInput: SEL.captchaInput, googleCaptcha: SEL.googleCaptcha });
}

/** Refreshes the captcha image via a refresh button or the /get/new/captcha endpoint. */
async function refreshCaptchaImage(page) {
  const refreshed = await page.evaluate(async (base, { captchaBlock }) => {
    const block = document.querySelector(captchaBlock);
    if (!block) return false;

    const btn = block.querySelector("[onclick*='captcha'], [onclick*='refresh'], .fa-refresh, .fa-sync, .fa-redo");
    if (btn) { btn.click(); return true; }

    try {
      const res = await fetch(`${base}/get/new/captcha`, {
        method: "GET",
        credentials: "include",
        headers: { "X-Requested-With": "XMLHttpRequest" },
      });
      if (res.ok) { block.innerHTML = await res.text(); return true; }
    } catch { /* fall through */ }

    return false;
  }, VTOP_BASE, { captchaBlock: SEL.captchaBlock });

  if (!refreshed) return false;

  await page
    .waitForFunction(
      sel => { const img = document.querySelector(sel); return img?.complete && img.naturalWidth > 0; },
      { timeout: TIMEOUT.captcha },
      SEL.captchaImg,
    )
    .catch(() => null);

  return true;
}

/**
 * Reads the captcha image from the DOM canvas and classifies each of the
 * 6 character blocks using the provided neural-net weights and biases.
 */
async function solveCaptcha(page, weights, biases) {
  return page.evaluate((w, b, charset, imgSel) => {
    const img = document.querySelector(imgSel);
    if (!img?.complete || !img.naturalWidth) return { ok: false, reason: "image-not-ready" };

    const canvas = Object.assign(document.createElement("canvas"), { width: 200, height: 40 });
    const ctx = canvas.getContext("2d");
    if (!ctx) return { ok: false, reason: "no-canvas-ctx" };

    ctx.drawImage(img, 0, 0, 200, 40);
    const { data } = ctx.getImageData(0, 0, 200, 40);

    // Convert RGBA pixels → saturation values
    const saturation = Array.from({ length: data.length / 4 }, (_, i) => {
      const [r, g, b_] = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2]];
      const max = Math.max(r, g, b_);
      const min = Math.min(r, g, b_);
      return max === 0 ? 0 : Math.round(255 * (max - min) / max);
    });

    // Reshape flat saturation array into a 2-D grid (rows × cols)
    const grid = Array.from({ length: 40 }, (_, r) =>
      Array.from({ length: 200 }, (_, c) => saturation[r * 200 + c])
    );

    /** Runs a single softmax layer and returns the predicted character. */
    const classifyBlock = block => {
      const flat = block.flat();
      const avg = flat.reduce((s, n) => s + n, 0) / flat.length;
      const bits = flat.map(v => (v > avg ? 1 : 0));
      const logits = w[0].map((_, j) => bits.reduce((s, bit, k) => s + bit * w[k][j], 0) + b[j]);
      const exps = logits.map(Math.exp);
      const sum = exps.reduce((a, v) => a + v, 0);
      const probs = exps.map(v => v / sum);
      return charset[probs.indexOf(Math.max(...probs))];
    };

    // Each character occupies a 25 px wide column; alternate rows are offset by 5 px vertically
    const text = Array.from({ length: 6 }, (_, i) => {
      const rowStart = 7 + (i % 2) * 5 + 1;
      const rowEnd = 35 - ((i + 1) % 2) * 5;
      const colStart = 25 * (i + 1) + 2;
      const colEnd = 25 * (i + 2) + 1;
      return grid.slice(rowStart, rowEnd).map(row => row.slice(colStart, colEnd));
    }).map(classifyBlock).join("");

    return text.length === 6
      ? { ok: true, text }
      : { ok: false, reason: "bad-length" };
  }, weights, biases, CAPTCHA_CHARSET, SEL.captchaImg);
}

/** Fills the login form and fires submission via button click / VTOP globals / form.submit fallback. */
async function fillAndSubmit(page, regNo, password, captchaText) {
  await Promise.all(
    [SEL.username, SEL.password, SEL.captchaInput]
      .map(s => page.waitForSelector(s, { timeout: TIMEOUT.formInput }))
  );

  // Set values and dispatch change/input events so VTOP's validators notice
  await page.evaluate(
    ({ username, password: passField, captchaInput }, u, p, c) => {
      const fire = (el, ...events) => events.forEach(e => el?.dispatchEvent(new Event(e, { bubbles: true })));
      const set = (sel, val, ...events) => {
        const el = document.querySelector(sel);
        if (el) { el.value = val; fire(el, ...events); }
      };
      set(username, u, "input", "change", "keyup");
      set(passField, p, "input", "change");
      set(captchaInput, c, "input", "change", "keyup");
    },
    { username: SEL.username, password: SEL.password, captchaInput: SEL.captchaInput },
    regNo, password, captchaText,
  );

  // Submit: prefer explicit button → VTOP globals → form.submit
  await page.evaluate(
    (delayMs, { loginForm }) => new Promise(resolve => setTimeout(() => {
      const submitBtn = [...document.querySelectorAll('button[type="button"], button[type="submit"], input[type="submit"]')]
        .find(b => /submit|login|sign.?in/i.test(b.innerText || b.value || ""));

      if (submitBtn) { submitBtn.click(); resolve(); }
      else if (typeof callBuiltValidation === "function") { callBuiltValidation(); resolve(); }
      else if (typeof callGoogleValidation === "function") { callGoogleValidation(); resolve(); }
      else { document.querySelector(loginForm)?.submit(); resolve(); }
    }, delayMs)),
    LIMITS.submitDelayMs,
    { loginForm: SEL.loginForm },
  );

  await page.keyboard.press("Enter").catch(() => null);

  await Promise.race([
    page.waitForNavigation({ waitUntil: "networkidle2", timeout: TIMEOUT.nav }),
    page.waitForFunction(
      ({ loginIndicators, loginError }) =>
        document.querySelector(loginIndicators)
        || document.querySelector(loginError)
        || location.href.includes("/vtop/content")
        || location.href.includes("processLogin"),
      { timeout: TIMEOUT.loginSignal },
      { loginIndicators: SEL.loginIndicators, loginError: SEL.loginError },
    ),
  ]).catch(() => null);
}

/** Scrapes visible error text from the login page. */
async function getLoginError(page) {
  return page.evaluate(sel => {
    const messages = [...document.querySelectorAll(sel)]
      .map(el => el.textContent.replace(/\s+/g, " ").trim())
      .filter(t => t.length > 2);

    if (messages.length) return [...new Set(messages)].join(" | ");

    // Fallback: scan body text for common error phrases
    const match = document.body?.innerText?.replace(/\s+/g, " ").match(
      /(invalid\s+captcha[^.]*\.?|incorrect\s+captcha[^.]*\.?|wrong\s+captcha[^.]*\.?|invalid\s+credentials[^.]*\.?|incorrect\s+password[^.]*\.?|authentication\s+failed[^.]*\.?)/i
    );
    return match?.[1] ?? "";
  }, SEL.loginErrorExt);
}

const CREDENTIAL_ERROR_PATTERN =
  /invalid\s*(credentials|password|username|register)|incorrect\s*(credentials|password|username)|authentication\s*failed/i;

/** Runs a single login attempt (one page load + up to MAX_CAPTCHA captcha solves). */
async function runLoginAttempt(page, regNo, password, attempt) {
  await page.goto(URL.login, { waitUntil: "networkidle2" });
  const { ready, alreadyLoggedIn } = await ensureLoginForm(page);

  if (alreadyLoggedIn) return { loggedIn: true, message: "already authenticated" };
  if (!ready) return { loggedIn: false, message: "login form not ready" };

  const captchaMode = await detectCaptchaMode(page);

  if (captchaMode === "google") {
    await page.goto(URL.open, { waitUntil: "networkidle2" }).catch(() => null);
    await sleep(900 + 220 * attempt);
    return { loggedIn: false, message: "google captcha — rerouted" };
  }

  if (captchaMode !== "image") {
    await sleep(900 + 200 * attempt);
    return { loggedIn: false, message: `unexpected captcha mode: ${captchaMode}` };
  }

  let lastReason = "login not confirmed";

  for (let solve = 1; solve <= LIMITS.captchaRetries; solve++) {
    if (solve > 1) await refreshCaptchaImage(page);

    const captcha = await solveCaptcha(page, bitmaps.weights, bitmaps.biases);
    if (!captcha.ok) {
      lastReason = `captcha solve failed (${captcha.reason})`;
      await refreshCaptchaImage(page);
      await sleep(450);
      continue;
    }

    await fillAndSubmit(page, regNo, password, captcha.text);

    if (await isLoggedIn(page)) return { loggedIn: true, message: `logged in (solve ${solve}/${LIMITS.captchaRetries})` };

    const error = await getLoginError(page);
    if (CREDENTIAL_ERROR_PATTERN.test(error)) throw new Error(`Credentials rejected: ${error}`);

    lastReason = error ? compact(error).slice(0, 100) : `solve ${solve}/${LIMITS.captchaRetries} failed`;
    await sleep(850);
  }

  await sleep(750 + 220 * attempt);
  return { loggedIn: false, message: lastReason };
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function runScraper() {
  const regNo = process.env.VTOP_REGNO?.trim();
  const password = process.env.VTOP_PASSWORD?.trim();
  if (!regNo || !password) throw new Error("Missing VTOP_REGNO or VTOP_PASSWORD in .env");

  log.section("🎯 Target: Scraping Faculty Data");

  const browser = await puppeteer.launch({
    headless: true,
    executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || undefined,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });
  const page = await browser.newPage();
  page.on("dialog", d => d.accept());

  try {
    // ── Login ──────────────────────────────────────────────────────────────
    log.section("🔐 Login");
    let loggedIn = false;

    for (let i = 1; i <= LIMITS.loginAttempts && !loggedIn; i++) {
      const result = await runLoginAttempt(page, regNo, password, i);
      if (result.loggedIn) {
        log.ok(`[${i}/${LIMITS.loginAttempts}] ${result.message}`);
        loggedIn = true;
      } else {
        log.warn(`[${i}/${LIMITS.loginAttempts}] ${result.message}`);
      }
    }

    if (!loggedIn) throw new Error("Failed to log in after all attempts.");

    await sleep(TIMEOUT.postLogin);

    // Dismiss any post-login modals/overlays
    await page.evaluate(() => {
      document.querySelector(".bootbox-accept, .modal-footer .btn-primary")?.click();
      document.querySelector(".sweet-alert .confirm")?.click();
      document.querySelectorAll(".modal-backdrop, .sweet-overlay").forEach(el => el.remove());
      document.body.classList.remove("modal-open", "stop-scrolling");
    });

    // Guard against mandatory feedback blocking the menu
    await page
      .waitForSelector(`${SEL.facultyLink}, ${SEL.feedbackLink}`, { timeout: TIMEOUT.semUi })
      .catch(() => { throw new Error("Could not find faculty or feedback link."); });

    if (await page.$(SEL.feedbackLink)) {
      log.warn("Mandatory feedback form detected — menu is blocked.");
      log.info("Complete feedback at https://web.vit.ac.in/endfeedback, then re-run.");
      return;
    }

    // ── Faculty search ─────────────────────────────────────────────────────
    log.section("👨‍🏫 Navigating to Faculty Search");
    await page.evaluate(s => document.querySelector(s)?.click(), SEL.facultyLink);
    await page.waitForSelector("#searchEmployee", { timeout: TIMEOUT.semUi });

    log.section("🔍 Fetching Faculty List");
    log.info("Querying with three spaces to fetch the complete list…");

    // Three spaces satisfies minlength=3 and returns all results
    await page.evaluate(() => {
      document.getElementById("searchEmployee").value = "   ";
      getEmployeeInfo();
    });

    await page.waitForResponse(
      res => res.url().includes("EmployeeSearchForStudent") && res.status() === 200,
      { timeout: TIMEOUT.nav },
    );
    await sleep(2500); // allow large table to finish rendering

    // ── Extract data ───────────────────────────────────────────────────────
    log.section("📥 Extracting Names and IDs");

    const facultyData = await page.evaluate(() => {
      const seen = new Map();

      for (const btn of document.querySelectorAll('button[onclick^="getEmployeeIdNo"]')) {
        const match = btn.getAttribute("onclick").match(/getEmployeeIdNo\(['"]([^'"]+)['"]\)/);
        if (!match) continue;

        const emp_id = match[1];
        if (seen.has(emp_id)) continue; // deduplicate

        const cells = btn.closest("tr")?.querySelectorAll("td");
        let name = cells?.length ? (btn.innerText.trim() || cells[0].innerText.trim()) : "Unknown";
        name = name.substring(name.indexOf(".") + 1).trim();

        seen.set(emp_id, { emp_id, name });
      }

      return [...seen.values()];
    });

    log.ok(`Extracted ${facultyData.length} unique faculty records.`);

    const outPath = path.join(__dirname, "data", "vellore.json");
    fs.mkdirSync(path.dirname(outPath), { recursive: true });

    const ist = new Date(Date.now() + 5.5 * 60 * 60 * 1000); // UTC + 5:30
    const pad = n => String(n).padStart(2, "0");
    const generatedAt = `${ist.getUTCFullYear()}-${pad(ist.getUTCMonth() + 1)}-${pad(ist.getUTCDate())}::${pad(ist.getUTCHours())}:${pad(ist.getUTCMinutes())}:${pad(ist.getUTCSeconds())}`;

    writeJsonAtomically(outPath, { generatedAt, data: facultyData });
    log.ok(`Saved to ${outPath}`);

  } finally {
    await browser.close();
  }
}

runScraper().catch(err => console.error("❌ Fatal:", err));