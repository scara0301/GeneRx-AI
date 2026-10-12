# 06 — Frontend walkthrough (`index.html`, `app.js`, `style.css`)

## 1. Why vanilla JS + a single page?

* The UI is one screen with two modes; there is no routing, auth, or complex shared state.
* **No build tooling** (no npm, webpack, React) → the backend container serves the files directly, the project starts with one command, and a reviewer can read the entire client in ~700 lines.
* Libraries are loaded from CDNs: **Lucide** (icons), **Chart.js 4** (chart), **Inter** (Google Fonts).
  Trade-off: needs internet in the browser; `lucide@latest` is unpinned (supply-chain/version-drift risk — pin it in production).

## 2. `index.html` structure

| Region | ID(s) | Purpose |
|--------|-------|---------|
| Top bar | `btnDoctor`, `btnPatient`, `apiStatus`, `apiStatusText` | brand, Clinician/Patient toggle, live API status |
| Sidebar form | `patientForm`, `patientName/Age/Sex/Weight`, `conditionsGrid`, `labEgfr/Alt/Hba1c/Sbp/Ldl/Inr`, `currentMedsGrid`, `allergiesGrid`, `clearForm` | the clinician's patient profile |
| Drug card | `drugSelectionCard`, `drugGrid`, `runAssessment` | pick drugs, run |
| Results card | `resultsCard`, `resultsMeta`, `summaryTable`, `ddiSection/ddiList`, `detailCards`, `mlSection/mlChart` | output |
| Wizard | `patientWizard`, `wizardProgress`, `wizardContent`, `wizardNav` | patient mode |
| Footer | `.disclaimer` | medical disclaimer |

Accessibility/UX details: `<meta viewport>`, theme-color, inline SVG favicon (so no extra file), `disabled` Run button until a drug is chosen, labels bound to inputs (`for=`).

## 3. State (module-level variables in `app.js`)

| Variable | Type | Meaning |
|----------|------|---------|
| `API_BASE` | string | `http://localhost:8000` on localhost; otherwise `window.location.pathname` with trailing slash stripped (works under HF's `/embed/user/space` path) |
| `selectedDrugs` | `Set<string>` | drugs ticked in the clinician grid (Set → O(1) toggle, no duplicates) |
| `drugCatalog` | array | `[{name, category, description}]` from `/api/drugs` (or a hard-coded fallback) |
| `patientWizardStep` | int 1–6 | current wizard step |
| `patientData` | object | wizard answers accumulated across steps |

## 4. Function-by-function (what / why)

### Boot
* **`DOMContentLoaded`** → `lucide.createIcons()` (turns `<i data-lucide>` into SVGs), `checkApiStatus()`, `loadDrugCatalog()`, `setupEventListeners()`.
* **`checkApiStatus`** — `fetch(/api/health)` with `AbortSignal.timeout(5000)` so the UI doesn't hang. Green dot + "ML Model Active"/"Rules Engine Only", or red + "API Offline". *Why:* tells the user immediately whether results will include ML.
* **`loadDrugCatalog`** — fetches `/api/drugs`; on failure uses a built-in list of the same 10 drugs so the UI still renders (progressive degradation). Then `renderDrugGrid` and `renderCurrentMedsCheckboxes`.
* **`setupEventListeners`** — click handlers for mode buttons, Run, Clear.

### Mode switching
* **`switchMode(mode)`** — toggles `.active` on the buttons and `hidden` on sidebar / drug card / results / wizard. Entering patient mode resets to wizard step 1.
  *Note:* `currentMode` is assigned without declaration (implicit global; harmless in sloppy-mode scripts, would throw in strict mode/ES modules).

### Clinician flow
* **`renderDrugGrid` / `toggleDrug` / `updateRunButton`** — re-render the whole grid on each click (simple; fine for 10 items). Selected cards get the `selected` class. Run is `disabled` when the set is empty.
* **`renderCurrentMedsCheckboxes`** — same catalog, used for "current medications" (feeds interaction checks and `num_concomitant_drugs`).
* **`collectPatientProfile`** — reads inputs and checked boxes into the exact JSON shape of the backend's `PatientProfile`. Details:
  * `parseInt/parseFloat(...) || default` — blank/NaN falls back to a normal value.
  * `diastolic_bp = round(0.65 × SBP)` — the form doesn't ask for diastolic; no rule uses it.
  * **Bug:** `ast` is read from `labAlt` (copy-paste); there's no AST input.
* **`runAssessment`** — shows an "Analyzing…" state, POSTs `{patient, drugs}`, calls `renderResults`, and in `finally` restores the button. A failed request replaces the results card with a "Connection Error" message. *Why `try/catch/finally`:* the button must never stay stuck on "Analyzing".
* **`renderResults`** — shows the card, writes the header ("N drugs assessed for X"), calls the four sub-renderers, scrolls to results.
* **`riskClass`** — Suitable→`safe`, Caution→`caution`, Avoid→`avoid`, Contraindicated→`critical` (CSS colour classes: green/amber/orange-red/red).
* **`renderSummaryTable`** — one row per drug: badge, risk level, ML confidence (or "N/A"), and the first reason as "Key concern". Because allergy alerts are inserted at index 0, they surface as the key concern.
* **`renderInteractions`** — shows the DDI section only if there are interactions.
* **`renderDetailCards` / `toggleDetail`** — accordion; first card expanded; sections shown only if non-empty (`a.dose_notes?.length ? … : ''`, using optional chaining). Side-effects are shown as tags with the parenthetical stripped (`"Lactic Acidosis (rare)"` → `"Lactic Acidosis"`) — note this hides the "rare" qualifier.
* **`renderMLChart`** — builds a **stacked bar chart**: one bar per drug, stacks = the four class probabilities (%), y-axis max 100. Destroys any previous chart (`window._mlChart.destroy()`) to avoid Chart.js canvas-reuse errors. Hidden if no drug has an ML result.

### Patient wizard
* **`WIZARD_STEPS`** — titles/subtitles for 6 steps: About You → Health Conditions → Health Numbers → Current Medications → Check Medications → Your Results.
* **`PLAIN_CONDITIONS`** — maps technical keys sent to the API to lay language ("CKD" → "Kidney problems").
* **`renderPatientWizard`** — draws the progress dots, the step's form (via a `switch`), and nav buttons; at step 6 calls `runPatientAssessment`.
* **`wizardSaveStep` / `wizardNext` / `wizardBack` / `wizardRestart`** — persist inputs to `patientData` before moving (so Back doesn't lose answers); Next on step 5 validates that ≥ 1 drug is chosen.
* **`runPatientAssessment`** — builds the patient object; things the wizard *doesn't ask*, it fixes: ALT/AST 25, INR 1.0, **no allergies** (so patient mode can never trigger the allergy rules — a limitation worth admitting), diastolic from SBP.
* **`renderPatientResults`** — maps suitability to friendly labels/descriptions and shows only the *first* reason; interactions are phrased as "to discuss with your doctor". *Why:* avoid overwhelming a layperson and avoid sounding like medical advice.

### Utility
* **`resetForm`** — `form.reset()` then re-assigns the default values explicitly (redundant — `reset()` already restores the HTML `value` attributes — but harmless and self-documenting), clears `selectedDrugs`, re-renders the grid, hides results.

## 5. `style.css` — design system

* **CSS custom properties** in `:root` (colour tokens: backgrounds, text, accent indigo `#6366f1`, risk colours `--safe/--caution/--avoid/--critical`) → one place to retheme.
* **Dark glassmorphism**: semi-transparent cards, blur, subtle borders; Inter font.
* Component sections: topbar, sidebar/form, drug grid cards, primary button (+loading state), results table, risk badges & dots, DDI alert, accordion detail cards, ML section, wizard (progress dots, nav), patient result cards, disclaimer footer, animations (`@keyframes pulse-dot`, `slideInLeft`, `spin`, `fadeIn`, `skeleton-loading`), and responsive rules at `max-width: 900px` and `600px`.
* Tokens actually defined: `--bg-primary #0a0e17`, `--bg-card` (3 % white), `--accent #6366f1`, `--safe #22c55e`, `--caution #f59e0b`, `--avoid #ef4444`, `--critical #dc2626`, radii 6/10/14 px, `--blur: blur(16px)` used by `backdrop-filter` (the "glass" effect).
* `.hidden { display:none }` is the visibility mechanism the JS toggles.

## 6. Front-end security & quality notes (interview fodder)

| Topic | State | Better |
|-------|-------|--------|
| XSS | Uses `innerHTML` with template strings. Data comes from our own API (drug names, rule messages) and from the user's *name* field only inside `patient_name` → `resultsMeta` (this one **is** user-supplied and unescaped). | escape or use `textContent` for user input |
| Inline `onclick` handlers | used for dynamic cards | `addEventListener` / event delegation |
| Global state | module-level `let`s | a small store or framework if it grows |
| Testing | none | Playwright e2e for both modes |
| Pinned CDN versions | `lucide@latest` | pin + SRI hashes |
| a11y | labels yes; no ARIA live region for results; colour-only risk coding is backed by text labels | add `aria-live`, focus management |
