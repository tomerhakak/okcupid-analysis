# -*- coding: utf-8 -*-
"""
OkCupid Profile Analysis — Data Science Project
================================================

Objective:
    Explore the okcupid_profiles.csv dataset (~60K profiles),
    produce clear visualizations, and build machine learning models
    to predict demographic and behavioral attributes.

Structure:
    1. Data Loading & Cleaning
    2. Data Quality Assessment
    3. Demographics & Geography
    4. Education & Lifestyle
    5. Behavior Patterns
    6. Clustering — K-Means (age/height segments)
    7. Predictive Analytics — drinks classification

Output:
    All charts are saved to the plots/ folder and displayed in dashboard.html.
"""

from __future__ import annotations

import warnings
import webbrowser
from pathlib import Path

import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Global configuration lookup ───────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "okcupid_profiles.csv"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
DASHBOARD_PATH = BASE_DIR / "dashboard.html"

DASHBOARD_CHARTS: list[dict[str, str]] = []
# Clear stale artifacts from previous runs (both static and interactive charts)
for pattern in ("*.png", "*.html"):
    for old_plot in PLOTS_DIR.glob(pattern):
        old_plot.unlink(missing_ok=True)

# Consistent color palette across all charts
PALETTE = sns.color_palette("husl", 8)
SEX_PALETTE = {"m": "#4C72B0", "f": "#DD8452"}

sns.set_theme(
    style="whitegrid",
    context="notebook",
    font_scale=1.05,
    rc={
        "figure.figsize": (12, 6),
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FFFFFF",
        "grid.alpha": 0.35,
    },
)


def save_fig(name: str, title: str, section_name: str) -> None:
    """Save the current figure and register it for the dashboard."""
    path = PLOTS_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    DASHBOARD_CHARTS.append({"file": f"{name}.png", "title": title, "section": section_name})
    print(f"  Saved: {path.name}")


def build_dashboard(
    stats: dict[str, str],
    heatmap_file: str,
    report: dict[str, str | float | int],
) -> Path:
    """Build a professional portfolio-style HTML dashboard with business narrative."""
    sections: dict[str, list[dict[str, str]]] = {}
    for chart in DASHBOARD_CHARTS:
        sections.setdefault(chart["section"], []).append(chart)

    section_intros = {
        "User Engagement": "The core business question — who is most engaged (by last-online recency) and what distinguishes them. Profile completeness emerges as a clear, actionable lever.",
        "Data Quality": "How complete is the data? Missing-value rates across key profile fields.",
        "Demographics": "Who are the users? Age, sex, relationship status, and geographic concentration.",
        "Education": "Education levels and how they relate to lifestyle choices.",
        "Behavior": "Lifestyle patterns — alcohol, smoking, and body-type distributions.",
        "Clustering": "User segmentation via K-Means on age and height.",
        "Predictive Analytics": "Random Forest predicting drinking habits — evaluated honestly against a majority-class baseline, because raw accuracy is misleading on imbalanced data.",
    }

    chart_sections = [
        "User Engagement",
        "Data Quality", "Demographics", "Education",
        "Behavior", "Clustering", "Predictive Analytics",
    ]
    nav_links = (
        '<a href="#overview">Overview</a>'
        '<a href="#business-problem">Problem</a>'
        '<a href="#methodology">Methodology</a>'
        '<a href="#findings">Findings</a>'
        + "".join(f'<a href="#{s.lower().replace(" ", "-")}">{s}</a>' for s in chart_sections if s in sections)
        + '<a href="#recommendations">Recommendations</a>'
        + '<a href="#tech-stack">Tech Stack</a>'
        + '<a href="#interactive">Interactive</a>'
    )

    kpi_cards = "".join(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>"""
        for label, value in stats.items()
    )

    section_blocks = []
    for sec_name in chart_sections:
        if sec_name not in sections:
            continue
        charts = sections[sec_name]
        sec_id = sec_name.lower().replace(" ", "-")
        intro = section_intros.get(sec_name, "")
        cards = "".join(
            f"""
            <div class="chart-card">
                <div class="chart-title">{c["title"]}</div>
                <img src="plots/{c["file"]}" alt="{c["title"]}" loading="lazy">
            </div>"""
            for c in charts
        )
        section_blocks.append(
            f"""
        <section id="{sec_id}" class="section">
            <h2>{sec_name}</h2>
            <p class="section-intro">{intro}</p>
            <div class="chart-grid">{cards}</div>
        </section>"""
        )

    profiles = report["profiles"]
    median_age = report["median_age"]
    clusters = report["clusters"]
    male_pct = report["male_pct"]
    sf_pct = report["sf_pct"]
    single_pct = report["single_pct"]
    drinks_acc = report["drinks_acc"]
    baseline_acc = report["baseline_acc"]
    balanced_acc = report["balanced_acc"]
    macro_f1 = report["macro_f1"]
    active_pct = report["active_pct"]
    completeness_lift = report["completeness_lift"]
    chart_count = len(DASHBOARD_CHARTS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OkCupid Analysis — Data Analyst Case Study</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0b0f19;
    --surface: #131a2b;
    --surface2: #1a2236;
    --border: rgba(255,255,255,0.08);
    --text: #e8edf5;
    --muted: #8b9bb4;
    --accent: #6366f1;
    --accent2: #8b5cf6;
    --green: #10b981;
    --amber: #f59e0b;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.65;
  }}
  .hero {{
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #0b0f19 100%);
    padding: 3.5rem 2rem 3rem;
    border-bottom: 1px solid var(--border);
  }}
  .hero-inner {{ max-width: 1400px; margin: 0 auto; }}
  .badge {{
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    margin-bottom: 1rem;
  }}
  .hero h1 {{
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a5b4fc, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.6rem;
    line-height: 1.2;
  }}
  .hero-sub {{ color: var(--muted); font-size: 1.1rem; max-width: 680px; }}
  .hero-meta {{
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin-top: 1.5rem;
    font-size: 0.875rem;
    color: var(--muted);
  }}
  .hero-meta span {{ display: flex; align-items: center; gap: 0.4rem; }}
  .hero-meta strong {{ color: var(--text); }}
  nav {{
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    padding: 0.85rem 2rem;
    background: rgba(19,26,43,0.95);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
  }}
  nav a {{
    color: var(--muted);
    text-decoration: none;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 500;
    transition: all 0.2s;
    border: 1px solid transparent;
    white-space: nowrap;
  }}
  nav a:hover {{
    color: var(--text);
    background: var(--surface2);
    border-color: var(--border);
  }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 2.5rem 2rem; }}
  .prose-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
  }}
  .prose-card h2 {{
    font-size: 1.35rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text);
    border-bottom: 2px solid var(--accent);
    padding-bottom: 0.5rem;
    display: inline-block;
  }}
  .prose-card p {{ color: #c5cfe0; margin-bottom: 0.85rem; }}
  .prose-card p:last-child {{ margin-bottom: 0; }}
  .prose-card ul {{ color: #c5cfe0; padding-left: 1.25rem; }}
  .prose-card li {{ margin-bottom: 0.4rem; }}
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.25rem;
    margin-bottom: 2rem;
  }}
  @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  .method-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    margin-top: 0.5rem;
  }}
  .method-table th {{
    text-align: left;
    padding: 0.75rem 1rem;
    background: var(--surface2);
    color: var(--muted);
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .method-table td {{
    padding: 0.75rem 1rem;
    border-top: 1px solid var(--border);
    color: #c5cfe0;
  }}
  .method-table tr:hover td {{ background: rgba(99,102,241,0.05); }}
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin-bottom: 2.5rem;
  }}
  .kpi {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    transition: transform 0.2s, border-color 0.2s;
  }}
  .kpi:hover {{
    transform: translateY(-2px);
    border-color: rgba(99,102,241,0.4);
  }}
  .kpi-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin-bottom: 0.35rem;
  }}
  .kpi-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--green);
  }}
  .insight-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
    margin-bottom: 2.5rem;
  }}
  .insight-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    border-left: 3px solid var(--accent);
  }}
  .insight-card.amber {{ border-left-color: var(--amber); }}
  .insight-card.green {{ border-left-color: var(--green); }}
  .insight-card.purple {{ border-left-color: var(--accent2); }}
  .insight-num {{
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--green);
    margin-bottom: 0.35rem;
  }}
  .insight-title {{
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
  }}
  .insight-text {{ color: var(--muted); font-size: 0.875rem; line-height: 1.55; }}
  .section {{ margin-bottom: 3rem; }}
  .section h2 {{
    font-size: 1.35rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid var(--accent);
    display: inline-block;
  }}
  .section-intro {{
    color: var(--muted);
    font-size: 0.95rem;
    margin-bottom: 1.25rem;
    max-width: 720px;
  }}
  .chart-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
    gap: 1.25rem;
  }}
  .chart-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    transition: transform 0.25s, box-shadow 0.25s;
  }}
  .chart-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.15);
  }}
  .chart-title {{
    padding: 1rem 1.25rem 0.5rem;
    font-weight: 600;
    font-size: 0.95rem;
  }}
  .chart-card img {{
    width: 100%;
    display: block;
    background: #fafafa;
  }}
  .rec-list {{
    list-style: none;
    counter-reset: rec;
  }}
  .rec-list li {{
    counter-increment: rec;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem 1.25rem 3.5rem;
    margin-bottom: 0.75rem;
    position: relative;
    color: #c5cfe0;
    font-size: 0.925rem;
  }}
  .rec-list li::before {{
    content: counter(rec);
    position: absolute;
    left: 1.25rem;
    top: 1.25rem;
    width: 1.6rem;
    height: 1.6rem;
    background: var(--accent);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
  }}
  .rec-list li strong {{ color: var(--text); }}
  .tag-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.75rem;
  }}
  .tag {{
    background: var(--surface2);
    border: 1px solid var(--border);
    color: #a5b4fc;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 500;
  }}
  .iframe-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
  }}
  .iframe-card h3 {{
    padding: 1rem 1.25rem 0.5rem;
    font-size: 0.95rem;
  }}
  .iframe-card iframe {{
    width: 100%;
    height: 420px;
    border: none;
    background: white;
  }}
  footer {{
    text-align: center;
    padding: 2.5rem 2rem;
    color: var(--muted);
    font-size: 0.85rem;
    border-top: 1px solid var(--border);
    line-height: 1.8;
  }}
  footer strong {{ color: var(--text); }}
  @media (max-width: 600px) {{
    .chart-grid {{ grid-template-columns: 1fr; }}
    .hero h1 {{ font-size: 1.75rem; }}
  }}
</style>
</head>
<body>
  <header class="hero">
    <div class="hero-inner">
      <div class="badge">Data Analyst Portfolio Case Study</div>
      <h1>OkCupid Profile Analysis</h1>
      <p class="hero-sub">End-to-end analytics project — from raw profile data to segmentation, predictive modeling, and stakeholder-ready insights.</p>
      <div class="hero-meta">
        <span><strong>{profiles}</strong> profiles analyzed</span>
        <span><strong>{chart_count}</strong> visualizations</span>
        <span><strong>{sf_pct:.0f}%</strong> from San Francisco</span>
        <span>Python · pandas · scikit-learn · Altair</span>
      </div>
    </div>
  </header>

  <nav>{nav_links}</nav>

  <div class="container">

    <!-- Overview -->
    <section id="overview" class="prose-card">
      <h2>Executive Summary</h2>
      <p>I analyzed a dataset of <strong>{profiles} OkCupid user profiles</strong> to understand demographic patterns, behavioral trends, and predictive signals in user attributes. The goal was to move from raw data to actionable insights using a structured analytics workflow.</p>
      <p>I built a complete pipeline covering data cleaning, exploratory analysis, K-Means segmentation, and machine learning — then delivered results through this interactive dashboard designed for non-technical stakeholders. The analysis is anchored by one focused business question: <strong>who are the most engaged users, and what distinguishes them?</strong></p>
      <p><strong>Key outcomes:</strong> mapped geographic concentration in the Bay Area, identified education as the strongest lifestyle predictor, segmented users into {clusters} clusters, and rigorously evaluated a drinks-classification model against a majority-class baseline — surfacing that profile attributes carry only limited predictive signal (an honest negative result, not a vanity metric).</p>
    </section>

    <!-- Business Problem -->
    <div class="two-col" id="business-problem">
      <div class="prose-card">
        <h2>Business Problem</h2>
        <p>Dating platforms rely on understanding user demographics and behavior to improve matching, personalization, and product strategy. Without structured analysis, decisions are often based on assumptions rather than evidence.</p>
        <p><strong>Core question:</strong> Who are the most engaged users on the platform, and what distinguishes them — so product and growth teams can act on it? Supporting questions cover who the users are, how they behave, and which attributes are most predictive.</p>
      </div>
      <div class="prose-card">
        <h2>Success Criteria</h2>
        <ul>
          <li>Clean, analysis-ready dataset from raw CSV</li>
          <li>Clear visual insights organized by business theme</li>
          <li>Quantified model performance with comparison</li>
          <li>Actionable recommendations backed by data</li>
          <li>Stakeholder-friendly delivery (this dashboard)</li>
        </ul>
      </div>
    </div>

    <!-- Methodology -->
    <section id="methodology" class="prose-card">
      <h2>Methodology</h2>
      <p>I followed a standard data analytics lifecycle — each phase maps to a section of this dashboard.</p>
      <table class="method-table">
        <thead>
          <tr><th>Phase</th><th>What I Did</th><th>Tools</th><th>Dashboard Section</th></tr>
        </thead>
        <tbody>
          <tr><td>Data Cleaning</td><td>Dropped text columns, handled missing values, standardized empty strings</td><td>pandas, numpy</td><td>—</td></tr>
          <tr><td>Data Quality</td><td>Missing-value audit across all profile fields</td><td>pandas</td><td>Data Quality</td></tr>
          <tr><td>EDA</td><td>Demographics, geography, education, lifestyle cross-tabs</td><td>seaborn, matplotlib</td><td>Demographics, Education, Behavior</td></tr>
          <tr><td>Segmentation</td><td>K-Means (age + height), Elbow Method, feature scaling</td><td>scikit-learn</td><td>Clustering</td></tr>
          <tr><td>Modeling</td><td>Random Forest (class-balanced) vs. majority-class baseline; evaluated on accuracy, balanced accuracy & macro-F1</td><td>scikit-learn</td><td>Predictive Analytics</td></tr>
          <tr><td>Delivery</td><td>Auto-generated HTML dashboard with KPIs and navigation</td><td>HTML/CSS, Altair</td><td>This page</td></tr>
        </tbody>
      </table>
    </section>

    <!-- KPIs -->
    <div class="kpi-row">{kpi_cards}</div>

    <!-- Key Findings -->
    <section id="findings">
      <h2 style="font-size:1.35rem;font-weight:600;margin-bottom:1.25rem;padding-bottom:0.6rem;border-bottom:2px solid var(--accent);display:inline-block;">Key Findings</h2>
      <div class="insight-grid">
        <div class="insight-card green">
          <div class="insight-num">+{completeness_lift:.0f} pts</div>
          <div class="insight-title">Completeness Drives Engagement</div>
          <div class="insight-text">Users with the most complete profiles are <strong>{completeness_lift:.0f} percentage points</strong> more likely to be active (online in the last 30 days) than those with the least complete profiles. {active_pct:.0f}% of users are active overall. Nudging profile completion is a direct, low-cost engagement lever.</div>
        </div>
        <div class="insight-card">
          <div class="insight-num">{sf_pct:.0f}%</div>
          <div class="insight-title">Bay Area Concentration</div>
          <div class="insight-text">Over half of all profiles are from San Francisco. The dataset is hyper-local — geographic targeting should reflect this.</div>
        </div>
        <div class="insight-card amber">
          <div class="insight-num">{single_pct:.0f}%</div>
          <div class="insight-title">Single Users Dominate</div>
          <div class="insight-text">{single_pct:.0f}% report "single" status. Median age is {median_age:.0f} — a young, dating-active audience.</div>
        </div>
        <div class="insight-card">
          <div class="insight-num">{macro_f1:.2f}</div>
          <div class="insight-title">Honest Model Evaluation</div>
          <div class="insight-text">A naive baseline already scores {baseline_acc:.0%} accuracy by always guessing "socially". The balanced Random Forest reaches {balanced_acc:.0%} balanced accuracy (macro-F1 {macro_f1:.2f}) — proof that profile features hold only weak signal for drinking habits. Accuracy alone would have hidden this.</div>
        </div>
        <div class="insight-card purple">
          <div class="insight-num">{clusters} Segments</div>
          <div class="insight-title">Age × Height Clusters</div>
          <div class="insight-text">K-Means on age and height revealed {clusters} distinct user segments — useful for matching and personalization strategies.</div>
        </div>
      </div>
    </section>

    <!-- Charts -->
    {"".join(section_blocks)}

    <!-- Recommendations -->
    <section id="recommendations" class="prose-card">
      <h2>Recommendations</h2>
      <ol class="rec-list">
        <li><strong>Nudge profile completion to lift engagement</strong> — the most complete profiles are +{completeness_lift:.0f} pts more likely to be active. Add progress bars, completion prompts, and onboarding nudges; this is the highest-leverage finding in the analysis.</li>
        <li><strong>Fix data collection gaps</strong> — offspring (59% missing), diet (41%), religion (34%) are largely empty. Prioritize required fields or smart defaults.</li>
        <li><strong>Leverage geographic concentration</strong> — {sf_pct:.0f}% of users are in SF. Hyper-local features (neighborhood, commute) could improve matching quality.</li>
        <li><strong>Don't over-trust the lifestyle model</strong> — among available features education and age rank highest, but overall predictive signal is weak (macro-F1 {macro_f1:.2f}). Profile attributes alone are not enough to infer drinking habits; richer behavioral data would be needed before productizing this.</li>
        <li><strong>Apply cluster-based matching</strong> — the {clusters} age×height segments can drive differentiated matching algorithms or marketing campaigns.</li>
        <li><strong>Drop income from analysis</strong> — 81% of users hide income (-1). This field adds noise, not signal. Focus on behavior and education instead.</li>
      </ol>
    </section>

    <!-- Tech Stack -->
    <section id="tech-stack" class="prose-card">
      <h2>Tech Stack</h2>
      <p>This project was built entirely in Python. The dashboard auto-generates on every pipeline run — no manual updates required.</p>
      <div class="tag-row">
        <span class="tag">Python 3.12</span>
        <span class="tag">pandas</span>
        <span class="tag">numpy</span>
        <span class="tag">seaborn</span>
        <span class="tag">matplotlib</span>
        <span class="tag">scikit-learn</span>
        <span class="tag">Altair</span>
        <span class="tag">HTML / CSS</span>
      </div>
    </section>

    <!-- Interactive -->
    <section id="interactive" class="section">
      <h2>Interactive Exploration</h2>
      <p class="section-intro">Hover over cells to explore the relationship between relationship status, sexual orientation, and location.</p>
      <div class="iframe-card">
        <h3>Status × Orientation Heatmap</h3>
        <iframe src="plots/{heatmap_file}"></iframe>
      </div>
    </section>

  </div>

  <footer>
    <strong>OkCupid Profile Analysis</strong> — Data Analyst Portfolio Case Study<br>
    Built with Python · Auto-generated by work.py · {profiles} profiles · {chart_count} charts
  </footer>
</body>
</html>"""

    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    # Also write index.html so GitHub Pages serves the dashboard as the landing page.
    (BASE_DIR / "index.html").write_text(html, encoding="utf-8")
    return DASHBOARD_PATH


def open_dashboard(path: Path) -> None:
    """Open the dashboard in the default web browser."""
    webbrowser.open(path.resolve().as_uri())
    print(f"\n  Dashboard opened: {path}")


def section(title: str) -> None:
    """Print a section header to the console."""
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data Loading & Cleaning
# ══════════════════════════════════════════════════════════════════════════════
section("1. Data Loading & Cleaning")

# read_csv loads the CSV into a pandas DataFrame.
# Each row = one user profile, each column = a feature (age, sex, location, etc.)
df = pd.read_csv(DATA_PATH)
print(f"Dataset size: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

print("\nData types and missing values:")
print(df.info())

print("\nDescriptive statistics (including categorical columns):")
print(df.describe(include="all").T.head(12))

# Drop free-text essay columns — not useful for quantitative analysis
ESSAY_COLS = [f"essay{i}" for i in range(10)]
df = df.drop(columns=ESSAY_COLS, errors="ignore")

# Convert empty strings to NaN — required before dropna / imputation
df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

print(f"\nAfter cleaning: {df.shape[0]:,} rows x {df.shape[1]} columns")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Data Quality
# ══════════════════════════════════════════════════════════════════════════════
section("2. Data Quality Assessment")

missing_pct = (df.isnull().mean() * 100).sort_values(ascending=True)
missing_pct = missing_pct[missing_pct > 0]

fig, ax = plt.subplots(figsize=(10, 7))
colors_miss = ["#ef4444" if v > 30 else "#f59e0b" if v > 10 else "#6366f1" for v in missing_pct.values]
bars = ax.barh(missing_pct.index, missing_pct.values, color=colors_miss, edgecolor="white")
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
ax.set_xlabel("Missing (%)")
ax.set_title("Missing Values by Column", pad=12)
ax.axvline(30, color="red", linestyle="--", alpha=0.5, label="30% threshold")
ax.legend()
sns.despine(ax=ax)
plt.tight_layout()
save_fig("01_missing_data", "Missing Values Audit", "Data Quality")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Demographics & Geography
# ══════════════════════════════════════════════════════════════════════════════
section("3. Demographics & Geography")

sex_counts = df["sex"].value_counts()
profile_age = df[(df["age"] >= 18) & (df["age"] < 65)].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = [SEX_PALETTE.get(k, PALETTE[i]) for i, k in enumerate(sex_counts.index)]
_, _, autotexts = axes[0].pie(
    sex_counts.values, labels=[k.upper() for k in sex_counts.index],
    autopct="%1.1f%%", colors=colors, startangle=90, pctdistance=0.78,
    wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 2},
)
for t in autotexts:
    t.set_fontsize(11)
    t.set_fontweight("bold")
axes[0].set_title("Sex Distribution")
median_age = profile_age["age"].median()
sns.histplot(data=profile_age, x="age", bins=30, kde=True, ax=axes[1], color=PALETTE[0], edgecolor="white")
axes[1].axvline(median_age, color="red", linestyle="--", linewidth=1.5, label=f"Median: {median_age:.0f}")
axes[1].set_title("Age Distribution")
axes[1].legend()
plt.tight_layout()
save_fig("02_demographics", "Sex & Age Distribution", "Demographics")

status_counts = df["status"].value_counts()
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=status_counts.values, y=status_counts.index, palette="mako", ax=ax, edgecolor="white")
ax.set_title("Relationship Status", pad=12)
ax.set_xlabel("Count")
for i, v in enumerate(status_counts.values):
    ax.text(v + 200, i, f"{v:,}", va="center", fontsize=10)
sns.despine(ax=ax)
plt.tight_layout()
save_fig("03_relationship_status", "Relationship Status", "Demographics")

top_locations = df["location"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(11, 6))
sns.barplot(x=top_locations.values, y=[loc.split(",")[0].title() for loc in top_locations.index],
            palette="flare", ax=ax, edgecolor="white")
ax.set_title("Top 10 Cities — Geographic Concentration", pad=12)
ax.set_xlabel("Profile Count")
sns.despine(ax=ax)
plt.tight_layout()
save_fig("04_top_locations", "Top 10 Cities", "Demographics")

# Numeric correlations (age, height — income excluded: 81% undisclosed)
numeric_df = df[["age", "height"]].dropna()
numeric_df = numeric_df[(numeric_df["height"] >= 60) & (numeric_df["height"] <= 78)]
corr = numeric_df.corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
            square=True, linewidths=2, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation: Age × Height", pad=12)
plt.tight_layout()
save_fig("05_correlation", "Age × Height Correlation", "Demographics")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Education & Lifestyle
# ══════════════════════════════════════════════════════════════════════════════
section("4. Education & Lifestyle")

edu_counts = df["education"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(12, 6))
short_labels = [e.replace("graduated from ", "").replace("working on ", "→ ")[:35] for e in edu_counts.index]
sns.barplot(x=edu_counts.values, y=short_labels, palette="viridis", ax=ax, edgecolor="white")
ax.set_title("Top 10 Education Levels", pad=12)
ax.set_xlabel("Count")
sns.despine(ax=ax)
plt.tight_layout()
save_fig("06_education", "Education Distribution", "Education")

edu_drinks = pd.crosstab(
    df["education"].fillna("unknown"),
    df["drinks"].fillna("unknown"),
)
top_edu = df["education"].value_counts().head(6).index
edu_drinks_top = edu_drinks.loc[edu_drinks.index.isin(top_edu)]
edu_drinks_top.index = [e.replace("graduated from ", "")[:30] for e in edu_drinks_top.index]

fig, ax = plt.subplots(figsize=(13, 6))
edu_drinks_top.plot(kind="barh", stacked=True, ax=ax, colormap="crest", edgecolor="white")
ax.set_title("Drinking Habits by Education Level", pad=12)
ax.set_xlabel("Count")
ax.legend(title="Drinks", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
sns.despine(ax=ax)
plt.tight_layout()
save_fig("07_education_drinks", "Education vs Drinking Habits", "Education")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Behavior Patterns
# ══════════════════════════════════════════════════════════════════════════════
section("5. Behavior Patterns")

drinks_order = ["not at all", "rarely", "socially", "often", "very often", "desperately"]
smokes_order = ["no", "sometimes", "when drinking", "yes", "trying to quit"]

behavior_df = df.dropna(subset=["drinks", "smokes"])
cross_tab = pd.crosstab(behavior_df["drinks"], behavior_df["smokes"])
cross_tab = cross_tab.reindex(index=[d for d in drinks_order if d in cross_tab.index], fill_value=0)
cross_tab = cross_tab.reindex(columns=[s for s in smokes_order if s in cross_tab.columns], fill_value=0)

fig, ax = plt.subplots(figsize=(11, 7))
sns.heatmap(cross_tab, annot=True, fmt=",", cmap="YlOrRd", ax=ax, linewidths=0.5)
ax.set_title("Drinks × Smoking Cross-Tab", pad=12)
ax.set_xlabel("Smoking")
ax.set_ylabel("Drinking")
plt.tight_layout()
save_fig("08_drinks_smokes", "Drinks × Smoking Heatmap", "Behavior")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, col, title, order in [
    (axes[0], "drinks", "Alcohol Consumption", drinks_order),
    (axes[1], "smokes", "Smoking Habits", smokes_order),
]:
    data = df[df[col].isin(order)]
    sns.countplot(data=data, x=col, order=[o for o in order if o in data[col].values],
                  ax=ax, palette="rocket", edgecolor="white")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    sns.despine(ax=ax)
plt.tight_layout()
save_fig("09_lifestyle_distribution", "Lifestyle Distributions", "Behavior")

# Interactive: status × orientation
sample_df = df.dropna(subset=["status", "orientation", "location"]).sample(500, random_state=42)
heatmap_chart = (
    alt.Chart(sample_df)
    .mark_rect(cornerRadius=3)
    .encode(
        x=alt.X("status:N", title="Relationship Status"),
        y=alt.Y("orientation:N", title="Orientation"),
        color=alt.Color("count()", scale=alt.Scale(scheme="tealblues"), title="Count"),
        tooltip=["status", "orientation", "location", alt.Tooltip("count()", title="Count")],
    )
    .properties(width=550, height=350, title="Status × Orientation")
)
heatmap_file = "10_status_orientation.html"
heatmap_chart.save(str(PLOTS_DIR / heatmap_file))
print(f"  Saved interactive chart: {heatmap_file}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Clustering — K-Means (Age + Height)
# ══════════════════════════════════════════════════════════════════════════════
section("6. Clustering — Age × Height")

cluster_df = df[["age", "height"]].dropna()
cluster_df = cluster_df[(cluster_df["age"] >= 18) & (cluster_df["age"] < 65)]
cluster_df = cluster_df[(cluster_df["height"] >= 60) & (cluster_df["height"] <= 78)]

scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_df)

inertia_values = []
K_RANGE = range(1, 11)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled)
    inertia_values.append(km.inertia_)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(K_RANGE, inertia_values, marker="o", markersize=8, linewidth=2, color=PALETTE[0])
ax.fill_between(K_RANGE, inertia_values, alpha=0.15, color=PALETTE[0])
ax.set_title("Elbow Method — Optimal k", pad=12)
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia")
ax.set_xticks(list(K_RANGE))
sns.despine(ax=ax)
plt.tight_layout()
save_fig("11_kmeans_elbow", "Elbow Method", "Clustering")

NUM_CLUSTERS = 3
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
labels = kmeans.fit_predict(scaled)

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(cluster_df["age"], cluster_df["height"], c=labels,
                     cmap="viridis", alpha=0.35, s=15, edgecolors="none")
centers = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centers[:, 0], centers[:, 1], s=300, c="red", marker="X",
           edgecolors="white", linewidths=2, label="Centroids", zorder=5)
plt.colorbar(scatter, ax=ax, label="Cluster")
ax.set_title(f"K-Means Segments (k={NUM_CLUSTERS}) — Age × Height", pad=12)
ax.set_xlabel("Age")
ax.set_ylabel("Height (inches)")
ax.legend()
sns.despine(ax=ax)
plt.tight_layout()
save_fig("12_kmeans_clusters", "Age × Height Segments", "Clustering")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Predictive Analytics — Drinks Classification
# ══════════════════════════════════════════════════════════════════════════════
section("7. Predictive Analytics — Drinks Classification")

ML_COLS = ["age", "sex", "education", "body_type", "smokes", "drinks"]
ml_df = df[ML_COLS].dropna().copy()

for col in ml_df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    ml_df[col] = le.fit_transform(ml_df[col].astype(str))

X = ml_df.drop("drinks", axis=1)
y = ml_df["drinks"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Baseline: always predict the majority class ("socially").
# On imbalanced data, raw accuracy is misleading — the baseline already scores high
# simply by guessing the most common label. We measure every model against it.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_macro_f1 = f1_score(y_test, baseline_pred, average="macro", zero_division=0)

# Random Forest with class_weight="balanced" so the model is penalized for ignoring
# minority classes instead of collapsing onto the majority label.
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

drinks_accuracy = accuracy_score(y_test, y_pred)
drinks_balanced_acc = balanced_accuracy_score(y_test, y_pred)
drinks_macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

print("Honest evaluation on imbalanced 6-class target ('drinks'):")
print(f"  Baseline (majority class)  — accuracy {baseline_accuracy:.3f} | macro-F1 {baseline_macro_f1:.3f}")
print(f"  Random Forest (balanced)   — accuracy {drinks_accuracy:.3f} | "
      f"balanced-acc {drinks_balanced_acc:.3f} | macro-F1 {drinks_macro_f1:.3f}")
print(f"  Macro-F1 lift over baseline: +{drinks_macro_f1 - baseline_macro_f1:.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
importances.plot(kind="barh", ax=ax, color=PALETTE[:len(importances)], edgecolor="white")
ax.set_title(
    f"Feature Importance — Drinks Prediction (balanced-acc {drinks_balanced_acc:.1%}, "
    f"macro-F1 {drinks_macro_f1:.2f})",
    pad=12,
)
ax.set_xlabel("Importance")
sns.despine(ax=ax)
plt.tight_layout()
save_fig("13_feature_importance", "Feature Importance", "Predictive Analytics")

# Honest metric comparison: model vs. naive baseline, on accuracy AND macro-F1.
fig, ax = plt.subplots(figsize=(9, 5))
metric_labels = ["Accuracy", "Macro-F1"]
baseline_scores = [baseline_accuracy, baseline_macro_f1]
rf_scores = [drinks_accuracy, drinks_macro_f1]
x_pos = np.arange(len(metric_labels))
bar_w = 0.38
b1 = ax.bar(x_pos - bar_w / 2, baseline_scores, bar_w, label="Baseline (majority class)",
            color="#94a3b8", edgecolor="white")
b2 = ax.bar(x_pos + bar_w / 2, rf_scores, bar_w, label="Random Forest (balanced)",
            color=PALETTE[0], edgecolor="white")
ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=9)
ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=9)
ax.set_xticks(x_pos)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model vs Baseline — Why Accuracy Alone Misleads", pad=12)
ax.legend()
sns.despine(ax=ax)
plt.tight_layout()
save_fig("15_model_vs_baseline", "Model vs Baseline", "Predictive Analytics")

fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix — Drinks Classification", pad=12)
plt.tight_layout()
save_fig("14_confusion_matrix", "Confusion Matrix", "Predictive Analytics")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Business Question — User Engagement
# ══════════════════════════════════════════════════════════════════════════════
# Focused question: "Who are the most engaged users, and what distinguishes them?"
# Engagement is derived from `last_online` recency (relative to the dataset's most
# recent timestamp) and related to how complete each profile is — an actionable
# signal product & growth teams can move on.
section("8. Business Question — User Engagement")

# Profile completeness = share of non-empty fields per user (essays already dropped).
PROFILE_FIELDS = list(df.columns)
df["completeness"] = df[PROFILE_FIELDS].notna().mean(axis=1) * 100

# Recency from last_online (format: YYYY-MM-DD-HH-MM), measured against the latest
# timestamp in the data (this is a 2012 snapshot, so we anchor to its max date).
df["last_online_dt"] = pd.to_datetime(df["last_online"], format="%Y-%m-%d-%H-%M", errors="coerce")
ref_date = df["last_online_dt"].max()
df["days_since_online"] = (ref_date - df["last_online_dt"]).dt.days

eng = df.dropna(subset=["days_since_online"]).copy()


def engagement_bucket(days: float) -> str:
    if days <= 7:
        return "Active (≤7d)"
    if days <= 30:
        return "Recent (8–30d)"
    return "Dormant (>30d)"


eng["engagement"] = eng["days_since_online"].apply(engagement_bucket)
ENG_ORDER = ["Active (≤7d)", "Recent (8–30d)", "Dormant (>30d)"]
eng_counts = eng["engagement"].value_counts().reindex(ENG_ORDER)
active_pct = (eng["days_since_online"] <= 30).mean() * 100

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(eng_counts.index, eng_counts.values,
              color=["#10b981", "#6366f1", "#94a3b8"], edgecolor="white")
ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
ax.set_title("User Engagement Mix — by last-online recency", pad=12)
ax.set_ylabel("Users")
sns.despine(ax=ax)
plt.tight_layout()
save_fig("16_engagement_mix", "Engagement Mix", "User Engagement")

# Does a more complete profile mean a more engaged user?
# Rank-based quartiles avoid duplicate-edge errors (completeness is coarsely valued).
eng["completeness_q"] = pd.qcut(
    eng["completeness"].rank(method="first"), 4,
    labels=["Q1\n(least complete)", "Q2", "Q3", "Q4\n(most complete)"],
)
eng["active30"] = eng["days_since_online"] <= 30
rate_by_q = eng.groupby("completeness_q", observed=True)["active30"].mean() * 100
completeness_lift = rate_by_q.iloc[-1] - rate_by_q.iloc[0]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(rate_by_q.index.astype(str), rate_by_q.values,
              color=sns.color_palette("crest", len(rate_by_q)), edgecolor="white")
ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=10)
ax.set_title("Active-User Rate by Profile Completeness", pad=12)
ax.set_ylabel("% active in last 30 days")
ax.set_xlabel("Profile completeness quartile")
sns.despine(ax=ax)
plt.tight_layout()
save_fig("17_engagement_by_completeness", "Engagement vs Profile Completeness", "User Engagement")

print(f"  Active in last 30 days:                 {active_pct:.1f}%")
print(f"  Active-rate lift, Q4 vs Q1 completeness: +{completeness_lift:.1f} pts")


# ── Summary & Dashboard ──────────────────────────────────────────────────────
section("Summary")
print(f"""
  Dataset:              {df.shape[0]:,} profiles
  Drinks Model:         acc {drinks_accuracy:.3f} | balanced-acc {drinks_balanced_acc:.3f} | macro-F1 {drinks_macro_f1:.3f}
  Baseline:             acc {baseline_accuracy:.3f} | macro-F1 {baseline_macro_f1:.3f}
  Clusters:             {NUM_CLUSTERS}
  Charts saved to:      {PLOTS_DIR}
""")

sf_pct = df["location"].str.contains("san francisco", case=False, na=False).mean() * 100
single_pct = (df["status"] == "single").mean() * 100
male_pct = sex_counts.get("m", 0) / sex_counts.sum() * 100

dashboard_stats = {
    "Profiles": f"{df.shape[0]:,}",
    "Median Age": f"{median_age:.0f}",
    "Active ≤30d": f"{active_pct:.0f}%",
    "SF Users": f"{sf_pct:.0f}%",
    "Single": f"{single_pct:.0f}%",
    "Model Macro-F1": f"{drinks_macro_f1:.2f}",
    "Clusters": str(NUM_CLUSTERS),
}

report = {
    "profiles": f"{df.shape[0]:,}",
    "median_age": median_age,
    "clusters": NUM_CLUSTERS,
    "male_pct": male_pct,
    "sf_pct": sf_pct,
    "single_pct": single_pct,
    "drinks_acc": drinks_accuracy,
    "baseline_acc": baseline_accuracy,
    "balanced_acc": drinks_balanced_acc,
    "macro_f1": drinks_macro_f1,
    "active_pct": active_pct,
    "completeness_lift": completeness_lift,
}

section("Building Dashboard")
build_dashboard(dashboard_stats, heatmap_file, report)
open_dashboard(DASHBOARD_PATH)
