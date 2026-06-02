# component_fab Visual Explainer 🎨

This directory contains the **Visual Explainer**, an interactive, browser-based dashboard designed to turn the complex math of neural architectures into an intuitive, animated story.

It is built specifically for visual learners to demystify how different "lanes" (model designs) process information, manage memory, and evolve over time.

---

## 🚀 How to Access

To start the visualizer, run the following command from the project root:

```bash
python -m component_fab.viz
```

Once the server is running, open your browser to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🗺️ What’s Inside?

The site is organized into three main types of views, accessible via the sidebar:

### 1. 🏗️ Architecture Explainer (The "Lanes")
Select any specific model design from the sidebar to see its inner workings. Each lane page features:
*   **Animated Token-Flow:** A "cartoon-style" infographic showing tokens (words) flying through the specific stages of that architecture.
*   **🔰 Plain English Analogies:** Jargon-free explanations that use metaphors (like notebooks, drawers, and focus knobs) to describe the math.
*   **Mechanism Equations:** The raw "Write" and "Read" recipes for technical users.
*   **Token Mixing Map:** A heatmap showing how far a word's influence travels into the future.
*   **Memory Trace (Live):** An interactive player where you can watch the model's "scratchpad" (memory matrix) fill up token-by-token.
*   **Algebra Spectrum:** A chart showing if a model has taught itself to be "fuzzy" (blending facts) or "sharp" (exact recall).

### 2. 🧪 Live Testing
Watch the "factory floor" in real-time. This page instantiates every registered design and runs a battery of "smoke tests" to ensure they don't blow up and measures how well they mix information.
*   **Shows:** Stability, parameter counts, and "half-life" (how quickly the model forgets).

### 3. 🏆 The Hall of Fame
The "Tournament Bracket" of the project's history. This view replays the results of past research runs.
*   **Shows:** Which designs scored the highest, which were **Promoted** to the next round of research, and which were archived. It turns the raw data of a "ledger" into a clear leaderboard of architectural evolution.

---

## 🛠️ Technical Stack
*   **Backend:** FastAPI (Python)
*   **Frontend:** Vanilla JS / CSS (No build step required)
*   **Charts:** Plotly.js
*   **Animations:** CSS Keyframes & SVG
