# 🚦 Traffic Intersection Visualizer

A professional, high‑performance visualization tool for traffic intersection simulations. Reads standard `roadnet.json` and `replay.txt` formats, generates publication‑ready static frames, and animates complete traffic flow with lane‑accurate traffic lights.

---

## 🎬 Demo Video

The demo animation is available in this directory:

**`2026-02-12.mp4`**

### Embedded Video Preview
You can embed the video directly in Markdown for GitHub or local preview:

```markdown
<video width="800" controls>
  <source src="2026-02-12.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

<video width="100%" controls autoplay loop muted poster="frame_000.png">
  <source src="https://raw.githubusercontent.com/210041258/CityFlow-generated-001/master/simulation/2026-02-12.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

This will display a video player directly in your README when viewed in compatible platforms.

---

## ✨ Features

### 🎯 Core Visualization
- Accurate road geometry from `roadnet.json`
- Solid outer lane edges and dashed lane dividers
- Lane‑accurate traffic lights (R/G/Y per lane)
- Detailed vehicles with headlights, taillights, and shadows
- Stable vehicle colors across all frames

### 🚀 Performance
- Blitted animation (static infrastructure drawn once)
- 10–50× faster than full redraw approaches
- Smooth playback with 1000+ frames

### 📊 Reporting & Export
- Console summary tables (intersections, edges, lights)
- High‑resolution static PNG (1800×1800)
- Vector PDF export
- Optional MP4 export (requires ffmpeg)

---

## ⚡ Quick Start

```bash
python traffic_vis.py
```

Or with custom paths:

```bash
python traffic_vis.py \
  --roadnet ./data/roadnet.json \
  --replay ./data/replay.txt
```

---

## 📁 Expected Files

```
.
├── traffic_vis.py
├── roadnet.json
├── replay.txt
├── 2026-02-12.mp4
└── README.md
```

---

## 📦 Requirements

Required:

```bash
pip install matplotlib numpy
```

Optional:

```bash
pip install tabulate tqdm ffmpeg-python
```

System requirement for MP4 export:

- Install **ffmpeg** and ensure it is available in PATH

---

## 📤 Output

- `frame_000.png` – High‑resolution static frame
- `frame_000.pdf` – Vector export
- `traffic_simulation.mp4` – Optional exported animation

---

## 📄 License

MIT License

---

## 👤 Author

Traffic Intersection Visualizer – 2026 Edition

---

## 💾 Download
You can download this README as a standalone file [here](./README.md).
