# 🚦 Traffic Intersection Visualizer

A professional, high‑performance visualization tool for traffic intersection simulations. Reads standard `roadnet.json` and `replay.txt` formats, generates publication‑ready static frames, and animates complete traffic flow with lane‑accurate traffic lights.

---

## 🎬 Simulation Video

Click below to watch the simulation video in a new browser tab:

<div align="center"> <a href="https://youtu.be/hb3mCM8DzbM" target="_blank"> <img src="frame_000.png" alt="Traffic Intersection Simulation" width="800" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"> <br> <img src="https://img.shields.io/badge/Watch%20on-YouTube-red?style=for-the-badge&logo=youtube" alt="Watch on YouTube"> </a> <p><em>▶️ Click the image above to watch the full simulation video on YouTube</em></p> </div>
                                                                                                                                                                                                                                                                                                                                                                                                                                                        
**📥 Download Options:**

* [Download MP4](https://raw.githubusercontent.com/210041258/CityFlow-generated-001/refs/heads/master/simulation/frontend_replay/2026-02-12.mp4)

* [Download WebM]([https://raw.githubusercontent.com/210041258/CityFlow-generated-001/master/simulation/2026-02-12.webm](https://raw.githubusercontent.com/210041258/CityFlow-generated-001/refs/heads/master/simulation/frontend_replay/2026-02-12.mp4))

---

## ✨ Features

### 🎯 Core Visualization

* Accurate road geometry from `roadnet.json`
* Solid outer lane edges and dashed lane dividers
* Lane‑accurate traffic lights (R/G/Y per lane)
* Detailed vehicles with headlights, taillights, and shadows
* Stable vehicle colors across all frames

### 🚀 Performance

* Blitted animation (static infrastructure drawn once)
* 10–50× faster than full redraw approaches
* Smooth playback with 1000+ frames

### 📊 Reporting & Export

* Console summary tables (intersections, edges, lights)
* High‑resolution static PNG (1800×1800)
* Vector PDF export
* Optional MP4 export (requires ffmpeg)

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

* Install **ffmpeg** and ensure it is available in PATH

---

## 📤 Output

* `frame_000.png` – High‑resolution static frame
* `frame_000.pdf` – Vector export
* `traffic_simulation.mp4` – Optional exported animation

---

## 📄 License

MIT License

---

## 👤 Author

Traffic Intersection Visualizer – 2026 Edition
