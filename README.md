# Inky Stylise — Local Image Stylisation for E-Paper Display

A cross-platform system that transforms photos of people and animals into classic historical painting styles, displayed on a **Pimoroni Inky Impression 13.3"** e-paper display. Runs on **Raspberry Pi** and **MacBook Pro (Apple Silicon)**.

---

## 1. Project Summary

| Feature | Detail |
|---|---|
| **Input** | Photos from a shared Google Drive folder (or local files) |
| **Processing** | Neural Style Transfer (NST) and/or Diffusion img2img |
| **Output** | 1600×1200 image for Inky Impression 13.3" e-paper |
| **Schedule** | Refreshes once per day |
| **Platforms** | Raspberry Pi (arm64), macOS (Apple Silicon) |

The system watches a Google Drive folder, selects an image, applies a classic painting style, and updates the e-paper display. NST runs on any hardware. Diffusion img2img is available on macOS with MPS acceleration and optionally on Pi (slow, CPU-only).

---

## 2. Assumptions

- **Python 3.14** managed via [uv](https://docs.astral.sh/uv/) on both platforms.  
- **Raspberry Pi**: 64-bit OS (Bookworm), 4GB+ RAM recommended. Diffusion on Pi is experimental — NST is the reliable baseline.  
- **MacBook Pro**: Apple Silicon (M1/M2/M3/M4). MPS acceleration used for both NST and diffusion.  
- **Google Drive**: Service-account credentials are used for unattended sync. Drive is for shared storage only — no cloud processing.  
- **Inky Impression 13.3"**: Connected via SPI on the Pi. On macOS the display is simulated (PNG file saved).  
- Style reference paintings must be provided by the user in `data/styles/<style_name>/`.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────┐
│                    CLI / Scheduler               │
│                  (src/cli.py, cron)               │
├─────────────────────────────────────────────────┤
│                Main Orchestrator                 │
│                  (src/main.py)                    │
├──────────┬──────────┬──────────┬────────────────┤
│  Google  │  Image   │ Style    │  Inky Display  │
│  Drive   │ Pipeline │ Profiles │  Integration   │
│  Sync    │          │          │                │
├──────────┼──────────┼──────────┼────────────────┤
│ gdrive.py│selector  │profiles  │ inky_display.py│
│          │preprocess│          │                │
│          │nst       │          │                │
│          │diffusion │          │                │
│          │postproc  │          │                │
├──────────┴──────────┴──────────┴────────────────┤
│              Utilities & Config                  │
│         (config, logging, files, image_ops)      │
└─────────────────────────────────────────────────┘
```

---

## 4. Folder Structure

```
Inky/
├── .env.example              # Environment template
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
├── tasks.md                  # Task tracker
│
├── data/
│   ├── cache/                # Local sync cache (raw/, parsed/, etc.)
│   ├── styles/               # Style reference paintings
│   │   ├── renaissance_portrait/
│   │   ├── baroque_oil_painting/
│   │   └── ...
│   ├── output/               # High-resolution stylised outputs
│   ├── display/              # Display-ready images (1600×1200)
│   ├── archive/              # Processed source images
│   ├── metadata/             # display_history.json
│   └── logs/                 # Application logs
│
├── scripts/
│   ├── run_daily.sh          # Cron wrapper
│   ├── setup_pi.sh           # Raspberry Pi setup
│   └── setup_mac.sh          # macOS setup
│
├── src/
│   ├── __init__.py
│   ├── cli.py                # Click CLI
│   ├── main.py               # Pipeline orchestrator
│   ├── config.py             # Pydantic settings
│   ├── logging_utils.py      # Structured logging
│   │
│   ├── models/
│   │   └── style_profiles.py # Style registry (10 built-in styles)
│   │
│   ├── pipeline/
│   │   ├── selector.py       # Image selection logic
│   │   ├── preprocess.py     # Image prep (resize, crop)
│   │   ├── nst.py            # Neural Style Transfer (VGG-19)
│   │   ├── diffusion.py      # Diffusion img2img
│   │   └── postprocess.py    # Display prep & save
│   │
│   ├── integrations/
│   │   ├── google_drive.py   # Drive sync & upload
│   │   └── inky_display.py   # Inky Impression driver
│   │
│   ├── scheduler/
│   │   └── daily_job.py      # Python-based scheduler
│   │
│   └── utils/
│       ├── files.py          # File operations
│       ├── image_ops.py      # Pillow helpers
│       └── metadata.py       # Display history tracking
│
└── tests/
    ├── test_config.py
    ├── test_selector.py
    └── test_style_profiles.py
```

---

## 5. Quick Start

### macOS

```bash
git clone <repo-url> && cd Inky
chmod +x scripts/setup_mac.sh && ./scripts/setup_mac.sh
# Edit .env with your settings
cp .env.example .env

# Place a style reference image:
mkdir -p data/styles/renaissance_portrait
cp ~/my_reference_painting.jpg data/styles/renaissance_portrait/reference.jpg

# Run with a local image (no Drive, no display):
uv run python -m src.cli run -i ~/photo.jpg --skip-sync --skip-display --skip-upload
```

### Raspberry Pi

```bash
git clone <repo-url> && cd Inky
chmod +x scripts/setup_pi.sh && ./scripts/setup_pi.sh
cp .env.example .env
# Edit .env ...

uv run python -m src.cli run -i ~/photo.jpg --skip-sync
```

---

## 6. CLI Commands

```bash
# Full pipeline
uv run python -m src.cli run

# Explicit image + style
uv run python -m src.cli run -i photo.jpg -s baroque_oil_painting

# Force diffusion algorithm
uv run python -m src.cli run -a diffusion

# Use a specific selection mode
uv run python -m src.cli run -m random_raw

# List available styles
uv run python -m src.cli styles

# Show configuration
uv run python -m src.cli config

# Sync Google Drive
uv run python -m src.cli sync
uv run python -m src.cli sync --subfolder parsed

# Push image to display
uv run python -m src.cli display data/display/some_image.png

# View display history
uv run python -m src.cli history

# Start built-in scheduler (blocking)
uv run python -m src.cli schedule
```

---

## 7. Style System

10 built-in styles, each with diffusion prompts, NST reference guidance, and subject affinity:

| Style | Best For | Notes |
|---|---|---|
| `renaissance_portrait` | Humans | Soft chiaroscuro, warm tones |
| `baroque_oil_painting` | Both | Dramatic Caravaggio lighting |
| `rococo_portrait` | Humans | Pastel, delicate, 18th-century French |
| `dutch_golden_age` | Both | Rembrandt lighting, dark backgrounds |
| `romanticism` | Both | Dramatic, emotional, saturated |
| `impressionism` | Both | Visible brushstrokes, plein air |
| `post_impressionism` | Both | Bold colour, Van Gogh / Cézanne |
| `victorian_animal_portrait` | Animals | Edwin Landseer, dignified noble poses |
| `classical_equestrian` | Animals | George Stubbs, horse portraits |
| `naturalist_oil_portrait` | Both | Sargent, refined realism |

### Adding a custom style

1. Add a reference painting to `data/styles/<style_name>/reference.jpg`
2. Register a new `StyleProfile` in `src/models/style_profiles.py`

---

## 8. Image Selection Logic

| Priority | Condition | Action |
|---|---|---|
| 1 | `--input` flag provided | Use that file |
| 2 | Mode = `latest_parsed` | Newest file from `cache/parsed/` |
| 3 | Mode = `random_raw` | Random file from `cache/raw/` |
| 4 | Mode = `random_any` | Random from raw + parsed |

If `latest_parsed` finds no files, it falls back to `random_raw`.

---

## 9. Pipeline Design

```
Source Image → Preprocess → Stylise (NST or Diffusion) → Postprocess → Display
                                                              ↓
                                                        Save hi-res
                                                        Save display
                                                        Upload to Drive
                                                        Record metadata
                                                        Archive source
```

### NST Pipeline
- VGG-19 feature extractor (content layers: conv_4; style layers: conv_1–5)
- L-BFGS optimiser with configurable content/style weights and step count
- Runs on CPU, MPS, or CUDA

### Diffusion Pipeline
- Stable Diffusion img2img via HuggingFace `diffusers`
- Style-specific prompts and negative prompts
- Hardware-aware: MPS on macOS, CPU fallback, optional reduced resolution on Pi
- Auto-fallback to NST when diffusion is unavailable or impractical

---

## 10. Raspberry Pi Notes

| Concern | Detail |
|---|---|
| **NST** | Fully supported. 300 steps at 512px takes ~5–15 min on Pi 4/5. Increase `NST_OUTPUT_LONG_EDGE` cautiously. |
| **Diffusion** | Technically possible but extremely slow (30+ min at 512px). Disabled by default. |
| **RAM** | 4GB minimum. Close other applications during processing. |
| **Display** | Inky Impression 13.3" connected via SPI. Full colour (7-colour) supported. |
| **Scheduling** | Use cron or systemd timer (see below). |

### Systemd timer (recommended)

```ini
# /etc/systemd/system/inky-stylize.service
[Unit]
Description=Inky Stylise Daily Job
After=network-online.target

[Service]
Type=oneshot
User=pi
WorkingDirectory=/home/pi/Inky
ExecStart=/home/pi/Inky/scripts/run_daily.sh
```

```ini
# /etc/systemd/system/inky-stylize.timer
[Unit]
Description=Run Inky Stylise daily

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now inky-stylize.timer
```

---

## 11. macOS Notes

| Concern | Detail |
|---|---|
| **NST** | Fast with MPS acceleration (M1/M2/M3). |
| **Diffusion** | Recommended path. MPS-accelerated, ~30s–2min per image. |
| **Display** | Simulated — saves PNG to `data/display/simulated_display.png`. |
| **Scheduling** | Use cron or launchd plist. |

### launchd plist

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.inky.stylize</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/YOU/Inky/scripts/run_daily.sh</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>6</integer>
    <key>Minute</key>
    <integer>0</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>/Users/YOU/Inky/data/logs/launchd.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/YOU/Inky/data/logs/launchd_err.log</string>
</dict>
</plist>
```

Save as `~/Library/LaunchAgents/com.inky.stylize.plist`, then:
```bash
launchctl load ~/Library/LaunchAgents/com.inky.stylize.plist
```

---

## 12. Google Drive Integration

The system uses a **service account** for unattended access.

### Setup
1. Create a project in Google Cloud Console
2. Enable the Google Drive API
3. Create a service-account key (JSON)
4. Share the Drive folder with the service account's email address
5. Set `GDRIVE_SERVICE_ACCOUNT_KEY` and `GDRIVE_ROOT_FOLDER_ID` in `.env`

### Expected Drive folder structure
```
<shared folder>/
├── raw/        ← unprocessed source photos
├── parsed/     ← manually prepared photos (from MacBook)
├── styled/     ← stylised hi-res outputs (uploaded by system)
├── display/    ← display-ready images (uploaded by system)
├── archive/    ← processed inputs (future)
└── logs/       ← run logs (future)
```

### Offline resilience
Files are synced to `data/cache/`. The pipeline works from cache even when offline.

---

## 13. Subject-Aware Guidance

### Portraits (humans)
- Crop tightly to head + shoulders for best identity preservation
- Styles like `renaissance_portrait`, `rococo_portrait`, `dutch_golden_age` emphasise facial features
- NST preserves face geometry better at lower style weights

### Animals
- Preserve muzzle, eyes, and fur silhouette
- `victorian_animal_portrait` and `classical_equestrian` are purpose-built
- Wider crops often work better for animals than tight crops
- Higher style weights are usually acceptable since "identity" is more about shape than face

---

## 14. Testing

```bash
uv run pytest tests/ -v
uv run pytest tests/ -v --cov=src
```

### Testing strategy
- **Unit tests**: Config parsing, selector logic, style registry (no ML deps)
- **Integration tests** (manual): Run full pipeline on test images, verify output quality
- **Hardware tests**: Test Inky display on Pi, test MPS acceleration on Mac
- **Mocking**: Google Drive calls are best tested via mocking `googleapiclient`

---

## 15. Future Improvements

- [ ] Web UI for style selection and preview
- [ ] Face detection for automatic subject classification (human vs animal)
- [ ] Multi-display support (multiple Inky screens)
- [ ] ControlNet integration for structure-preserving diffusion
- [ ] ONNX Runtime optimisation for faster Pi inference
- [ ] Style transfer quality metrics (SSIM, LPIPS)
- [ ] Gallery mode — rotate through recent outputs
- [ ] Webhook / notification when display updates
- [ ] Remote monitoring dashboard
- [ ] Custom style upload via web interface

---

## License

MIT
