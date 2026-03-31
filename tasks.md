# Inky Impression Image Stylization System — Task Tracker

## Project Summary
Cross-platform local image stylization system that transforms photos of people and animals into classic historical painting styles, displayed on a Pimoroni Inky Impression 13 e-paper display. Runs on Raspberry Pi and MacBook Pro with Google Drive sync for source images.

---

## Phase 1: Foundation (Core Infrastructure)
- [x] Project folder structure
- [x] tasks.md
- [x] README.md
- [x] requirements.txt
- [x] pyproject.toml
- [x] .env.example
- [x] Configuration system (src/config.py)
- [x] Logging utilities (src/logging_utils.py)
- [x] File utilities (src/utils/files.py)
- [x] Image operations (src/utils/image_ops.py)
- [x] Metadata tracking (src/utils/metadata.py)

## Phase 2: Style System & Image Pipeline
- [x] Style profile definitions (src/models/style_profiles.py)
- [x] Image selector (src/pipeline/selector.py)
- [x] Image preprocessor (src/pipeline/preprocess.py)
- [x] Neural Style Transfer engine (src/pipeline/nst.py)
- [x] Diffusion img2img engine (src/pipeline/diffusion.py)
- [x] Postprocessor / display preparation (src/pipeline/postprocess.py)

## Phase 3: Integrations
- [x] Google Drive sync (src/integrations/google_drive.py)
- [x] Inky Impression display (src/integrations/inky_display.py)

## Phase 4: CLI & Orchestration
- [x] CLI entrypoint (src/cli.py)
- [x] Main pipeline orchestrator (src/main.py)
- [x] Daily job scheduler (src/scheduler/daily_job.py)

## Phase 5: Scripts & Deployment
- [x] scripts/run_daily.sh
- [x] scripts/setup_pi.sh
- [x] scripts/setup_mac.sh

## Phase 6: Testing
- [x] tests/test_config.py
- [x] tests/test_selector.py
- [x] tests/test_style_profiles.py

## Phase 7: Future Improvements (Backlog)
- [ ] Web UI for style selection and preview
- [ ] Face detection for auto subject classification
- [ ] Multi-display support
- [ ] Style transfer quality metrics
- [ ] Gallery mode (rotate through recent outputs)
- [ ] Webhook/notification when new display image is set
- [ ] ControlNet integration for diffusion pipeline
- [ ] ONNX Runtime optimization for Pi
- [ ] Remote monitoring dashboard
