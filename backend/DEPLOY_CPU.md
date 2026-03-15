# PureArc CPU Droplet Deploy

## 1) System packages
```bash
sudo apt update
sudo apt install -y python3 python3-venv ffmpeg
```

`ffmpeg` is only required when `PUREARC_SAVE_OUTPUT=1`.

## 2) Project setup
```bash
cd backend
cp .env.example .env
# Edit .env with your agent URLs/keys and runtime settings
```

## 3) Start services
```bash
chmod +x scripts/start_cpu_droplet.sh
./scripts/start_cpu_droplet.sh
```

This script:
- creates/uses `venv`
- installs `requirements-cpu.txt`
- starts API on port `8000`
- optionally starts `main.py` if `PUREARC_RUN_MAIN=1`

## 4) Verify
```bash
curl http://127.0.0.1:8000/health
```

## 5) Logs
```bash
tail -f logs/api.log
tail -f logs/main.log
```

## Notes
- If `PUREARC_VIDEO_SOURCE` is empty or invalid, API still runs and `main.py` is skipped.
- Ensure shot detector weights exist at `runs/shot_detector/weights/best.pt` if trajectory scoring is required.
- Keep `.env` out of git; rotate any keys previously exposed.
