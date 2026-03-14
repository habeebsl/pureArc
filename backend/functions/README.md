# DigitalOcean Function: YouTube Drill Links

This serverless function returns YouTube video recommendations for drill titles.

Function name:
- `recommend/youtube_links`

## Inputs

```json
{
  "drill_titles": ["Speed Shooting Drill", "Partner Shooting"],
  "topic": "basketball shooting drills",
  "max_results_per_drill": 2,
  "region_code": "US"
}
```

## Output

```json
{
  "ok": true,
  "provider": "youtube_data_api",
  "results": [
    {
      "drill_title": "Speed Shooting Drill",
      "videos": [
        {
          "title": "...",
          "url": "https://www.youtube.com/watch?v=...",
          "channel": "...",
          "published_at": "...",
          "thumbnail": "...",
          "source": "youtube_data_api"
        }
      ]
    }
  ],
  "errors": []
}
```

If `YOUTUBE_API_KEY` is not set, it returns deterministic YouTube search URLs as fallback.

## Deploy

```bash
cd backend/functions

doctl serverless connect <namespace>
doctl serverless deploy .
```

Or use the helper script:

```bash
cd backend/functions
./deploy.sh
```

## Invoke (CLI)

```bash
doctl serverless functions invoke recommend/youtube_links \
  --param drill_titles:='["Speed Shooting Drill","Partner Shooting"]' \
  --param topic:='basketball shooting drills' \
  --param max_results_per_drill:=2
```

## Required Secret

Set in your function namespace/package environment:
- `YOUTUBE_API_KEY`

Without this secret, function still works with fallback query links.
