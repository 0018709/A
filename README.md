# NBIS VEX / GEX heatmap

Mobile-friendly static GitHub Pages dashboard for NBIS options.

## What it does

- Downloads NBIS option chains from Yahoo Finance via `yfinance`
- Calculates Black-Scholes Gamma and Vanna
- Builds heuristic NET GEX/VEX (`calls - puts`)
- Saves `data/latest.json`
- Saves timestamped hourly snapshots in `data/history/`
- Displays the heatmap in `index.html`

## Important

This is an analytical model, not a source of known dealer positions.
Open interest does not reveal whether dealers are long or short the contracts.

## GitHub Pages

After uploading the files:
1. Run **Actions → Update NBIS options heatmap → Run workflow** once.
2. Go to **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select branch **main** and folder **/(root)**.
5. Save.

The site will then be available at your GitHub Pages URL.
