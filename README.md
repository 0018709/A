# NBVEXGEXheatmap

Mobile-friendly static GitHub Pages dashboard for NBIS options.

## What it does

- Downloads option chains from Yahoo Finance via `yfinance`
- Calculates Black-Scholes Gamma and Vanna
- Builds heuristic NET GEX/VEX (`calls - puts`)
- Saves `data/latest.json`
- Saves timestamped hourly snapshots in `data/history/`
- Displays the heatmap in `index.html`

## Important

This is an analytical model, not a source of known dealer positions.
Open interest does not reveal whether dealers are long or short the contracts.
