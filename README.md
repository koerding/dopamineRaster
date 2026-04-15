# Dopamine raster digitization and replot

This repository packages the full workflow from the chat session:

- the raw source image
- the inferred spike-time table
- diagnostic overlays and reconstruction figures
- the final modern replot
- scripts that reproduce the extraction and plotting workflow

## Contents

- `raw/dopamine_raster_source.png`: original raster image used as input
- `data/dopamine_raster_spike_times.csv`: extracted spike times and positions
- `data/dopamine_raster_panel_metadata.csv`: panel bounds and extraction metadata
- `figures/`: diagnostic and final figures
- `notes/dopamine_raster_extraction_notes.txt`: short notes on assumptions
- `scripts/00_extract_spikes_from_raster.py`: approximate spike extraction script
- `scripts/01_replot_modern.py`: plotting script for the final modern figure
- `requirements.txt`: Python dependencies

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/00_extract_spikes_from_raster.py
python scripts/01_replot_modern.py
```

## Important caveat

The spike table is an approximate digitization from a raster scan. The extraction script is designed to be editable and practical:
it models the image as a combination of vertical event lines and roughly circular spike dots, then uses residual darkness near
the lines to recover touching dots and merges nearby detections.

This should reproduce the workflow and outputs at a practical level, but it is not exact ground truth.

## Final axis conventions

The final figure follows the requested axis ranges:
- top panel: `-0.5` to `2.0`
- middle panel: `-1.0` to `2.0`
- bottom panel: `-1.0` to `2.0`

All panels end at `2.0`.
