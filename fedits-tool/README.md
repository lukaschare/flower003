# FedITS-Tool (Veins + SUMO + FL Orchestrator skeleton)

This repository is a minimal, runnable scaffold for an *online* co-simulation tool:
- Mobility + wireless/network decisions come from **Veins/SUMO (sim time)**
- Local training runs in **Docker/Flower (real compute)**
- **Orchestrator** enforces commit/drop semantics and logs per-round latency/energy/carbon

## Quick start (runs immediately, no Veins/Flower required)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run mock end-to-end (produces outputs/runs/<run_id>/csv)
make run-mock
