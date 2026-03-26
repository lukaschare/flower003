
# FedITS: Federated Learning Orchestration in Vehicular Networks 🚗📡🤖

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Instant%20Veins%20%7C%20Linux-lightgrey)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)

## 1. Overview
This project provides a comprehensive co-simulation framework that integrates **Federated Learning (FL)** with **Vehicular Ad Hoc Networks (VANETs)**. By combining **Flower** (for FL) with **Veins, SUMO, and OMNeT++** (for mobility and network simulation), it enables researchers to evaluate FL algorithms under realistic vehicular mobility, communication latency, and dynamic topology constraints.

### 🌟 Key Features
* **Decoupled Co-Simulation Architecture:** Isolates the FL logic into Docker containers (Flower + PyTorch) while running the network simulation natively in OMNeT++, bridged by a custom Python Orchestrator.
* **Realistic Constraints:** Judges FL model downlink/uplink reachability and transmission delays based on real-time vehicle positions, RSU coverage, and SINR.
* **Energy & Carbon Awareness:** Integrates with the Electricity Maps API to calculate energy consumption and carbon emissions (gCO2e) during computation and communication.
* **Non-IID Data Partitioning:** Built-in support for Dirichlet distribution-based label skew partitioning for datasets like CIFAR-10 and FashionMNIST.
* **Real-time Monitoring Dashboard:** An interactive Streamlit web UI to dynamically monitor drops, training accuracy, energy metrics, and simulation events.

---

## 2. Prerequisites
This guide assumes you are starting with a clean **Instant Veins VirtualBox VM** (Ubuntu/Debian based). 

* **RAM:** >= 16GB recommended (due to multiple Docker containers and OMNeT++ running simultaneously).
* **CPU:** Multi-core processor.
* **Disk Space:** **80GB or more** (to accommodate Docker images, build artifacts, and simulation logs).

---

## 3. Quick Installation

We provide a one-click setup script to fully automate the deployment process on an Instant Veins VM. It will install Docker, deploy the project directories, compile the necessary C++ modules, and install the Python dependencies.

**Step 1:** Clone the repository to your home directory:
```bash
cd ~ git clone [https://github.com/lukaschare/FedITS-Tool.git](https://github.com/YourUsername/FedITS-Tool.git) cd FedITS-Tool
```

**Step 2:** Run the automated setup script:

```Bash
chmod +x setup_fedits_eng.sh
./setup_fedits_eng.sh
```

_(Note: The script will ask for your `sudo` password to install system packages like Docker. If it adds you to the Docker group for the first time, it will prompt you to log out and log back into the VM.)_

---

## 4. Running the Simulation

The framework is driven by the `run_fedits_scenario_v2.sh` script, which handles launching SUMO, Veins, the Dockerized FL nodes, and the Dashboard concurrently.

Navigate to the deployed orchestrator directory:

```Bash
cd ~/fedits-tool
```

## Option A: GUI Mode (Recommended for observation)

This mode opens separate terminal windows for SUMO-GUI, Veins, Docker Compose, and the Streamlit Dashboard.

```Bash
./run_fedits_scenario_v2.sh gui
```

## Option B: Headless Mode (Recommended for large experiments)

Runs everything in the background without graphical interfaces.

```Bash
./run_fedits_scenario_v2.sh headless \
    --num-veh 100 \
    --m 10 \
    --rounds 10 \
    --scale 10 \
    --dataset cifar10 \
    --partition-scheme dirichlet \
    --dirichlet-alpha 0.3
```

## Option C: Using a Scenario Env File

You can define your variables in an `.env` style file instead of passing long command-line arguments:

```Bash
./run_fedits_scenario_v2.sh headless --scenario scenarios/my_experiment.env
```

---

## 5. Real-Time Dashboard 📊

Once the simulation starts, the Streamlit dashboard will automatically become available at:

👉 **http://localhost:8501**

The dashboard provides insights into:

- **Overview:** Global model validation accuracy, training loss, and aggregator scalar.
    
- **Drop Analysis:** Categorized dropout tracking (`out_of_map`, `out_of_range`, `bad_signal_or_deadline`).
    
- **Carbon Footprint:** Estimated Carbon Emissions (gCO2e) tracking for computation and communication per round.
    

---

## 6. Repository Structure

- `setup_fedits.sh` - Automated deployment script.
- `fedits-tool/docker/` - `docker-compose.yml` for deploying the Flower Server and Clients.
- `fedits-tool/orchestrator/` - The control plane (`orch_core.py`, `orch_service.py`) that bridges Veins and Flower.
- `fedits-tool/fl/` - Contains the custom FL logic (`client.py`, `server.py`) and the `dashboard_streamlit.py`.
- `fedits_veins_rsu/` - The C++ implementation of the RSU Control Server and vehicle communication logic for OMNeT++.

---

