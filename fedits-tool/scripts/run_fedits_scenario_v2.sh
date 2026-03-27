#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$PROJECT_ROOT_DEFAULT}"

MODE="headless"              # headless | gui
SUMO_MODE="sumo"             # sumo | sumo-gui
SCENARIO_FILE=""
NUM_VEH=100
M=10
ROUNDS=10
SCALE=10
DATASET="cifar10"
DATASET_TRAIN_SIZE=50000
PARTITION_SCHEME="dirichlet"
DIRICHLET_ALPHA="0.3"
RSU_X=1250
RSU_Y=1250
RSU_R=500
MAP_SIZE=2500
SUMO_FLOW_NUMBER=""
SUMO_FLOW_PERIOD=""
ENABLE_DASHBOARD=0
SUDO_DOCKER=0
BUILD=1
DEBUG_ON_ERRORS=true
CMDENV_EXPRESS_MODE=false
MIN_AVAILABLE_CLIENTS=""
MIN_FIT_CLIENTS=""

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [headless|gui] [options]

Modes:
  headless                  Start SUMO launchd + Veins in background, Docker in current shell.
  gui                       Open real terminal windows for SUMO, Veins, Docker, and Streamlit.

Common options:
  --scenario FILE           Read scenario variables from .env-style file.
  --project-root PATH       Project root. Default: parent of this script.
  --num-veh N               Logical vehicle/client count (NUM_VEH).
  --m N                     Clients selected per round (M).
  --rounds N                Total FL rounds.
  --scale N                 Docker fl_client replica count.
  --dataset NAME            Dataset name. Default: cifar10
  --dataset-train-size N    Training set size. Default: 50000
  --partition-scheme NAME   iid | dirichlet. Default: dirichlet
  --dirichlet-alpha A       Dirichlet alpha. Default: 0.3
  --rsu-x X                 RSU X coordinate.
  --rsu-y Y                 RSU Y coordinate.
  --rsu-r R                 RSU logical coverage radius.
  --map-size N              Map/playground size for Docker + OMNeT. Default: 2500
  --flow-number N           SUMO route flow0 number. Default: NUM_VEH
  --flow-period P           SUMO route flow0 period. Default: keep existing unless scenario sets it
  --sumo sumo|sumo-gui      Launchd mode. gui mode defaults to sumo-gui.
  --dashboard               Start Streamlit dashboard.
  --no-dashboard            Do not start Streamlit dashboard.
  --sudo-docker             Run docker compose with sudo.
  --no-build                Do not pass --build to docker compose.
  --min-available N         fl_server MIN_AVAILABLE_CLIENTS (default: SCALE)
  --min-fit N               fl_server MIN_FIT_CLIENTS (default: SCALE)
  --help                    Show this help.

Scenario file variables supported:
  NUM_VEH, M, ROUNDS, SCALE, DATASET, DATASET_TRAIN_SIZE,
  PARTITION_SCHEME, DIRICHLET_ALPHA,
  RSU_X, RSU_Y, RSU_R, MAP_SIZE,
  SUMO_FLOW_NUMBER, SUMO_FLOW_PERIOD,
  MIN_AVAILABLE_CLIENTS, MIN_FIT_CLIENTS,
  SUMO_MODE, ENABLE_DASHBOARD
USAGE
}

load_scenario_file() {
  local file="$1"
  [[ -f "$file" ]] || { echo "Scenario file not found: $file" >&2; exit 1; }
  while IFS='=' read -r raw_key raw_value; do
    [[ -z "${raw_key// }" ]] && continue
    [[ "$raw_key" =~ ^[[:space:]]*# ]] && continue
    local key="${raw_key## }"
    key="${key%% }"
    local value="$raw_value"
    value="${value%%#*}"
    value="${value#${value%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    case "$key" in
      NUM_VEH) NUM_VEH="$value" ;;
      M) M="$value" ;;
      ROUNDS) ROUNDS="$value" ;;
      SCALE) SCALE="$value" ;;
      DATASET) DATASET="$value" ;;
      DATASET_TRAIN_SIZE) DATASET_TRAIN_SIZE="$value" ;;
      PARTITION_SCHEME) PARTITION_SCHEME="$value" ;;
      DIRICHLET_ALPHA) DIRICHLET_ALPHA="$value" ;;
      RSU_X) RSU_X="$value" ;;
      RSU_Y) RSU_Y="$value" ;;
      RSU_R) RSU_R="$value" ;;
      MAP_SIZE) MAP_SIZE="$value" ;;
      SUMO_FLOW_NUMBER) SUMO_FLOW_NUMBER="$value" ;;
      SUMO_FLOW_PERIOD) SUMO_FLOW_PERIOD="$value" ;;
      MIN_AVAILABLE_CLIENTS) MIN_AVAILABLE_CLIENTS="$value" ;;
      MIN_FIT_CLIENTS) MIN_FIT_CLIENTS="$value" ;;
      SUMO_MODE) SUMO_MODE="$value" ;;
      ENABLE_DASHBOARD) ENABLE_DASHBOARD="$value" ;;
      MODE) MODE="$value" ;;
      '' ) ;;
      *) echo "[warn] ignored scenario key: $key" ;;
    esac
  done < "$file"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    headless|gui)
      MODE="$1"
      [[ "$MODE" == "gui" ]] && SUMO_MODE="sumo-gui" && ENABLE_DASHBOARD=1
      shift
      ;;
    --scenario)
      SCENARIO_FILE="$2"; shift 2 ;;
    --project-root)
      PROJECT_ROOT="$2"; shift 2 ;;
    --num-veh)
      NUM_VEH="$2"; shift 2 ;;
    --m)
      M="$2"; shift 2 ;;
    --rounds)
      ROUNDS="$2"; shift 2 ;;
    --scale)
      SCALE="$2"; shift 2 ;;
    --dataset)
      DATASET="$2"; shift 2 ;;
    --dataset-train-size)
      DATASET_TRAIN_SIZE="$2"; shift 2 ;;
    --partition-scheme)
      PARTITION_SCHEME="$2"; shift 2 ;;
    --dirichlet-alpha)
      DIRICHLET_ALPHA="$2"; shift 2 ;;
    --rsu-x)
      RSU_X="$2"; shift 2 ;;
    --rsu-y)
      RSU_Y="$2"; shift 2 ;;
    --rsu-r)
      RSU_R="$2"; shift 2 ;;
    --map-size)
      MAP_SIZE="$2"; shift 2 ;;
    --flow-number)
      SUMO_FLOW_NUMBER="$2"; shift 2 ;;
    --flow-period)
      SUMO_FLOW_PERIOD="$2"; shift 2 ;;
    --sumo)
      SUMO_MODE="$2"; shift 2 ;;
    --dashboard)
      ENABLE_DASHBOARD=1; shift ;;
    --no-dashboard)
      ENABLE_DASHBOARD=0; shift ;;
    --sudo-docker)
      SUDO_DOCKER=1; shift ;;
    --no-build)
      BUILD=0; shift ;;
    --min-available)
      MIN_AVAILABLE_CLIENTS="$2"; shift 2 ;;
    --min-fit)
      MIN_FIT_CLIENTS="$2"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "$SCENARIO_FILE" ]]; then
  load_scenario_file "$SCENARIO_FILE"
fi

if [[ -z "$SUMO_FLOW_NUMBER" ]]; then
  SUMO_FLOW_NUMBER="$NUM_VEH"
fi
if [[ -z "$MIN_AVAILABLE_CLIENTS" ]]; then
  MIN_AVAILABLE_CLIENTS="$SCALE"
fi
if [[ -z "$MIN_FIT_CLIENTS" ]]; then
  MIN_FIT_CLIENTS="$SCALE"
fi


# Split the path into two parts: TOOL_ROOT is responsible for Docker/dashboard, and SIM_ROOT is responsible for Veins simulation
TOOL_ROOT="$PROJECT_ROOT"
SIM_ROOT="/home/veins/src/veins/examples/fedits_veins_rsu"

COMPOSE_FILE="$TOOL_ROOT/docker/docker-compose.yml"
OMNETPP_INI="$SIM_ROOT/omnetpp.ini"
ROUTE_FILE="$SIM_ROOT/erlangen.rou.xml"
VEINS_BIN="$SIM_ROOT/out/clang-release/fedits_veins_rsu"


# COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"
# OMNETPP_INI="$PROJECT_ROOT/omnetpp.ini"
# ROUTE_FILE="$PROJECT_ROOT/erlangen.rou.xml"
# VEINS_BIN="$PROJECT_ROOT/out/clang-release/fedits_veins_rsu"


LOGS_ROOT="$PROJECT_ROOT/logs"
STATE_DIR="$PROJECT_ROOT/.run_state"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
SCENARIO_STEM="custom"
if [[ -n "$SCENARIO_FILE" ]]; then
  SCENARIO_STEM="$(basename "$SCENARIO_FILE" | sed 's/\.[^.]*$//')"
fi
RUN_NAME="${SCENARIO_STEM}_${DATASET}_N${NUM_VEH}_M${M}_S${SCALE}_${TIMESTAMP}"
LOG_DIR="$LOGS_ROOT/$RUN_NAME"
LAUNCHER_DIR="$LOG_DIR/launchers"
STATE_FILE="$STATE_DIR/current_run.env"

# ==========================================
# [New Addition] Pre-download datasets to avoid download conflicts caused by concurrent operations of multiple containers
# ==========================================
echo ">>> [Pre-flight Check] Verifying environment and dataset: $DATASET ..."

# 1. Automatically detect and install torchvision (if it does not exist)
if ! python3 -c "import torchvision" &> /dev/null; then
    echo ">>> [Auto-Install] 'torchvision' not found. Installing it automatically..."
    # It is recommended to use python3 -m pip install --user to avoid system permission issues
    python3 -m pip install torchvision --user || pip3 install torchvision
fi

# 2. Call Python to download the dataset (torchvision is definitely available at this time)
python3 -c "
import os
import sys
import torchvision

data_dir = os.path.join('$PROJECT_ROOT', 'data')
os.makedirs(data_dir, exist_ok=True)

dataset_name = '$DATASET'.lower()
if dataset_name == 'cifar10':
    print('>>> Checking CIFAR10 in', data_dir)
    torchvision.datasets.CIFAR10(root=data_dir, download=True)
elif dataset_name == 'fashionmnist':
    print('>>> Checking FashionMNIST in', data_dir)
    torchvision.datasets.FashionMNIST(root=data_dir, download=True)
else:
    print(f'>>> [warn] Unknown dataset {dataset_name}, skipping pre-download.')
"
echo ">>> Dataset check complete."
# ==========================================

mkdir -p "$LOG_DIR" "$LAUNCHER_DIR" "$STATE_DIR"

for f in "$COMPOSE_FILE" "$OMNETPP_INI" "$ROUTE_FILE" "$VEINS_BIN"; do
  if [[ ! -e "$f" ]]; then
    echo "Required path not found: $f" >&2
    exit 1
  fi
done

cp "$COMPOSE_FILE" "$LOG_DIR/docker-compose.yml.bak"
cp "$OMNETPP_INI" "$LOG_DIR/omnetpp.ini.bak"
cp "$ROUTE_FILE" "$LOG_DIR/erlangen.rou.xml.bak"

python3 - "$COMPOSE_FILE" "$OMNETPP_INI" "$ROUTE_FILE" \
  "$NUM_VEH" "$M" "$ROUNDS" "$SCALE" "$DATASET" "$DATASET_TRAIN_SIZE" \
  "$PARTITION_SCHEME" "$DIRICHLET_ALPHA" "$RSU_X" "$RSU_Y" "$RSU_R" \
  "$MAP_SIZE" "$MIN_AVAILABLE_CLIENTS" "$MIN_FIT_CLIENTS" "$SUMO_FLOW_NUMBER" "$SUMO_FLOW_PERIOD" <<'PY'
from pathlib import Path
import re
import sys

compose_p, omnet_p, route_p = map(Path, sys.argv[1:4])
num_veh, m, rounds, scale = sys.argv[4:8]
dataset, dataset_train_size = sys.argv[8:10]
partition_scheme, dirichlet_alpha = sys.argv[10:12]
rsu_x, rsu_y, rsu_r = sys.argv[12:15]
map_size, min_available, min_fit = sys.argv[15:18]
flow_number, flow_period = sys.argv[18:20]


def replace_key(text: str, key: str, value: str, count_at_least: int = 1) -> str:
    pattern = rf'^(\s*{re.escape(key)}(?:\s*:\s*|\s*=\s*)).*$'
    new_text, count = re.subn(pattern, rf'\g<1>{value}', text, flags=re.MULTILINE)
    if count < count_at_least:
        raise SystemExit(f"Failed to update '{key}' in text; found {count}, expected >= {count_at_least}")
    return new_text

compose = compose_p.read_text(encoding='utf-8')
compose = replace_key(compose, 'NUM_VEH', num_veh)
compose = replace_key(compose, 'M', m)
compose = replace_key(compose, 'ROUNDS', rounds, count_at_least=2)
compose = replace_key(compose, 'DATASET', dataset, count_at_least=3)
compose = replace_key(compose, 'DATASET_TRAIN_SIZE', dataset_train_size)
compose = replace_key(compose, 'PARTITION_SCHEME', partition_scheme)
compose = replace_key(compose, 'DIRICHLET_ALPHA', dirichlet_alpha)
compose = replace_key(compose, 'RSU_X_M', rsu_x)
compose = replace_key(compose, 'RSU_Y_M', rsu_y)
compose = replace_key(compose, 'RSU_R_M', rsu_r)
compose = replace_key(compose, 'MAP_SIZE_M', map_size)

if re.search(r'^(\s*MIN_AVAILABLE_CLIENTS:\s*).*$' , compose, flags=re.MULTILINE):
    compose = replace_key(compose, 'MIN_AVAILABLE_CLIENTS', min_available)
else:
    compose = re.sub(
        r'^(\s*ORCH_URL:\s*.*)$',
        rf'\1\n      MIN_AVAILABLE_CLIENTS: {min_available}',
        compose,
        count=1,
        flags=re.MULTILINE,
    )
if re.search(r'^(\s*MIN_FIT_CLIENTS:\s*).*$' , compose, flags=re.MULTILINE):
    compose = replace_key(compose, 'MIN_FIT_CLIENTS', min_fit)
else:
    compose = re.sub(
        r'^(\s*MIN_AVAILABLE_CLIENTS:\s*.*)$',
        rf'\1\n      MIN_FIT_CLIENTS: {min_fit}',
        compose,
        count=1,
        flags=re.MULTILINE,
    )
compose_p.write_text(compose, encoding='utf-8')

omnet = omnet_p.read_text(encoding='utf-8')
omnet = replace_key(omnet, '*.control.rsuX', rsu_x, count_at_least=1)
omnet = replace_key(omnet, '*.control.rsuY', rsu_y, count_at_least=1)
omnet = replace_key(omnet, '*.control.rsuR', rsu_r, count_at_least=1)
omnet = replace_key(omnet, '*.rsu[0].mobility.x', rsu_x, count_at_least=1)
omnet = replace_key(omnet, '*.rsu[0].mobility.y', rsu_y, count_at_least=1)
omnet = replace_key(omnet, '*.playgroundSizeX', f'{map_size}m', count_at_least=1)
omnet = replace_key(omnet, '*.playgroundSizeY', f'{map_size}m', count_at_least=1)
omnet_p.write_text(omnet, encoding='utf-8')

route = route_p.read_text(encoding='utf-8')
route, count = re.subn(r'(<flow\s+id="flow0"[^>]*\bnumber=")\d+(")', rf'\g<1>{flow_number}\2', route, count=1)
if count != 1:
    raise SystemExit('Failed to update flow0 number in erlangen.rou.xml')
if flow_period:
    route, count = re.subn(r'(<flow\s+id="flow0"[^>]*\bperiod=")([^"]+)(")', rf'\g<1>{flow_period}\3', route, count=1)
    if count != 1:
        raise SystemExit('Failed to update flow0 period in erlangen.rou.xml')
route_p.write_text(route, encoding='utf-8')
PY

DOCKER_BIN="docker"
if [[ "$SUDO_DOCKER" == "1" ]]; then
  DOCKER_BIN="sudo docker"
fi
BUILD_ARG=""
if [[ "$BUILD" == "1" ]]; then
  BUILD_ARG="--build"
fi

SUMO_LOG="$LOG_DIR/veins_launchd_${TIMESTAMP}.log"
VEINS_LOG="$LOG_DIR/veins_run_${TIMESTAMP}.log"
DOCKER_LOG="$LOG_DIR/docker_${TIMESTAMP}.log"
DASHBOARD_LOG="$LOG_DIR/dashboard_${TIMESTAMP}.log"
INFO_LOG="$LOG_DIR/run_info.txt"

cat > "$INFO_LOG" <<INFO
run_name=$RUN_NAME
project_root=$PROJECT_ROOT
mode=$MODE
scenario_file=$SCENARIO_FILE
sumo_mode=$SUMO_MODE
dataset=$DATASET
dataset_train_size=$DATASET_TRAIN_SIZE
num_veh=$NUM_VEH
m=$M
rounds=$ROUNDS
scale=$SCALE
partition_scheme=$PARTITION_SCHEME
dirichlet_alpha=$DIRICHLET_ALPHA
rsu_x=$RSU_X
rsu_y=$RSU_Y
rsu_r=$RSU_R
map_size=$MAP_SIZE
sumo_flow_number=$SUMO_FLOW_NUMBER
sumo_flow_period=$SUMO_FLOW_PERIOD
min_available_clients=$MIN_AVAILABLE_CLIENTS
min_fit_clients=$MIN_FIT_CLIENTS
logs=$LOG_DIR
INFO

cat > "$STATE_FILE" <<STATE
RUN_NAME='$RUN_NAME'
PROJECT_ROOT='$PROJECT_ROOT'
LOG_DIR='$LOG_DIR'
DOCKER_BIN='$DOCKER_BIN'
STATE

make_launcher() {
  local path="$1"
  local cwd="$2"
  local cmd="$3"
  cat > "$path" <<LAUNCH
#!/usr/bin/env bash
cd "$cwd"
$cmd
status=\$?
echo
echo "[launcher] exit code=\$status"
read -n 1 -s -r -p "Press any key to close..."
exit \$status
LAUNCH
  chmod +x "$path"
}

SUMO_CMD="veins_launchd -vv -c $SUMO_MODE 2>&1 | tee '$SUMO_LOG'"
VEINS_CMD="./out/clang-release/fedits_veins_rsu -u Cmdenv -c Default -n .:../../src/veins --debug-on-errors=$DEBUG_ON_ERRORS --cmdenv-express-mode=$CMDENV_EXPRESS_MODE 2>&1 | tee '$VEINS_LOG'"
DOCKER_CMD="$DOCKER_BIN compose -f docker/docker-compose.yml up $BUILD_ARG --scale fl_client=$SCALE 2>&1 | tee '$DOCKER_LOG'"
DASHBOARD_CMD="python3 -m streamlit run fl/dashboard_streamlit.py --server.port 8501 2>&1 | tee '$DASHBOARD_LOG'"

# make_launcher "$LAUNCHER_DIR/sumo.sh" "$PROJECT_ROOT" "$SUMO_CMD"
# make_launcher "$LAUNCHER_DIR/veins.sh" "$PROJECT_ROOT" "$VEINS_CMD"
# make_launcher "$LAUNCHER_DIR/docker.sh" "$PROJECT_ROOT" "$DOCKER_CMD"
# make_launcher "$LAUNCHER_DIR/dashboard.sh" "$PROJECT_ROOT" "$DASHBOARD_CMD"

make_launcher "$LAUNCHER_DIR/sumo.sh" "$SIM_ROOT" "$SUMO_CMD"
make_launcher "$LAUNCHER_DIR/veins.sh" "$SIM_ROOT" "$VEINS_CMD"
make_launcher "$LAUNCHER_DIR/docker.sh" "$TOOL_ROOT" "$DOCKER_CMD"
make_launcher "$LAUNCHER_DIR/dashboard.sh" "$TOOL_ROOT" "$DASHBOARD_CMD"

detect_terminal() {
  for term in gnome-terminal x-terminal-emulator xfce4-terminal konsole xterm; do
    if command -v "$term" >/dev/null 2>&1; then
      echo "$term"
      return 0
    fi
  done
  return 1
}

open_terminal_script() {
  local title="$1"
  local script_path="$2"
  local term
  term="$(detect_terminal)" || {
    echo "No supported terminal emulator found for gui mode." >&2
    exit 1
  }
  case "$term" in
    gnome-terminal)
      gnome-terminal --title="$title" -- bash -lc "$script_path"
      ;;
    x-terminal-emulator)
      x-terminal-emulator -T "$title" -e bash -lc "$script_path"
      ;;
    xfce4-terminal)
      xfce4-terminal --title="$title" --command="bash -lc '$script_path'"
      ;;
    konsole)
      konsole --new-tab -p tabtitle="$title" -e bash -lc "$script_path"
      ;;
    xterm)
      xterm -T "$title" -e bash -lc "$script_path"
      ;;
  esac
}

cleanup_hint() {
  cat <<HINT

Started run: $RUN_NAME
Logs: $LOG_DIR
Stop command:
  $SCRIPT_DIR/stop_fedits_scenario.sh --project-root "$PROJECT_ROOT"
HINT
}

if [[ "$MODE" == "headless" ]]; then
  echo "[start] headless mode"
  echo "[start] logs -> $LOG_DIR"

  nohup bash -lc "cd '$SIM_ROOT' && $SUMO_CMD" >/dev/null 2>&1 &
  SUMO_PID=$!
  echo "SUMO_LAUNCHD_PID='$SUMO_PID'" >> "$STATE_FILE"
  sleep 2

  nohup bash -lc "cd '$SIM_ROOT' && $VEINS_CMD" >/dev/null 2>&1 &
  VEINS_PID=$!
  echo "VEINS_PID='$VEINS_PID'" >> "$STATE_FILE"
  sleep 3

  if [[ "$ENABLE_DASHBOARD" == "1" ]]; then
    nohup bash -lc "cd '$TOOL_ROOT' && $DASHBOARD_CMD" >/dev/null 2>&1 &
    DASHBOARD_PID=$!
    echo "DASHBOARD_PID='$DASHBOARD_PID'" >> "$STATE_FILE"
    echo "[start] dashboard -> http://localhost:8501"
  fi

  cleanup_hint
  echo "[start] docker running in current shell ..."
  cd "$PROJECT_ROOT"
  bash -lc "$DOCKER_CMD"
else
  echo "[start] gui mode"
  echo "[start] logs -> $LOG_DIR"
  open_terminal_script "SUMO launchd" "$LAUNCHER_DIR/sumo.sh"
  sleep 2
  open_terminal_script "Veins" "$LAUNCHER_DIR/veins.sh"
  sleep 2
  if [[ "$ENABLE_DASHBOARD" == "1" ]]; then
    open_terminal_script "Streamlit Dashboard" "$LAUNCHER_DIR/dashboard.sh"
    echo "[start] dashboard -> http://localhost:8501"
  fi
  sleep 2
  open_terminal_script "Docker Compose" "$LAUNCHER_DIR/docker.sh"
  cleanup_hint
fi
