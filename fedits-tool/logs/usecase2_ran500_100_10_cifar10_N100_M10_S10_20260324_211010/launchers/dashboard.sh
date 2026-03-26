#!/usr/bin/env bash
cd "/home/veins/fedits-tool"
python3 -m streamlit run fl/dashboard_streamlit.py --server.port 8501 2>&1 | tee '/home/veins/fedits-tool/logs/usecase2_ran500_100_10_cifar10_N100_M10_S10_20260324_211010/dashboard_20260324_211010.log'
status=$?
echo
echo "[launcher] exit code=$status"
read -n 1 -s -r -p "Press any key to close..."
exit $status
