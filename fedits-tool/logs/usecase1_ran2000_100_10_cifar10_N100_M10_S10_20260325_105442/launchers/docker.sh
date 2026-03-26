#!/usr/bin/env bash
cd "/home/veins/fedits-tool"
sudo docker compose -f docker/docker-compose.yml up --build --scale fl_client=10 2>&1 | tee '/home/veins/fedits-tool/logs/usecase1_ran2000_100_10_cifar10_N100_M10_S10_20260325_105442/docker_20260325_105442.log'
status=$?
echo
echo "[launcher] exit code=$status"
read -n 1 -s -r -p "Press any key to close..."
exit $status
