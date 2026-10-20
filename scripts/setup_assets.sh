#!/usr/bin/env bash
# Fetch the third-party robot / object assets into third_party/ (run once from the repo root).
#   - MuJoCo Menagerie: Unitree G1 (29 DoF)                        ~40 MB
#   - unitree_ros: G1 + Inspire RH56 hand URDFs and meshes         ~110 MB
#   - YCB google_16k: 005_tomato_soup_can                          ~7 MB download
# third_party/unitree_rl_gym (Unitree's pretrained G1 walking policy, BSD-3) is committed in the repo.
set -euo pipefail
mkdir -p third_party && cd third_party

if [ ! -d mujoco_menagerie ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
  (cd mujoco_menagerie && git sparse-checkout set unitree_g1)
fi

if [ ! -d unitree_ros ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_ros.git
  (cd unitree_ros && git sparse-checkout init --no-cone &&
   printf '/robots/g1_description/meshes/\n/robots/g1_description/inspire_hand/\n' > .git/info/sparse-checkout &&
   git read-tree -mu HEAD)
fi

mkdir -p ycb
for o in 005_tomato_soup_can; do
  if [ ! -d "ycb/$o" ]; then
    curl -sSfL -o "/tmp/$o.tgz" "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/${o}_google_16k.tgz"
    tar xzf "/tmp/$o.tgz" -C ycb && rm -f "/tmp/$o.tgz"
  fi
done
echo "assets ready"
