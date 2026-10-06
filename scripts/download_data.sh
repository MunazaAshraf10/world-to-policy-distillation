#!/usr/bin/env bash
# Occ3D-nuScenes occupancy labels and the OccWorld temporal metadata, verified by checksum.
set -euo pipefail

root="${1:-data}"
mkdir -p "${root}/nuscenes"

fetch() {
  local url="$1" out="$2" sha="$3"
  if [[ ! -f "${out}" ]] || ! echo "${sha}  ${out}" | sha256sum -c --quiet -; then
    curl -fL --retry 5 -o "${out}" "${url}"
    echo "${sha}  ${out}" | sha256sum -c -
  fi
}

infos="https://huggingface.co/synsin0/come/resolve/main/infos"
fetch "${infos}/nuscenes_infos_train_temporal_v3_scene.pkl" \
  "${root}/nuscenes_infos_train_temporal_v3_scene.pkl" \
  fcacfe8f08f9853d723e29e59134c3a506df72108b0a6695261ed1962a111c87
fetch "${infos}/nuscenes_infos_val_temporal_v3_scene.pkl" \
  "${root}/nuscenes_infos_val_temporal_v3_scene.pkl" \
  7e354777b175a06f8705320943f9e143068f145ba2929ba98cc3c416cdab8186
fetch "https://huggingface.co/datasets/RoyYao233/occ3d-nuscenes-transfer/resolve/main/gts.tar.gz" \
  "${root}/gts.tar.gz" \
  0635d1383d8b99cce26a0344ec3ca3c4346f53907a06e82ea525acf7b1abba53

if [[ ! -d "${root}/nuscenes/gts" ]]; then
  tar xzf "${root}/gts.tar.gz" -C "${root}/nuscenes"
fi
