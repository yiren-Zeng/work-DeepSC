#!/usr/bin/env bash
set -euo pipefail

root=/workspace/yi/work/RAQ-VAE-main
output_root="$root/experiments/eval/snr_sweep_awgn_rician_k10_seeds42_51_20261004"
script="$root/scripts/test_seq2seq_ds8_dualk_channel.sh"

bottom_ks=(8 64 4 16)
top_ks=(16 16 16 32)
ldpc_rates=(1/2 7/16 1/2 7/16)
modulations=(QPSK 16QAM QPSK 16QAM)
ratio_denominators=(48 48 64 64)
labels=(
  raq_vae_1over48_qpsk_k8_16
  raq_vae_1over48_16qam_k64_16
  raq_vae_1over64_qpsk_k4_16
  raq_vae_1over64_16qam_k16_32
)
snrs=(-10 -8 -6 -4 -2 0)

mkdir -p -- "$output_root"

for snr in "${snrs[@]}"; do
  pids=()
  for index in "${!labels[@]}"; do
    label="${labels[$index]}"
    snr_label="${snr/-/m}"
    case_output="$output_root/${label}_snr_${snr_label}"
    log="$output_root/${label}_snr_${snr_label}.log"
    env \
      GPU_ID=3 \
      CHANNEL_TYPES="awgn rician" \
      RICIAN_K_FACTOR=10 \
      CHANNEL_SEEDS="42 43 44 45 46 47 48 49 50 51" \
      BOTTOM_K="${bottom_ks[$index]}" \
      TOP_K="${top_ks[$index]}" \
      LDPC_RATE="${ldpc_rates[$index]}" \
      MODULATION="${modulations[$index]}" \
      SNR="$snr" \
      RATIO_DENOMINATOR="${ratio_denominators[$index]}" \
      OUTPUT_ROOT="$case_output" \
      bash "$script" >"$log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

touch "$output_root/COMPLETE"
