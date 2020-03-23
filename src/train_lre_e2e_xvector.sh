#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh
set -e

stage=6
train_stage=81
remove_egs=false
use_gpu=true

model_tag=segment_attn_blstm_lre

nnet_dir=exp/xvector_${model_tag}_nnet_1a
egs_dir=${nnet_dir}/egs


scripts/run_segment_blstm_attention_xvector.sh --nj 30 --stage ${stage} --train-stage ${train_stage} \
  --data data/lre17_train --nnet-dir ${nnet_dir} \
  --egs-dir ${egs_dir}