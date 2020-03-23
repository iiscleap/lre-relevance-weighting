#!/usr/bin/env bash

export KALDI_ROOT=/state/partition1/softwares/kaldi_sre
export PATH=$PWD/utils/:${KALDI_ROOT}/src/nnet3bin:${KALDI_ROOT}/tools/openfst/bin:${KALDI_ROOT}/tools/sph2pipe_v2.5:$PWD:/home/anandm/others/ffmpeg:$PATH

[[ ! -f ${KALDI_ROOT}/tools/config/common_path.sh ]] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. ${KALDI_ROOT}/tools/config/common_path.sh
export LC_ALL=C
