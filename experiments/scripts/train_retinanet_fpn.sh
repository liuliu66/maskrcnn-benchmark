set -x
set -e

NET=$1

LOG="experiments/logs/retinanet_fpn/retinanet_end2end_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
python tools/train_net.py \
    --config-file "experiments/cfgs/e2e_retinanet_"${NET}"-FPN.yaml"