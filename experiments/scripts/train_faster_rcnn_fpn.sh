set -x
set -e

NET=$1

LOG="experiments/logs/faster_rcnn_fpn/faster_rcnn_end2end_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
python tools/train_net.py \
    --config-file "experiments/cfgs/e2e_faster_rcnn_"${NET}"-FPN.yaml"