
scriptDir=$(cd $(dirname $0); pwd)

CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config ${scriptDir}/config.yaml
