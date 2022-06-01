
scriptDir=$(cd $(dirname $0); pwd)

dir=${scriptDir}/logs
if [ ! -d "$dir" ];then
mkdir $dir 
touch $dir/log.txt
fi

CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 \
tools/train_net.py --config=${scriptDir}/config.yaml --multi_gpu

