scriptDir=$(cd $(dirname $0); pwd)
test_type=$1

python tools/test_net.py \
--config ${scriptDir}/config_test.yaml \
--test_model_root ${scriptDir} --test_name ${test_type}
