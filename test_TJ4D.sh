CONFIG_PATH=./projects/SGDet3D/configs/TJ4D-SGDet3D_det3d_2x4_12e.py
CHECKPOINT_PATH=./projects/SGDet3D/checkpoints/FINAL-TJ4D.pth

GPUS="4"
PORT=${PORT:-19500}
CUDA_VISIBLE_DEVICES="0,1,2,3" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/tools_det3d/test.py \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --eval mAP \
    --launcher pytorch ${@:4}
