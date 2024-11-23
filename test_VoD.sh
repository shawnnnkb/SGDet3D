CONFIG_PATH=./projects/SGDet3D/configs/vod-SGDet3D_det3d_2x4_12e.py
CHECKPOINT_PATH=./projects/SGDet3D/checkpoints/FINAL-VoD.pth
OUTPUT_NAME=vod-SGDet3D
PRED_RESULTS=./tools_det3d/view-of-delft-dataset/pred_results/$OUTPUT_NAME 

GPUS="4"
PORT=${PORT:-1clear9500}
CUDA_VISIBLE_DEVICES="0,1,2,3" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/tools_det3d/test.py \
    --format-only \
    --eval-options submission_prefix=$PRED_RESULTS \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --launcher pytorch ${@:4}

python tools_det3d/view-of-delft-dataset/FINAL_EVAL.py \
--pred_results $PRED_RESULTS
