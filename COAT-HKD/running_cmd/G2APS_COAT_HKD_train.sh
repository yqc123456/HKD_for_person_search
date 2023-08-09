CONFIG=configs/cuhk_sysu.yaml
TEACHER_MODEL_PRETRAIN=teacherckpt/teacher_g2aps_coat.pth
OUTPUT_DIR=./logs/
DATA_ROOT={path_of_G2APS}/ssm
RESULT_LOG=./logs/result.txt
 
cd ..; CUDA_VISIBLE_DEVICES=0 nohup python train.py --cfg $CONFIG  --reid_ckpt $TEACHER_MODEL_PRETRAIN    \
lw_kdlogits 3.0 lw_kdfeat 0.0  lw_oim_loss2 0.5 lw_celoss_reid 3.0 lw_kd_sim_kl 350.0 \
use_reid_head2 False use_reid_head3 True INPUT.BATCH_SIZE_TRAIN 2 MODEL.LOSS.USE_SOFTMAX True \
SOLVER.LW_RCNN_SOFTMAX_2ND 3.0 SOLVER.BASE_LR 0.0030 SOLVER.MAX_EPOCHS 19  SOLVER.LR_DECAY_MILESTONES [8,11] \
MODEL.LOSS.LUT_SIZE 2078 MODEL.LOSS.CQ_SIZE 2000 SOLVER.LW_RCNN_SOFTMAX_3RD 3.0 kdlogits_bidir True \
OUTPUT_DIR $OUTPUT_DIR > $RESULT_LOG 2>&1 &