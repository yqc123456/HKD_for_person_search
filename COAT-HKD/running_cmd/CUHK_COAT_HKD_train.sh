	CONFIG=configs/cuhk_sysu.yaml
	TEACHER_MODEL_PRETRAIN=teacherckpt/teacher_cuhk_coat.pth
	OUTPUT_DIR=./logs/
	DATA_ROOT={path_to_CUHK-SYSU_dataset}/dataset
	RESULT_LOG=./logs/result.txt
 
	cd ..; CUDA_VISIBLE_DEVICES=0  nohup python train.py --cfg $CONFIG --reid_ckpt $TEACHER_MODEL_PRETRAIN    \
	lw_kdlogits 0.7 lw_kdfeat 0.0  lw_oim_loss2 0.6 lw_celoss_reid 0.5 lw_kd_sim_kl 100.0 use_reid_head2 False use_reid_head3 True INPUT.BATCH_SIZE_TRAIN 2 MODEL.LOSS.USE_SOFTMAX True  \
	SOLVER.BASE_LR 0.0010 SOLVER.MAX_EPOCHS 25  SOLVER.LR_DECAY_MILESTONES [10,20] MODEL.LOSS.LUT_SIZE 5532 MODEL.LOSS.CQ_SIZE 5000  EVAL_GALLERY_SIZE 100  kdlogits_bidir False \
	INPUT.DATASET CUHK-SYSU INPUT.DATA_ROOT $DATA_ROOT \
	OUTPUT_DIR $OUTPUT_DIR   > $RESULT_LOG 2>&1 &