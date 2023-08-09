	CONFIG=configs/prw.yaml 
	TEACHER_MODEL_PRETRAIN=teacherckpt/teacher_prw_coat.pth
	OUTPUT_DIR=./logs
	DATA_ROOT={path_of_PRW_dataset}
	RESULT_LOG=./logs/result.txt
 
	cd ..; CUDA_VISIBLE_DEVICES=0  nohup python train.py --cfg $CONFIG --reid_ckpt $TEACHER_MODEL_PRETRAIN    \
	lw_kdlogits 0.7 lw_kdfeat 0.0  lw_oim_loss2 0.5 lw_celoss_reid 0.5  lw_kd_sim_kl 400.0 use_reid_head2 False use_reid_head3 True INPUT.BATCH_SIZE_TRAIN 2 MODEL.LOSS.USE_SOFTMAX True  \
	SOLVER.BASE_LR 0.0030 SOLVER.MAX_EPOCHS 21   EVAL_GALLERY_SIZE 100 kdlogits_bidir True INPUT.DATA_ROOT $DATA_ROOT  \
	OUTPUT_DIR $OUTPUT_DIR   > $RESULT_LOG 2>&1 &	
	