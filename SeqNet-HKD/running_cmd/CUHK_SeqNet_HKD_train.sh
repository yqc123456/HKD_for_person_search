	CONFIG=configs/cuhk_sysu.yaml
	TEACHER_MODEL_PRETRAIN=teacherckpt/teacher_cuhk_seqnet.pth
	OUTPUT_DIR=./logs/
	DATA_ROOT={path_to_CUHK-SYSU_dataset}/dataset
	RESULT_LOG=./logs/result.txt
	
	cd ..;CUDA_VISIBLE_DEVICES=0  nohup python train.py --cfg $CONFIG --reid_ckpt $TEACHER_MODEL_PRETRAIN       \
	SOLVER.MAX_EPOCHS 21 SOLVER.BASE_LR 0.0018 lw_kdlogits 1.0 lw_kdfeat 0.0 lw_oim_loss2 1.0 kl_bidir True  lw_kd_sim_kl 50.0 \
	INPUT.BATCH_SIZE_TRAIN 3 INPUT.NUM_WORKERS_TRAIN 3 MODEL.LOSS.LUT_SIZE 5532 MODEL.LOSS.CQ_SIZE 5000 OUTPUT_DIR $OUTPUT_DIR \
	INPUT.DATA_ROOT $DATA_ROOT EVAL_GALLERY_SIZE 100  > $RESULT_LOG 2>&1 &

