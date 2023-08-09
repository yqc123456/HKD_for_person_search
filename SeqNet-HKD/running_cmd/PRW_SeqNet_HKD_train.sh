	CONFIG=configs/prw.yaml 
	TEACHER_MODEL_PRETRAIN=teacherckpt/teacher_prw_seqnet.pth
	OUTPUT_DIR=./logs/
	DATA_ROOT={path_of_PRW_dataset}
	RESULT_LOG=./logs/result.txt
	
	cd ..;CUDA_VISIBLE_DEVICES=0   python train.py --cfg $CONFIG --reid_ckpt $TEACHER_MODEL_PRETRAIN \
	SOLVER.MAX_EPOCHS 25 SOLVER.BASE_LR 0.0018 lw_kdlogits 3.0 lw_kdfeat 0.0 lw_oim_loss2 1.0 lw_kd_sim_kl 150.0 tempsimkl 2.0 simkl_bidir False \
	OUTPUT_DIR $OUTPUT_DIR INPUT.BATCH_SIZE_TRAIN 4 INPUT.NUM_WORKERS_TRAIN 4 \
	INPUT.DATA_ROOT $DATA_ROOT > $RESULT_LOG 2>&1 & 