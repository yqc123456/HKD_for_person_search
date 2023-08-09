	CONFIG=configs/cuhk_sysu.yaml
	TEACHER_MODEL_PRETRAIN=teacherckpt/teacher_g2aps_seqnet.pth
	OUTPUT_DIR=./logs/
	DATA_ROOT={path_of_G2APS}/ssm
	RESULT_LOG=./logs/result.txt

	cd ..;CUDA_VISIBLE_DEVICES=0 nohup python train.py --cfg $CONFIG --reid_ckpt $TEACHER_MODEL_PRETRAIN \
	SOLVER.MAX_EPOCHS 21 SOLVER.BASE_LR 0.001 lw_kdlogits 1.0 lw_kdfeat 0.0 lw_oim_loss2 1.0 lw_kd_sim_kl 200.0 \
	OUTPUT_DIR $OUTPUT_DIR INPUT.BATCH_SIZE_TRAIN 2 INPUT.NUM_WORKERS_TRAIN 2 MODEL.LOSS.LUT_SIZE 2078 MODEL.LOSS.CQ_SIZE 2000 \
	INPUT.DATA_ROOT $DATA_ROOT > $RESULT_LOG 2>&1 &
