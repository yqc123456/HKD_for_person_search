	CONFIG=configs/cuhk_sysu.yaml
	TEACHER_MODEL_PRETRAIN=teacherckpt/teacher_g2aps_seqnet.pth
	EVAL_CKPT=evalckpt/seqnet_g2aps_epoch_16.pth
	OUTPUT_DIR=./logs/
	DATA_ROOT={path_of_G2APS}/ssm
	RESULT_LOG=./logs/result.txt
	
	cd ..;CUDA_VISIBLE_DEVICES=0 nohup python train.py --cfg $CONFIG --reid_ckpt $TEACHER_MODEL_PRETRAIN \
	--eval --resume --ckpt $EVAL_CKPT  OUTPUT_DIR $OUTPUT_DIR MODEL.LOSS.LUT_SIZE 2078 MODEL.LOSS.CQ_SIZE 2000 \
	INPUT.DATA_ROOT $DATA_ROOT > $RESULT_LOG 2>&1 &
	