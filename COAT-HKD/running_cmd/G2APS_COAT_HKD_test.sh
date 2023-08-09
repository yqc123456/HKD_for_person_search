	CONFIG=configs/cuhk_sysu.yaml
	TEACHER_MODEL_PRETRAIN=teacherckpt/teacher_g2aps_coat.pth
	OUTPUT_DIR=./logs/
	EVAL_CKPT=evalckpt/coat_g2aps_epoch_8.pth
	DATA_ROOT={path_of_G2APS_dataset}/ssm
	RESULT_LOG=./logs/result.txt
	
	cd ..; CUDA_VISIBLE_DEVICES=1  nohup python train.py --cfg $CONFIG --reid_ckpt TEACHER_MODEL_PRETRAIN    \
	--eval --resume --ckpt $EVAL_CKPT MODEL.LOSS.LUT_SIZE 2078 MODEL.LOSS.CQ_SIZE 2000  \
	INPUT.DATA_ROOT $DATA_ROOT OUTPUT_DIR $OUTPUT_DIR   > $RESULT_LOG 2>&1 &
	