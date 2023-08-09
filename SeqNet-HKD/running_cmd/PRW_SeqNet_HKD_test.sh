	CONFIG=configs/prw.yaml
	TEACHER_MODEL_PRETRAIN=teacherckpt/teacher_prw_seqnet.pth
	OUTPUT_DIR=./logs/
	EVAL_CKPT=evalckpt/seqnet_prw_epoch_22.pth
	DATA_ROOT={path_of_PRW_dataset}
	RESULT_LOG=./logs/result.txt
	
	cd ..;CUDA_VISIBLE_DEVICES=0  nohup python train.py --cfg $CONFIG --reid_ckpt $TEACHER_MODEL_PRETRAIN \
	--eval --resume --ckpt $EVAL_CKPT	OUTPUT_DIR $OUTPUT_DIR	INPUT.DATA_ROOT $DATA_ROOT > $RESULT_LOG 2>&1 &
