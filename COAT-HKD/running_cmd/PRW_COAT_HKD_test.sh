	CONFIG=configs/prw.yaml
	TEACHER_MODEL_PRETRAIN=teacherckpt/teacher_prw_coat.pth
	OUTPUT_DIR=./logs/
	EVAL_CKPT=evalckpt/coat_prw_epoch_12.pth
	DATA_ROOT={path_of_PRW_dataset}
	RESULT_LOG=./logs/result.txt
	
	cd ..; CUDA_VISIBLE_DEVICES=1  nohup python train.py --cfg $CONFIG --reid_ckpt $TEACHER_MODEL_PRETRAIN    \
	--eval --resume --ckpt $EVAL_CKPT EVAL_GALLERY_SIZE 100 INPUT.DATA_ROOT $DATA_ROOT   \
	OUTPUT_DIR $OUTPUT_DIR   > $RESULT_LOG 2>&1 &
