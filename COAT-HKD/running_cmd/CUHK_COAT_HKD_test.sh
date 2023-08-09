	CONFIG=configs/cuhk_sysu.yaml
	TEACHER_MODEL_PRETRAIN=teacherckpt/teacher_cuhk_coat.pth
	OUTPUT_DIR=./logs/
	EVAL_CKPT=evalckpt/coat_cuhk_epoch_13.pth
	DATA_ROOT={path_of_CUHK-SYSU_dataset}/dataset
	RESULT_LOG=./logs/result.txt
	
	cd ..; CUDA_VISIBLE_DEVICES=0  nohup python train.py --cfg $CONFIG --reid_ckpt $TEACHER_MODEL_PRETRAIN   \
	--eval --resume --ckpt $EVAL_CKPT MODEL.LOSS.LUT_SIZE 5532 MODEL.LOSS.CQ_SIZE 5000  EVAL_GALLERY_SIZE 100 \
	INPUT.DATASET CUHK-SYSU INPUT.DATA_ROOT $DATA_ROOT OUTPUT_DIR $OUTPUT_DIR   > $RESULT_LOG 2>&1 &
	