	CONFIG=configs/prw.yaml
	DATA_ROOT=./PRW_market_style
	OUTPUT_DIR=./logs
	RESULT_LOG=./logs/result.txt
	cd ..; CUDA_VISIBLE_DEVICES=0  nohup python train.py --dataset PRW  --cfg $CONFIG SOLVER.LW_BOX_REID 7 \
	SOLVER.MAX_EPOCHS 300 SOLVER.BASE_LR 0.0036 eval_interval 10  SOLVER.LR_DECAY_MILESTONES [80,150] OUTPUT_DIR $OUTPUT_DIR \
	duke_path $DATA_ROOT > $RESULT_LOG 2>&1 &
