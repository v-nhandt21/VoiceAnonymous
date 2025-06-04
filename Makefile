set_up:
	conda install 'ffmpeg<5'

# Password: getdata
get_data:
	bash 01_download_data_model.sh

attack_baseline_pre:
	python inference.py --config configs/eval_pre_attacker.yaml --overwrite "{\"anon_data_suffix\": \"_B3\"}" --force_compute True

attack_baseline_post:
	python run_evaluation.py --config configs/eval_post_attacker.yaml --overwrite "{\"anon_data_suffix\": \"_B3\"}" --force_compute True

attack_mcadams_pre:
	python inference.py --config configs/eval_pre.yaml --overwrite "{\"anon_data_suffix\": \"_mcadams\"}" --force_compute True

attack_mcadams_post:
	python run_evaluation.py --config configs/eval_post.yaml --overwrite "{\"anon_data_suffix\": \"_mcadams\"}" --force_compute True


attack_baseline_inference:
	python inference.py --config configs/eval_pre_attacker.yaml --overwrite "{\"anon_data_suffix\": \"_B3\"}"

attack_speechworld_inference_B3:
	CUDA_VISIBLE_DEVICES=1 python inference.py --config configs/eval_speechworld_B3.yaml --overwrite "{\"anon_data_suffix\": \"_B3\"}"  --force_compute True

attack_speechworld_inference_B4:
	CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,2,3,4,5,6,7,8,9,10 nice -n 10 python inference.py --config configs/eval_speechworld_B4.yaml --overwrite "{\"anon_data_suffix\": \"_B4\"}"  --force_compute True

attack_speechworld_inference_T12-5:
	CUDA_VISIBLE_DEVICES=0 taskset -c 10,11,12,13,14,15,16,17,18,19 nice -n 10 python inference.py --config configs/eval_speechworld_T12-5.yaml --overwrite "{\"anon_data_suffix\": \"_T12-5\"}"  --force_compute True
