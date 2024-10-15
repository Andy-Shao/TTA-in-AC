export BASE_PATH=${BASE_PATH:-'\root'}

# python -m GALA.tent_analysis --model_weight_file_path './result/audio-mnist/tent/pre_train/RestNet50_batch_weight.pt' \
#     --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --severity_level 10.0 --model 'restnet50' \
#     --output_csv_name 'RestNet50_batch_accuracy_record_bg-rand-10.0.csv' \
#     --corruptions 'doing_the_dishes,exercise_bike,running_tap' \
#     --background_root_path $BASE_PATH'/data/speech_commands' --rand_bg --step 20

python -m GALA.tent_analysis --model_weight_file_path './result/audio-mnist/tent/pre_train/RestNet50_batch_weight.pt' \
    --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --severity_level 3.0 --model 'restnet50' \
    --output_csv_name 'RestNet50_batch_accuracy_record_bg-rand-3.0.csv' \
    --corruptions 'doing_the_dishes,exercise_bike,running_tap' \
    --background_root_path $BASE_PATH'/data/speech_commands' --rand_bg --step 20