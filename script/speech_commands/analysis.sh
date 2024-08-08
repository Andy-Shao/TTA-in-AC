export BASE_PATH='/home/andyshao'
# python -m tent.speech_commands.analysis --dataset_root_path $BASE_PATH'/data/speech_commands' --model 'cnn' \
#     --severity_level 0.005 --output_csv_name 'accuracy_record_cnn_gaussian_noise_005.csv' \
#     --model_weight_file_path './result/speech-commands/tent/pre_train/cnn_weights.pt' \
#     --corruptions 'gaussian_noise'

python -m tent.speech_commands.analysis --dataset_root_path $BASE_PATH'/data/speech_commands' --model 'cnn' \
    --severity_level 1.0 --output_csv_name 'accuracy_record_cnn_background_1.0.csv' \
    --model_weight_file_path './result/speech-commands/tent/pre_train/cnn_weights.pt' \
    --corruptions 'doing_the_dishes,exercise_bike,running_tap'

# python -m tent.speech_commands.analysis --dataset_root_path $BASE_PATH'/data/speech_commands' --model 'restnet50' \
#     --severity_level 0.005 --output_csv_name 'accuracy_record_RestNet50_gaussian_noise_005.csv' \
#     --model_weight_file_path './result/speech-commands/tent/pre_train/RestNet50_weight.pt' --normalized \
#     --corruptions 'gaussian_noise'

# python -m tent.speech_commands.analysis --dataset_root_path $BASE_PATH'/data/speech_commands' --model 'restnet50' \
#     --severity_level 1.0 --output_csv_name 'accuracy_record_RestNet50_background_1.0.csv' \
#     --model_weight_file_path './result/speech-commands/tent/pre_train/RestNet50_weight.pt' --normalized \
#     --corruptions 'doing_the_dishes,exercise_bike,running_tap'