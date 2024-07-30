
# python -m tent.speech_commands.analysis --dataset_root_path '/root/data/speech_commands' --model 'cnn' \
#     --severity_level 1.0 --output_csv_name 'accuracy_record_cnn_1.0.csv' \
#     --model_weight_file_path './result/speech-commands/tent/pre_train/cnn_weights.pt'

python -m tent.speech_commands.analysis --dataset_root_path '/root/data/speech_commands' --model 'restnet50' \
    --severity_level 1.0 --output_csv_name 'accuracy_record_RestNet50_1.0.csv' \
    --model_weight_file_path './result/speech-commands/tent/pre_train/RestNet50_weight.pt' --normalized