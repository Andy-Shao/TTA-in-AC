export BASE_PATH=${BASE_PATH:-'\root'}

python -m GALA.speech_commands.tent_analysis --dataset 'speech-commands' \
    --dataset_root_path $BASE_PATH'/data/speech_commands' --model 'restnet50' \
    --severity_level 10.0 --output_csv_name 'speech-commands_RestNet50_background_10.0.csv' \
    --model_weight_file_path './result/speech-commands/tent/pre_train/RestNet50_weight.pt' --normalized \
    --corruptions 'doing_the_dishes' --step 20 --threshold 0.75