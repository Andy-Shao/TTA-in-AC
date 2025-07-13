#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech_commands/guassian_noise/0.005-guassian_noise-weak' \
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --adapted_modelF_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelF-bg-1.0-guassian_noise.pt' \
#     --adapted_modelB_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelB-bg-1.0-guassian_noise.pt' \
#     --adapted_modelC_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelC-bg-1.0-guassian_noise.pt' \
#     --normalized --severity_level 0.005 --data_type 'final' --corruption 'gaussian_noise' \
#     --dataset 'speech-commands' --output_csv_name '0.005-guassian_noise_accuracy_record.csv'

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/3.0-doing_the_dishes-weak' \
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --adapted_modelF_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelF-bg-3.0-doing_the_dishes.pt' \
#     --adapted_modelB_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelB-bg-3.0-doing_the_dishes.pt' \
#     --adapted_modelC_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelC-bg-3.0-doing_the_dishes.pt' \
#     --normalized --severity_level 3.0 --data_type 'final' --corruption 'doing_the_dishes' \
#     --dataset 'speech-commands' --output_csv_name 'bg-3.0-doing_the_dishes_accuracy_record.csv' --analyze_STDA

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/3.0-exercise_bike-weak' \
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --adapted_modelF_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelF-bg-3.0-exercise_bike.pt' \
#     --adapted_modelB_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelB-bg-3.0-exercise_bike.pt' \
#     --adapted_modelC_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelC-bg-3.0-exercise_bike.pt' \
#     --normalized --severity_level 3.0 --data_type 'final' --corruption 'exercise_bike' \
#     --dataset 'speech-commands' --output_csv_name 'bg-3.0-exercise_bike_accuracy_record.csv' --analyze_STDA
    
# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/3.0-running_tap-weak' \
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --adapted_modelF_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelF-bg-3.0-running_tap.pt' \
#     --adapted_modelB_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelB-bg-3.0-running_tap.pt' \
#     --adapted_modelC_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelC-bg-3.0-running_tap.pt' \
#     --normalized --severity_level 3.0 --data_type 'final' --corruption 'running_tap' \
#     --dataset 'speech-commands' --analyze_STDA --output_csv_name 'bg-3.0-running_tap_accuracy_record.csv'

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech_commands_purity/running_tap-bg/3.0-running_tap-weak' \
#     --modelF_weight_path './result/speech-commands-purity/CoNMix/pre_train/speech-commands-purity_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands-purity/CoNMix/pre_train/speech-commands-purity_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands-purity/CoNMix/pre_train/speech-commands-purity_best_modelC.pt' \
#     --normalized --severity_level 3.0 --data_type 'final' --corruption 'running_tap' \
#     --dataset 'speech-commands-purity'

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/10.0-doing_the_dishes-weak' \
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --adapted_modelF_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelF-bg-10.0-doing_the_dishes.pt' \
#     --adapted_modelB_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelB-bg-10.0-doing_the_dishes.pt' \
#     --adapted_modelC_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelC-bg-10.0-doing_the_dishes.pt' \
#     --normalized --severity_level 10.0 --data_type 'final' --corruption 'doing_the_dishes' \
#     --dataset 'speech-commands' --output_csv_name 'bg-10.0-doing_the_dishes_accuracy_record.csv' --analyze_STDA

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/10.0-exercise_bike-weak' \
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --adapted_modelF_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelF-bg-10.0-exercise_bike.pt' \
#     --adapted_modelB_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelB-bg-10.0-exercise_bike.pt' \
#     --adapted_modelC_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelC-bg-10.0-exercise_bike.pt' \
#     --normalized --severity_level 10.0 --data_type 'final' --corruption 'exercise_bike' \
#     --dataset 'speech-commands' --output_csv_name 'bg-10.0-exercise_bike_accuracy_record.csv' --analyze_STDA
    
# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/10.0-running_tap-weak' \
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --adapted_modelF_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelF-bg-10.0-running_tap.pt' \
#     --adapted_modelB_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelB-bg-10.0-running_tap.pt' \
#     --adapted_modelC_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelC-bg-10.0-running_tap.pt' \
#     --normalized --severity_level 10.0 --data_type 'final' --corruption 'running_tap' \
#     --dataset 'speech-commands' --analyze_STDA --output_csv_name 'bg-10.0-running_tap_accuracy_record.csv'

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech-commands-random/doing_the_dishes-bg/10.0-doing_the_dishes-weak' \
#     --modelF_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --adapted_modelF_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelF-bg-10.0-doing_the_dishes.pt' \
#     --adapted_modelB_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelB-bg-10.0-doing_the_dishes.pt' \
#     --adapted_modelC_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelC-bg-10.0-doing_the_dishes.pt' \
#     --normalized --severity_level 10.0 --data_type 'final' --corruption 'doing_the_dishes' \
#     --dataset 'speech-commands-random' --output_csv_name 'bg-10.0-doing_the_dishes_accuracy_record.csv' --analyze_STDA

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech-commands-random/exercise_bike-bg/10.0-exercise_bike-weak' \
#     --modelF_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --adapted_modelF_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelF-bg-10.0-exercise_bike.pt' \
#     --adapted_modelB_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelB-bg-10.0-exercise_bike.pt' \
#     --adapted_modelC_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelC-bg-10.0-exercise_bike.pt' \
#     --normalized --severity_level 10.0 --data_type 'final' --corruption 'exercise_bike' \
#     --dataset 'speech-commands-random' --output_csv_name 'bg-10.0-exercise_bike_accuracy_record.csv' --analyze_STDA

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech-commands-random/running_tap-bg/10.0-running_tap-weak' \
#     --modelF_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --adapted_modelF_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelF-bg-10.0-running_tap.pt' \
#     --adapted_modelB_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelB-bg-10.0-running_tap.pt' \
#     --adapted_modelC_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelC-bg-10.0-running_tap.pt' \
#     --normalized --severity_level 10.0 --data_type 'final' --corruption 'running_tap' \
#     --dataset 'speech-commands-random' --analyze_STDA --output_csv_name 'bg-10.0-running_tap_accuracy_record.csv'

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech-commands-random/doing_the_dishes-bg/3.0-doing_the_dishes-weak' \
#     --modelF_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --adapted_modelF_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelF-bg-3.0-doing_the_dishes.pt' \
#     --adapted_modelB_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelB-bg-3.0-doing_the_dishes.pt' \
#     --adapted_modelC_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelC-bg-3.0-doing_the_dishes.pt' \
#     --normalized --severity_level 3.0 --data_type 'final' --corruption 'doing_the_dishes' \
#     --dataset 'speech-commands-random' --output_csv_name 'bg-3.0-doing_the_dishes_accuracy_record.csv' --analyze_STDA

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech-commands-random/exercise_bike-bg/3.0-exercise_bike-weak' \
#     --modelF_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --adapted_modelF_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelF-bg-3.0-exercise_bike.pt' \
#     --adapted_modelB_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelB-bg-3.0-exercise_bike.pt' \
#     --adapted_modelC_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelC-bg-3.0-exercise_bike.pt' \
#     --normalized --severity_level 3.0 --data_type 'final' --corruption 'exercise_bike' \
#     --dataset 'speech-commands-random' --output_csv_name 'bg-3.0-exercise_bike_accuracy_record.csv' --analyze_STDA

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech-commands-random/running_tap-bg/3.0-running_tap-weak' \
#     --modelF_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands-random/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --adapted_modelF_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelF-bg-3.0-running_tap.pt' \
#     --adapted_modelB_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelB-bg-3.0-running_tap.pt' \
#     --adapted_modelC_weight_path './result/speech-commands-random/CoNMix/STDA/speech-commands-random_modelC-bg-3.0-running_tap.pt' \
#     --normalized --severity_level 3.0 --data_type 'final' --corruption 'running_tap' \
#     --dataset 'speech-commands-random' --analyze_STDA --output_csv_name 'bg-3.0-running_tap_accuracy_record.csv'

python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --temporary_path $BASE_PATH'/tmp/speech-commands-numbers/exercise_bike-bg/10.0-exercise_bike-weak' \
    --modelF_weight_path './result/speech-commands-numbers/CoNMix/pre_train/speech-commands-numbers_best_modelF.pt' \
    --modelB_weight_path './result/speech-commands-numbers/CoNMix/pre_train/speech-commands-numbers_best_modelB.pt' \
    --modelC_weight_path './result/speech-commands-numbers/CoNMix/pre_train/speech-commands-numbers_best_modelC.pt' \
    --adapted_modelF_weight_path './result/speech-commands-numbers/CoNMix/STDA/speech-commands-numbers_modelF-bg-10.0-exercise_bike.pt' \
    --adapted_modelB_weight_path './result/speech-commands-numbers/CoNMix/STDA/speech-commands-numbers_modelB-bg-10.0-exercise_bike.pt' \
    --adapted_modelC_weight_path './result/speech-commands-numbers/CoNMix/STDA/speech-commands-numbers_modelC-bg-10.0-exercise_bike.pt' \
    --normalized --severity_level 10.0 --data_type 'final' --corruption 'exercise_bike' \
    --dataset 'speech-commands-numbers' --output_csv_name 'bg-10.0-exercise_bike_accuracy_record.csv' --analyze_STDA