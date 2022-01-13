INPUT_DATASET_TRAINING_IN='dataset/training/failed_training/S4'
OUTPUT_DATASET_TRAINING_IN='samples_saved/samples_training_in/S4'

INPUT_DATASET_TRAINING_OUT='dataset/training/original_training/S4'
OUTPUT_DATASET_TRAINING_OUT='samples_saved/samples_training_out/S4'

INPUT_DATASET_PREDICT_IN='dataset/predict/S4'
OUTPUT_DATASET_PREDICT_OUT='samples_saved/samples_predict/S4'

python3 main.py CreateSamples --input_file_swarm $INPUT_DATASET_TRAINING_IN --save_file_samples $OUTPUT_DATASET_TRAINING_IN
python3 main.py CreateSamples --input_file_swarm $INPUT_DATASET_TRAINING_OUT --save_file_samples $OUTPUT_DATASET_TRAINING_OUT
python3 main.py CreateSamples --input_file_swarm $INPUT_DATASET_PREDICT_IN --save_file_samples $OUTPUT_DATASET_PREDICT_OUT
