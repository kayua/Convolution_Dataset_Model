INPUT_DATASET_TRAINING_IN='dataset/training/failed_training/S4'
OUTPUT_DATASET_TRAINING_IN='samples_saved/samples_training_in/S4'

INPUT_DATASET_TRAINING_OUT='dataset/training/original_training/S4'
OUTPUT_DATASET_TRAINING_OUT='samples_saved/samples_training_out/S4'

INPUT_DATASET_PREDICT_IN='dataset/predict/S4'
OUTPUT_DATASET_PREDICT_OUT='samples_saved/samples_predict/S4'

SAVE_NEURAL_MODEL='models_saved/model'

python3 main.py CreateSamples --input_file_swarm $INPUT_DATASET_TRAINING_IN --save_file_samples $OUTPUT_DATASET_TRAINING_IN
python3 main.py CreateSamples --input_file_swarm $INPUT_DATASET_TRAINING_OUT --save_file_samples $OUTPUT_DATASET_TRAINING_OUT
python3 main.py CreateSamples --input_file_swarm $INPUT_DATASET_PREDICT_IN --save_file_samples $OUTPUT_DATASET_PREDICT_OUT

python3 main.py Training --load_samples_training_in $OUTPUT_DATASET_TRAINING_IN --load_samples_training_out $OUTPUT_DATASET_TRAINING_OUT --save_model $SAVE_NEURAL_MODEL

python3 main.py Predict --input_predict $OUTPUT_DATASET_PREDICT_OUT --output_predict $OUTPUT_DATASET_TRAINING_OUT --load_model $SAVE_NEURAL_MODEL