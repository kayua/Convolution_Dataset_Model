INPUT_DATASET_TRAINING_IN='dataset/training/failed_training/S4'
OUTPUT_DATASET_TRAINING_IN='samples_saved/samples_training_in/S4'

INPUT_DATASET_TRAINING_OUT='dataset/training/original_training/S4'
OUTPUT_DATASET_TRAINING_OUT='samples_saved/samples_training_out/S4'

INPUT_DATASET_PREDICT_IN='dataset/predict/S4'
OUTPUT_DATASET_PREDICT_OUT='samples_saved/samples_predict/S4'

SAVE_NEURAL_MODEL='models_saved/model'

INPUT_ANALYSE_ORIGINAL='dataset/original/S4'
INPUT_ANALYSE_CORRECTED='dataset/corrected/S4'
INPUT_ANALYSE_FAILED='dataset/failed/S4'
RESULT_METRICS='results/results.txt'

#python3 main.py CreateSamples --input_file_swarm $INPUT_DATASET_TRAINING_IN --save_file_samples $OUTPUT_DATASET_TRAINING_IN
#python3 main.py CreateSamples --input_file_swarm $INPUT_DATASET_TRAINING_OUT --save_file_samples $OUTPUT_DATASET_TRAINING_OUT
#python3 main.py CreateSamples --input_file_swarm $INPUT_DATASET_PREDICT_IN --save_file_samples $OUTPUT_DATASET_PREDICT_OUT
python3 main.py Training --load_samples_training_in $OUTPUT_DATASET_TRAINING_IN --load_samples_training_out $OUTPUT_DATASET_TRAINING_OUT --save_model $SAVE_NEURAL_MODEL
#python3 main.py Predict --input_predict $OUTPUT_DATASET_PREDICT_OUT --output_predict $INPUT_ANALYSE_CORRECTED --load_model $SAVE_NEURAL_MODEL
python3 main.py Analyse --file_original $INPUT_ANALYSE_ORIGINAL --file_corrected $INPUT_ANALYSE_CORRECTED --file_failed $INPUT_ANALYSE_FAILED --file_analyse $RESULT_METRICS
