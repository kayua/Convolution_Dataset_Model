# Sound Classification with neural network



## Steps to Install:

1. Installation external dependencies
    - Install OpenCV https://github.com/opencv/opencv/tree/4.5.4
    - Install Tensorflow https://github.com/tensorflow/tensorflow
    - sudo apt install ffmpeg
    
2. Installation of application and internal dependencies
    - git clone https://github.com/kayua/Mosquito-Tensorflow
    - pip install -r requirements.txt
    
3. Test installation:
    - python3 main.py -h


## Feature extraction

This API has a great diversity of scales, chromas, transforms and spectrograms for creating training samples for neural networks. The different features allow use in classification systems, source separation, audio treatment and sound modification, among others.

Below are some of the features available for use:

<table>
    <tbody>
        <tr>
            <th width="20%">Chroma Constant</th>
            <th width="20%">Chroma Normalized</th>
            <th width="20%">Chroma Power</th>
            <th width="20%">Tonal Centroids</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/MosquitoClassification-TensorFlow/blob/master/spectrograms/chroma_constant/aedes_aegypti17_chroma_constant.png" alt="2018-06-04 4 33 16" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/MosquitoClassification-TensorFlow/blob/master/spectrograms/chroma_normalized/aedes_aegypti7_chroma_normalized.png" alt="2018-06-04 4 40 06" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/MosquitoClassification-TensorFlow/blob/master/spectrograms/chroma_power/aedes_aegypti7_chroma_power.png" alt="2018-06-04 4 43 02" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/MosquitoClassification-TensorFlow/blob/master/spectrograms/tonal_centroids/aedes_aegypti7_tonal_centroids.png" alt="2018-06-04 4 47 40" style="max-width:100%;"></td>
        </tr>


</table>

<table>
    <tbody>
        <tr>
            <th width="20%">Chroma Constant</th>
            <th width="20%">Chroma Normalized</th>
            <th width="20%">Chroma Power</th>
            <th width="20%">Tonal Centroids</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/MosquitoClassification-TensorFlow/blob/master/spectrograms/chroma_constant/aedes_aegypti17_chroma_constant.png" alt="2018-06-04 4 33 16" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/MosquitoClassification-TensorFlow/blob/master/spectrograms/chroma_normalized/aedes_aegypti7_chroma_normalized.png" alt="2018-06-04 4 40 06" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/MosquitoClassification-TensorFlow/blob/master/spectrograms/chroma_power/aedes_aegypti7_chroma_power.png" alt="2018-06-04 4 43 02" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/MosquitoClassification-TensorFlow/blob/master/spectrograms/tonal_centroids/aedes_aegypti7_tonal_centroids.png" alt="2018-06-04 4 47 40" style="max-width:100%;"></td>
        </tr>


</table>



## Datasets available:
This table contains five dataset available for download.

|Dataset Name| Dataset Size | Data Description  | Repository Available
|---|------|-----|------- 
| test_1.tar.gz | 144 MB | Ae. Aegypti Male vs Noise| <a href="https://drive.google.com/file/d/1Qv0FCV5XW-K7D4AzSTTDA99QAq4CnnEZ/view?usp=sharing" target="_blank">Link for test_1.tar.gz</a>
| test_2.tar.gz | 93 MB |  Ae. Aegypti Female vs Noise | <a href="https://drive.google.com/file/d/1DNBpXlePvcNBhbclz3X1xr9de9H9S8L9/view?usp=sharing" target="_blank">Link for test_2.tar.gz</a>
| test_3.tar.gz | 181 MB | Ae. Aegypti vs Noise | <a href="https://drive.google.com/file/d/14-bplhwmxQROsrh1JyH_s33tZC17ZbAi/view?usp=sharing" target="_blank">Link for test_3.tar.gz</a>
| test_4.tar.gz | 191 MB | Ae. Aegypti vs Noise, Other species | <a href="https://drive.google.com/file/d/1xw3Sdo_2hGp3CkmRTu8yXfS6N-rdLStO/view?usp=sharing" target="_blank">Link for test_4.tar.gz</a>
| test_5.tar.gz | 187 MB | Ae. Aegypti vs Noise, Other species  (Balanced) | <a href="https://drive.google.com/file/d/1VBVp4w2_VjlgxDHSTdrfcGxd_sYiRryB/view?usp=sharing" target="_blank">Link for test_4.tar.gz</a>

## Basic commands:

### Convert (python3 main.py Convert):
      Convert Sound
      --input_path_convert PATH_INPUT_3GP --output_path_convert PATH_TEMP_WAV

### Visual Audio Analyser(Optional) (python3 main.py Analyse):
      Plot Spectrogram
      --plot_spectrogram --input_path_sound_plot PATH_TEMP_WAV --output_path_spectrogram PATH_OUTPUT

      Plot Histogram
      --plot_histogram --input_path_sound_plot PATH_TEMP_WAV --output_path_histogram PATH_OUTPUT

      Plot Heat Map
      --plot_heat_map --input_path_sound_plot PATH_TEMP_WAV --output_path_heat_map PATH_OUTPUT
      --file_load_model NEURAL_MODEL

### Create Samples (python3 main.py CreateSamples):
      Create Samples
      --input_path_convert PATH_TEMP_WAV --output_path_convert FILE_SAMPLES

### Training model (python3 main.py Training):
      Training Model
      --samples_training FILE_SAMPLES --neural_model NEURAL_MODEL --file_save_model FILE_PREFIX_MODEL

### Predict model (python3 main.py Predict):
      Predict Model
      --file_samples_predict FILE_INPUT --file_load_model FILE_PREFIX_MODEL

### Evaluation model (python3 main.py Evaluation):
      Evaluation Model
      --samples_training PATH_TEMP_WAV --neural_model NEURAL_MODEL

## Commands:

    Arguments(main.py):
         
        Posicional arguments:

          - Analyse                 Tool pack for analyzing neural models
          - Convert                 Decoding and sample rate adjust for media files
          - CreateSamples           Create samples for training models
          - Training                Training neural network model
          - Predict                 Predict label using neural network
          - Evaluation              Evaluation neural network model

        Optional arguments:

          -h, --help                Show this help message and exit
          --jump_size               Transformation time hop Size (Default 128)
          --number_bands            Number of bands on the Mel scale (Default 1024)
          --size_frame              Feature frame size (Default 60)
          --sample_rate             Input sound sampling rate (Default 8000)
          --amplify_log_scale       Amplifier logarithmic scale value (Default 80)
          --amplify_gain            Amplifier gain strength (Default 1)
          --number_classes          Number of mosquito classes (Default 3)
          --window_length_fft       Fourier transform window size (Default 2048)
          --feature_type            Define type resource for extraction (Default Mel-scale)
          --min_frequency           Minimum cutoff frequency (Default 300)
          --max_frequency           Maximum cutoff frequency (Default 300)
          --signal_attenuation      Value cutoff signal attenuation (Default 16)
          --max_attenuation         Maximum value cutoff signal attenuation (Default 64)
          --feature_series          Maximum value cutoff signal attenuation (Default 0)
          --neural_model            Define type neural network model (Default model_v2a)
          --loss LOSS               Define training loss function (Default binary_crossentropy)
          --optimizer               Define training optimizer function (Default adam)
          --quantization            Model export quantization range (Default 16)
          --metrics                 Define training metrics (Default accuracy)
          --steps_per_epochs        Number steps per epochs of training (Default 1024)
          --epochs                  Number epochs of training (Default 10)
          --file_save_model         Default path save file model (Default models_saved/model_v1b)
          --file_load_model         Default path save file model (Default models_saved/model_v1b)
          --plot_spectrogram_type   Define type spectrograms to plot (Default Mel-scale)
          --plot_spectrogram_zoom   Define zoom spectrogram plots (Default 2)
          --number_folds            Define number stratification folds (Default 2)
          --results_evaluation      Define evaluation results (Default 2)
          --file_samples_predict    Input file for prediction (Default )
          --path_samples_predict    Input path for prediction (Default dataset/dataset_test)
          --file_output_results     File output results predict (Default results/results.csv)
          --format_predict_output   Format output predict (Default multiple_labels_cuts)
          --output_path_ROC_curve   Output path ROC Curve (Default plots/roc_curve)
          --verbosity, -v           Set level verbose
          --output_path_spectrograms Output path spectrogram (Default plots/spectrogram)
          --output_evolution_error  Output plot evaluation error (Default {False})
          --plot_histogram          Plot histograms (Default {False})
          --plot_spectrogram        Plot spectrogram (Default {False})
          --plot_evolution_error    Plot spectrogram (Default {False})
          --plot_ROC_curve          Plot ROC curve (Default {False})
          --plot_heat_map           Plot heat map (Default {False})
          --value_seed              Define shuffle mode training samples (Default 0)
          --shuffle_mode            Define shuffle mode training samples (Default True)
          --samples_training        File samples training (Default dataset/samples_saved/samples_saved.npz)
          --input_file_convert      Input file to sound converter (Default )
          --output_file_convert     Output file of sound converter (Default )
          --input_path_convert      Input path to sound converter (Default dataset/dataset_unconverted)
          --output_path_convert     Output path of sound converter (Default dataset/dataset_converted)
          --path_samples_input      Load file samples (Default dataset/dataset)
          --file_samples_input      Load file samples (Default )
          --file_samples_output     Load file samples (Default dataset/samples_saved/samples_saved.npz)
          --output_path_histogram   Output path histograms (Default plots/histogram)
          --input_path_sound_plot   Input sound path (Default dataset/dataset_converted)
          --input_file_sound_plot   Input sound path (Default )
          --export_model_lite       Export model tensor lite format
          --output_path_heat_maps   Output path heat maps (Default plots/heat_map)

        --------------------------------------------------------------

## Requirements:

`librosa 0.8.0`
`pydub 0.25.1`
`moviepy 1.0.3`
`tqdm 4.60.0`
`SoundFile 0.10.3.1`

`matplotlib 3.4.1`
`numpy 1.18.5`
`scikit-learn 0.24.1`
`scipy 1.6.2`
`tensorflow 2.7.0`

`keras 2.7.0`
`setuptools 45.2.0`
`tensorflow 2.7.0`
`opencv-python 4.5.2.52`
