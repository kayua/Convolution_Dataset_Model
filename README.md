# Regenerating Networked Systems’ Monitoring Traces Using Deep Learning

Algorithm for correcting sessions of users of large-scale networked systems based on deep learning.

![Examples of traces: ground truth (obtained with 27 monitors), failed
(obtained with 7 monitors/20 failed), and recovered (using NN).](layout/example2.png?raw=true "Examples of traces: ground truth (obtained with 27 monitors), failed
(obtained with 7 monitors/20 failed), and recovered (using NN).")



<table>
    <tbody>
        <tr>
            <th width="20%">Mel Scale</th>
            <th width="20%">Chroma Contrast</th>
            <th width="20%">Chroma Power</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/layout/blob/master/dense_model.png" alt="2018-06-04 4 33 16" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/API-Aedes-Detector/blob/master/layout/contrast/aedes_aegypti.png" alt="2018-06-04 4 40 06" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/MosquitoClassification-TensorFlow/blob/master/spectrograms/chroma_power/aedes_aegypti7_chroma_power.png" alt="2018-06-04 4 43 02" style="max-width:100%;"></td>
        </tr>


</table>


## Input parameters:

    Arguments(run_NOMS22.py):
        
        -h, --help          Show this help message and exit
        --append, -a        Append output logging file with analysis results (default=False)
        --trials, -r        Number of trials (default=1)
        --start_trials, -s  Start trials (default=0)
        --skip_train, -t    Skip training of the machine learning model training?
        --campaign, -c      Campaign [demo, lstm, no-lstm, deterministic](default=demo)
        --verbosity, -v     Verbosity logging level (INFO=20 DEBUG=10)

    --------------------------------------------------------------
   
    Arguments(main.py):

        -h, --help                  Show this help message and exit
        --original_swarm_file       File of ground truth.
        --training_swarm_file       File of training samples
        --corrected_swarm_file      File of correction
        --validation_swarm_file     File of validation
        --failed_swarm_file         File of failed swarm
        --analyse_file              Analyse results with statistics
        --dense_layers              Number of dense layers (e.g. 1, 2, 3)
        --neurons NEURONS           Number neurons per layer
        --cells CELLS               Numbers cells(neurons) LSTM
        --num_sample_training       Number samples for training
        --num_epochs                Number epochs training
        --analyse_file_mode         Open mode (e.g. 'w' or 'a')
        --model_architecture_file   Full model architecture file
        --model_weights_file        Full model weights file
        --size_window_left          Left window size
        --size_window_right         Right window size
        --threshold                 i.e. alpha (e.g. 0.5 - 0.95)
        --pif PIF                   Pif (only for statistics)
        --dataset DATASET           Dataset (only for statistics)
        --seed SEED                 Seed (only for statistics)
        --lstm_mode                 Activate LSTM mode
        --no-lstm_mode              Deactivate LSTM mode
        --skip_train, -t            Skip training of the machine learning model
        --deterministic_mode        Set deterministic correction mode
        --skip_correct, -c          Skip correction of the dataset
        --skip_analyse, -a          Skip analysis of the results
        --verbosity, -v             Verbosity logging level (INFO=20 DEBUG=10)
        --mode MODE                 Mode

        --------------------------------------------------------------
        Full traces available at: https://github.com/ComputerNetworks-UFRGS/TraceCollection/tree/master/01_traces

##  Run (all F_prob experiments):
`python3 run_nom22.py -c lstm`

## Run (only one F_prob scenario)
`python3 main.py`

##  Run (all F_mon experiments):
`python3 run_mif.py -c lstm`

## Run (only one F_mon scenario)
`python3 main_mif.py`

## Requirements:

`matplotlib 3.4.1`
`tensorflow 2.4.1`
`tqdm 4.60.0`
`numpy 1.18.5`

`keras 2.4.3`
`setuptools 45.2.0`
`h5py 2.10.0`

## ACKNOWLEDGMENTS


This study was financed in part by the Coordenação
de Aperfeiçoamento de Pessoal de Nível Superior - Brasil
(CAPES) - Finance Code 001. We also received funding from
Rio Grande do Sul Research Foundation (FAPERGS) - Grant
ARD 10/2020 and Nvidia – Academic Hardware Grant

