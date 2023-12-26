# Regenerating Networked Systems’ Monitoring Traces Using Deep Learning

Algorithm for correcting sessions of users of large-scale networked systems based on deep learning.
[View publication](https://doi.org/10.1007/s10922-023-09790-9)

![Examples of traces: ground truth (obtained with 27 monitors), failed
(obtained with 7 monitors/20 failed), and recovered (using NN).](layout/example3.png?raw=true "Examples of traces: ground truth (obtained with 27 monitors), failed
(obtained with 7 monitors/20 failed), and recovered (using NN).")

## Neural Network Topologies

Three Neural Network topologies are proposed, named MLP, LSTM and CNN (Conv), according to their fundamental structures. Each neural network is composed of input, intermediate (also known as hidden layers), and output structures. Below, we provide more details of each proposed neural network topology.

<table>
    <tbody>
        <tr>
            <th width="20%">MLP Topology</th>
            <th width="20%">LSTM Topology</th>
            <th width="20%">CNN Topology</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/dense_model.png"></td>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/lstm.png"></td>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/conv.png"></td>
        </tr>


</table>

## Experimental Evaluation

### Fitting Analysis
Impact of the number of epochs on average error for Dense topology (arrangements A=3, window width W=11), LSTM topology (arrangements A=3, window width W=11), and Conv. topology (arrangements A=8, squared window width W=H=256).

<table>
    <tbody> 
        <tr>
            <th width="10%">MLP</th>
            <th width="10%">LSTM</th>
            <th width="10%">CNN</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/dense_error.png" alt="2018-06-04 4 33 16" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/lstm_error.png" alt="2018-06-04 4 40 06" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/conv_error.png" alt="2018-06-04 4 43 02" style="max-width:100%;"></td>
        </tr>


</table>

###  Parameter Sensitivity Analysis

Parameter sensitivity of Conv. topology withuniform probabilistic injected failure Fprob =10%
<table>
    <tbody>
        <tr>
            <th width="20%">Convolutional Topology</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/sens_conv.png" alt="2018-06-04 4 33 16" style="max-width:50%;"></td>
        </tr>


</table>


### Comparing our Neural Networks
Comparison of topologies MLP, LSTM (LS), and CNN for probabilistic injected failure and monitoring injected failure.
<table>
    <tbody> 
        <tr>
            <th width="10%">Probabilistic Injected Failure</th>
            <th width="10%">Monitoring Injected Failure</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/comparison_nn_pif.png" alt="2018-06-04 4 33 16" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/comparison_nn_mif.png" alt="2018-06-04 4 40 06" style="max-width:101%;"></td>
        </tr>
        
</table>

### Comparison with the State-of-the-Art (Convolutional vs Probabilistic)

Comparison between the best neural network model and state-of-the-art probabilistic technique. Values obtained for probabilistic error injection and monitoring error injection.
<table>
    <tbody>
        <tr>
            <th width="20%">Convolutional vs Probabilistic</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/results.png" alt="2018-06-04 4 33 16" style="max-width:120%;"></td>
        </tr>


</table>

### Qualitative Analysis

Impact, in terms of number (left) and duration (right) of a trace (S1) failed (Fmon = 20) and regenerated using the proposed BB-based (topology=Conv., threshold α =0.50, arrangements A =8, squared window width W = H =256) and prior probabilistic-based (threshold α =0.75).

<table>
    <tbody> 
        <tr>
            <th width="10%">Sessions Duration</th>
            <th width="10%">Number Sessions</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/CDF_duration.png" alt="2018-06-04 4 33 16" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/CDF_number_sessions.png" alt="2018-06-04 4 40 06" style="max-width:100%;"></td>
        </tr>
        
</table>

## Steps to Install:

1. Upgrade and update
    - sudo apt-get update
    - sudo apt-get upgrade 
    
2. Installation of application and internal dependencies
    - git clone [https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network]
    - pip install -r requirements.txt
    
3. Test installation:
    - python3 main.py -h


## Run experiments:

###  Run (all F_prob experiments)
`python3 run_jnsm_mif.py -c lstm`

### Run (only one F_prob scenario)
`python3 main.py`

###  Run (all F_mon experiments)
`python3 run_mif.py -c lstm`

### Run (only one F_mon scenario)
`python3 main_mif.py`


### Input parameters:

    Arguments(run_TNSM.py):
        
       -h, --help            Show this help message and exit
       --append, -a          Append output logging file with analysis results
       --demo, -d            Demo mode (default=False)
       --trials, -r          Mumber of trials (default=1)
       --start_trials,-s     Start trials (default=0)
       --skip_train, -t      Skip training of the machine learning model training?
       --campaign -c         Campaign [demo, mif, pif] (default=demo)
       --verbosity, -v       Verbosity logging level (INFO=20 DEBUG=10)


    --------------------------------------------------------------
   
    Arguments(main.py):

          -h, --help            Show this help message and exit
          --snapshot_column     Snapshot column position (Default 1)
          --peer_column         Peer column position (Default 2)
          --window_length       Define length window (Default 256)
          --window_width        Define width window (Default 256)
          --number_blocks       Define number blocks (Default 32)
          --topology            Neural topology (Default model_v1)
          --verbosity           Verbosity (Default 20)
          --epochs              Define number epochs (Default 120)
          --metrics             Define metrics (Default mse)
          --loss LOSS           Define loss (Default mse)
          --optimizer           Define optimizer (Default adam)
          --steps_per_epoch     Define batch size (Default 32)
          --threshold           Threshold (Default 0.75)
          --seed                Seed (Default 0)
          --learning_rate       Learning rate (Default 0.001)
          --pif PIF             PIF(0<x<1) MIF(>1) (Default 0)
          --duration            Duration
          --input_file_swarm    Input file swarm (Default )
          --save_file_samples   Save file samples (Default )
          --load_samples_in     Load file samples in (Default )
          --load_samples_out    Load file samples out (Default )
          --save_model          File save model (Default models_saved/model)
          --load_model          File load model (Default None)
          --input_predict       File input to predict (Default )
          --output_predict      File output to predict (Default )
          --file_corrected      File corrected for evaluation (Default )
          --file_failed         File failed for evaluation (Default )
          --file_original       File failed for evaluation (Default )
          --file_analyse_mode   File evaluation file mode (Default +a)
          --file_analyse        File evaluation file (Default results.txt)


        --------------------------------------------------------------
        Full traces available at: https://github.com/ComputerNetworks-UFRGS/TraceCollection/tree/master/01_traces



## Requirements:

`matplotlib 3.4.1`
`tensorflow 2.4.1`
`tqdm 4.60.0`
`numpy 1.18.5`

`keras 2.4.3`
`setuptools 45.2.0`
`h5py 2.10.0`





## Complementary Results

### Comparison with the State-of-the-Art (MLP vs Probabilistic)
Comparison between the neural network MLP and state-of-the-art probabilistic technique. Values obtained for probabilistic error injection and monitoring error injection.

<table>
    <tbody> 
        <tr>
            <th width="10%">Probabilistic Inject Failure</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/comparison_pif_dense_prob.png" alt="2023-03-16 4 33 16" style="max-width:100%;"></td>
        </tr>
</table>

<table>
    <tbody> 
        <tr>
            <th width="10%">Monitoring Inject Failure</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/comparison_mif_dense_prob.png" alt="2023-03-16 4 33 16" style="max-width:100%;"></td>
        </tr>
</table>


### Comparison with the State-of-the-Art (LSTM vs Probabilistic) 

Comparison between the neural network LSTM and state-of-the-art probabilistic technique. Values obtained for probabilistic error injection and monitoring error injection.

<table>
    <tbody> 
        <tr>
            <th width="10%">Probabilistic Inject Failure</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/comparison_pif_lstm_prob.png" alt="2023-03-16 4 33 16" style="max-width:100%;"></td>
        </tr>
</table>

<table>
    <tbody> 
        <tr>
            <th width="10%">Monitoring Inject Failure</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/comparison_mif_lstm_prob.png" alt="2023-03-16 4 33 16" style="max-width:100%;"></td>
        </tr>
</table>


## ACKNOWLEDGMENTS


This study was financed in part by the Coordenação
de Aperfeiçoamento de Pessoal de Nível Superior - Brasil
(CAPES) - Finance Code 001. We also received funding from
Rio Grande do Sul Research Foundation (FAPERGS) - Grant
ARD 10/2020 and Nvidia – Academic Hardware Grant

## Reference

    @article{Paim2023,
      author    = {Paim, Kayuã Oleques and Quincozes, Vagner Ereno and Kreutz, Diego and Mansilha, Rodrigo Brandão and Cordeiro, Weverton},
      title     = {Regenerating Networked Systems’ Monitoring Traces Using Neural Networks},
      journal   = {Journal of Network and Systems Management},
      year      = {2023},
      volume    = {32},
      number    = {1},
      pages     = {16},
      month     = {},
      doi       = {10.1007/s10922-023-09790-9},
      url       = {https://doi.org/10.1007/s10922-023-09790-9},
      issn      = {1573-7705},
    }
