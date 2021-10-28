# imperialThesis
This is the code to go along with the thesis titled "Uncertainty and Bayesian Neural Networks in Environmental Sound Classification" by author, Lorcan Donnellan.

##Data load
The models take the Urban Sounds 8K dataset as input. This was not included in this repository for size reasons. It can be downloaded from: 
[Urban Sounds 8K](https://urbansounddataset.weebly.com/urbansound8k.html "UrbanSounds8K")

Once downloaded and unzipped, run the makeTfRecordsRawWav.py script to create the tfrecords. 
The empty directories tfRecordsRawWav/fold1, tfRecordsRawWav/fold2, ..., tfRecordsRawWav/fold10 will be filled with 2 tfrecords each.
The number of tfrecords per fold is a variable in the script, and can be changed. We selected 2, as this gives the tfrecords closest to what is optimal according to
the tfrecord documentation.
Line 68 in makeTfRecordsRawWav.py is where we define the audio directory. This should correspond to the audio directory containing the wav files from Urban Sounds 8K.
In our setup we copied makeTfRecordsRawWav.py directly into the unzipped UrbanSounds8K folder, and thus simply had what is seen in the script.
This code may take a while to run.

##Package Installation
It is always recommended to enter a virtual environment before installing tensorflow modules. Once this is done, cd into the package folder in this repo and run:

`pip3 install -r requirements.txt`

This should install the necessary packages in the correct versions, which are all saved in requirements.txt

##Running the deterministic model: M18
Go to the deterministic directory: `cd code/m18/deterministic`
Run: `CUDA_VISIBLE_DEVICES=0 python3 m18model.py`

The script responsible for data load in is dataRawWav.py. Line 32 in this script is where we define the location of the tfrecords. 
This will need to be adapted to where the user saved the tfrecords.
Test results are outputted in the results folder.

##Running the deterministic model: M18-P
Go to the probabilistic directory: `cd code/m18/probabilistic`.
The script responsible for data load in all scripts in is dataRawWav.py. Line 24 in this script is where we define the location of the tfrecords. 
This would need to be adapted to the users save location/


This folder contains several main scripts with names beginning with m18model. Each correspond to different hyperparameter scripts, all beginning with hp, which are loaded in inside their respective main script.
Similarlyl, there are several results csv files in the results directory, named by the variable being tested.
Each main script is responsible for different experiments, which are obvious from their names.
For instance, m18_run_batches.py pulls its hyperparameters from hpBatch.py and runs through multiple train and test times. All hyperparameters except batch size remain
constant throughout these runtimes. Batch size is varied over a predefined array of values. Results are stored in the results folder. Slightly different results are written for each experiment.

The generic code to run is `CUDA_VISIBLE_DEVICES=0 python3 m18_run.py 18`

The 18 argument is required as we initially wrote in the possibilty to have a mix of deterministic and probabilistic layers.
This argument was the number of the latter in the model.
`m18_run.py` above could be any of the following: m18_run_batches.py, m18_run_klDivSamples.py, m18_run_mc.py, m18_run_post.py, m18_run_prisHps.py, m18_run_pris.py, m18_run.py
