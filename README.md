# scale-sim-front
Generate input files for [SCALE-Sim](https://github.com/ARM-software/SCALE-Sim) easily from Keras/PyTorch models.  

## How to run
Just to see how it runs, try the following command.
```
./run.sh
```
This will use the custom model defined in the _alexnet.py_ to extract the layer information and outputs the result as an input format of SCALE-Sim.  
Try the following command to see the options available.
```
python scale-sim-front.py --help
```

The result file will be saved in the folder _out_.
