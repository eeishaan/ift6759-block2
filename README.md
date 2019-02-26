# ift6759-block2
Repository for horoma block 2 project

# Usage

### main

```
╔║ 11:44 AM ║ user20@helios1 ║ ~/ift6759-block2 ║╗
╚═> s_exec python3 -m horoma
usage: horoma [-h] {train,test} ...

optional arguments:
  -h, --help    show this help message and exit

commands:
  {train,test}
    train       Train models
    test        Test pre-trained models
```

### Train
```
╔║ 11:45 AM ║ user20@helios1 ║ ~/ift6759-block2 ║╗
╚═> s_exec python3 -m horoma train
usage: horoma train [-h] --embedding {vae} --cluster {kmeans,gmm}
                    [--mode {TRAIN_ALL,TRAIN_ONLY_EMBEDDING,TRAIN_ONLY_CLUSTER}]
                    [--params PARAMS]
horoma train: error: the following arguments are required: --embedding, --cluster
```
Example:
`s_exec python3 -m horoma train --mode TRAIN_ALL --embedding vae --cluster kmeans `

### Test

```
╔║ 11:45 AM ║ user20@helios1 ║ ~/ift6759-block2 ║╗
╚═> s_exec python3 -m horoma test
usage: horoma test [-h] --embedding {vae} --cluster {kmeans,gmm} -d DATA_DIR
                   [-f MODEL_PATH]
horoma test: error: the following arguments are required: --embedding, --cluster, -d/--data-dir

```

# Definitions
- Experiment: A collection of net, optimizer and loss function.
- Model: A class sub-classing `torch.nn.Module` or a sklearn model


# How to add new algorithms
- Make a new embedding model and place it inside `horoma/models` directory.
- Register model class with the embedding factory in `horoma/models/factory.py`. This will allow users to train model via command line.
- It is not mandatory to add an embedding model.
- Make a new cluster model and place it inside `horoma/models` directory.
- Register model class with the cluster factory in `horoma/models/factory.py`. This will allow users to train model via command line.
- It is not mandatory to add a cluster model.
- Now we need to associate an experiment with the embedding name for training the embedding model.
- You should either subclass the `HoromaExperiment` template experiment to make a new one or use the `HoromaExperiment` itself.
- `HoromaExperiment` provides a lot of boiler plate code and reduces code duplication. It also provides a lot of hooks that can be used to write custom algorithms.
- Register the chosen experiment in experiment factory at `horoma/experiments/factory.py`
- Each embedding model can have only one experiment associated with it, thus the experiment factory takes in the experiment name to create an experiment object.
