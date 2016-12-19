# How to Reproduce the Results in "Learning to Ski"

Thank you for taking the time to read [my research paper](final.pdf). This document is intended to walk you through the process of reproducing the results I reported.

### __1.__ Installation

The dependencies can be installed via `pip install -r requirements.txt`. I recommend installing into a virtual environment.

Depending on your environment, you may need to install some system level packages first. On Ubuntu you need at least:

`apt-get install -y python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig`

See [gym's pypi page](https://pypi.python.org/pypi/gym/0.5.6#installing-everything) for more info.

### __2.__ Running the Simulation

The simulation is run via the `eval.py` script. Running `python eval.py -h` will provide information about the possible arguments.

### __3.__ Baselines

The baselines can be run with the following commands:

* Random
`python eval.py -e 150 --seed 42 random`

* Straight
`python eval.py -e 150 --seed 42 straight`

### __4.__ Oracle

The oracle was computed with

`python eval.py -r -e 3 --seed 42 human`

However, the output will depend on the human playing it. If you would like to play, the keys "a", "s", and "d" control the skier:

* a: Turn to the skier's right (your left)
* s: Do nothing
* d: Turn to the skier's left (your right)

The game will wait for you to press a key at every step. You can hold down a key to let it play faster.

### __5.__ Learning To Ski (L2S)

The L2S result in Table 2 can be run with:

`python eval.py -e 150 --seed 42 l2s`

### __6.__ Changing Experience Replay Memory Length

The values in Table 3 were run with the following commands:

`python eval.py -e 150 --seed 42 --agent-args '{"batch_size": 500}' l2s`

`python eval.py -e 150 --seed 42 --agent-args '{"batch_size": 1000}' l2s`

`python eval.py -e 150 --seed 42 --agent-args '{"batch_size": 2500}' l2s`

`python eval.py -e 150 --seed 42 --agent-args '{"batch_size": 5000}' l2s`

`python eval.py -e 150 --seed 42 --agent-args '{"batch_size": 10000}' l2s`

### __7.__ Changing Fitted Q-Iteration Frequency

The values in Table 4 were run with the following commands:

`python eval.py -e 150 --seed 42 --agent-args '{"iteration_size": 10}' l2s`

`python eval.py -e 150 --seed 42 --agent-args '{"iteration_size": 30}' l2s`

`python eval.py -e 150 --seed 42 --agent-args '{"iteration_size": 60}' l2s`

`python eval.py -e 150 --seed 42 --agent-args '{"iteration_size": 90}' l2s`

`python eval.py -e 150 --seed 42 --agent-args '{"iteration_size": 120}' l2s`

### __8.__ Components of Per-Episode Reward

Every run of `eval.py` creates a folder `./results/{agent-name}/{timestamp}` with the following files:

* `agent_args.json`: A json dictionary of any args that were passed in on the command line (for repeatability)

* `rewards.csv`: A csv of the per-episode rewards, with columns for total reward, reward due to elapsed time, and number of slaloms missed

You can generate the graph in Figure 1 with the following command (replacing the paths with the location of the rewards.csv for the run you wish to plot and the path where you would like the result written):

`python plot_rewards.py path/to/rewards.csv path/to/output.png`
