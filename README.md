Reinforcement Learning in Pacman
================================

Introduction
------------

In this project experimented with various MDP and Reinforcement Learning techniques namely value iteration, Q-learning and approximate Q-learning. This is part of Pacman projects developed at [UC Berkeley](http://ai.berkeley.edu/reinforcement.html). 


Directory Structure
-------------------

---RL

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [qlearningAgents.py](RL/qlearningAgents.py)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [analysis.py](RL/analysis.py)

---[lab.pdf](lab.pdf)

---README.md

---[report.pdf](report.pdf)


Executing
---------

Then run the autograder using $python autograder.py

It gave me a score of 25/25.


Value Iteration
---------------
$python gridworld.py -a value -i 100 -k 10

$python gridworld.py -a value -i 5

Bridge Crossing Analysis
------------------------
$python gridworld.py -a value -i 100 -g BridgeGrid --discount 0.9 --noise 0.2

Policies
--------
$python autograder.py -q q3

Q-Learning
----------
python gridworld.py -a q -k 5 -m

Epsilon Greedy
--------------
$python gridworld.py -a q -k 100 

$python crawler.py

Bridge Crossing Revisited
-------------------------
$python gridworld.py -a q -k 50 -n 0 -g BridgeGrid -e 1

Q-Learning and Pacman
---------------------
$python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid 

Approximate Q-Learning
----------------------
$python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid 

$python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid 

$python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic 

Developed by
------------
[Sai Srinadhu K](https://www.linkedin.com/in/sai-srinadhu-katta-a189ab11b/)
