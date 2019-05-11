To run the random agent execute the following:\
`python3 trial_a3c.py --algorithm random`

To run the A3C algorithm execute the following:
1. Training:\ 
`python3 trial_a3c.py --algorithm a3c --train --lr <learning_rate> --update-freq <rate of display of results> --max-eps <maximum number of episodes> --gamma <value of discount factor> --save-dir <directory to save neural net>`
  
2. Prediction:\ 
`python3 trial_a3c.py --algorithm a3c --lr <learning_rate> --update-freq <rate of display of results> --max-eps <maximum number of episodes> --gamma <value of discount factor> --save-dir <directory to save neural net>`
