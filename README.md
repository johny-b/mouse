# Mouse-in-the-maze experiments

Code I have written when working on the [post about how goal misgeneralization happens](https://www.lesswrong.com/posts/vY9oE39tBupZLAyoC/localizing-goal-misgeneralization-in-a-maze-solving-policy).

## Installation

```
git clone https://github.com/johny-b/mouse-goal-misgeneralization.git
cd mouse-goal-misgeneralization
pip3 install -r requirements.txt
```

Developed on python 3.10.6, 3.8 should also work.

## Code

Scripts:

1. `browse_channels.py` - Compare sums of activations of a channel in a simple maze
2. `channel_sum.py` - Plot the relationship between channel sum and direction of the cheese in a set of random mazes
3. `high_channel_sum.py` - Analyze behaviour of the mouse when sum of a given channel is high
4. `ablated_channel_rollout.py` - Compare rollouts between original and modified policy
5. `find_channels.py` - A simple detector of important channels

Tools:
*   `tools.py` - Many different more-or-less general functions/classes. Messy.
*   `custom_mazes.py` - A set of simple predefined mazes. Messy.

Other:
*   `weird_seeds.txt` - Few seeds where mouse does a really weird thing.
