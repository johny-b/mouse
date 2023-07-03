# Mouse-in-the-maze experiments

## Installation

## Code

Scripts:
* `channel_sum_histogram.py` - Compare channel sum between on distribution and out of distribution data. 
   Hardcoded values for channel `121` in layer `relu3`, split into "is cheese UP?" groups.
* `ablated_channel_rollout.py` - Compare rolouts for original policy and a policy with some channels zeroed.
* `compare_decision_square.py` - [TODO]
* `find_cheese_dir_components.py` - Find parts of the network corresponding to "this way to cheese" feature

Tools:
*   `tools.py` - Many different more-or-less general functions/classes. Messy.
*   `custom_mazes.py` - A set of predefined mazes with simple interpretation.
