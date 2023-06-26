## What with a weird mouse behavior?

1. Find the mouse in a "wacky" state
2. Take a look at the directions in the relu\_3 layer
3. Find which channel makes it do weird things
4. Go backward - find what influences this particular channel

## MAKE IT GO TO CHEESE

This should make mouse more likely go to cheese:
1. Probe for mouse location
2. Probe for cheese location
3. Calculate desired direction
4. Modify appropriate vectors

BUT: we could just do the same thing in the last layer, so this is not really interesting?


