## What with a weird mouse behavior?

1. Find the mouse in a "wacky" state
2. Take a look at the directions in the relu\_3 layer
3. Find which channel makes it do weird things
4. Go backward - find what influences this particular channel

## Rescale channels, maybe?

Maybe there is assymetry between "go left" and "go right" channels, and between "go up" and "go down"?
Maybe e.g. we could rescale them to remove the topright effect?

More precise:
* Compare mean/variance
* If they are similar, forget
* If they are different, e.g. "left" has 0.5 mean of right - fix this (i.e. multiply left by two or something)
* Maybe this removes top-right bias? 

## MAKE IT GO TO CHEESE

This should make mouse more likely go to cheese:
1. Probe for mouse location
2. Probe for cheese location
3. Calculate desired direction
4. Modify appropriate vectors

BUT: we could just do the same thing in the last layer, so this is not really interesting?
