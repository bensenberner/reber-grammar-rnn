# Overview:
In this project, I constructed an RNN to classify whether or not a string matched the [embedded Reber Grammar](https://web.archive.org/web/20190114051314/https://cnl.salk.edu/~schraudo/teach/NNcourse/reber.html) which I will refer to as the `ERG`.

# Data
## ERG-conforming strings
Ultimately, the data that I feed into this keras-based RNN had to be a matrix numerically representing strings. Each value in the matrix represented the index of a particular character in the reber alphabet `BEPSTVX`, or it contained a padding character.
First, I had to synthetically generate examples of strings that matched the ERG and strings that did not.
Sampling from the ERGs was pretty simple: I constructed the grammar as a graph, and as I traversed the graph, whenever I came upon a node with multiple outgoing edges, I simply chose one uniformly randomly until I reached the final node. I discarded any strings that exceeded a `max_length` (since they all had to fit into a rectangular matrix).
## non-ERG-conforming strings
However, coming up with the strings that didn't match the ERG required an *iterative process*.
### random
First I tried creating strings of a random length between `[min_length, max_length]` (the `min_length` represented the shortest possible reber string). Later, when I modeled just these two classes, the model performed well but it was unable to recognize when I took an ERG and perturbed it by adding an additional invalidating character, changed a character to invalidate the grammar, or removed a character.
### perturbed
In order to capture these "almost but not quite" ERG strings, I created a new class of strings, the **perturbed** strings, which are valid-ERG strings that I perform `num_perturbations` edits upon, each one rendering it invalid.
### symmetry disturbed
Next, I focused on a specific perturbation, one that speaks to the very purpose of the ERG. The reason why the ERG is difficult for RNNs to model well is because of a long range dependency - **the second and penultimate characters MUST be the same**, although there are two possible values for thm (`T` or `P`). I did find that, even with my random perturbations, it was difficult for my model to recognize when I "disturbed the symmetry" of these two characters (making them `T,P` or `P,T` thereby breaking the grammar for the entire string).
To that end, I created another class of strings, the **symmetry disturbed** strings, which are a very specific subset of the perturbed strings: I created a valid reber string and then I "flip the bit" of either the second or penultimate character (if you pretend like the "bits" are `T` and `P`).

I created a class that allows the user to easily specify the proportions of each of these types of data before synthesizing it (and it performs validation too!)

# Model
------
Because of the long-range dependency of these strings (stated above), it is very difficult for vanilla RNNs to recognize the grammar: the part of their hidden state representing their "memory" of what the second character is quickly degrades by the time they reach the penultimate character, so it is hard for them to recognize the "symmetry-disturbed" strings are fake.
To that end, I used an `LSTM`, which *can* remember long-range dependencies.
In order to create the architecture of the LSTM, I tried a few different numbers of layers and layer sizes, but then I thought it'd be worthwhile to try using *neural architecture search* to pick out the architecture for me. I used [autokeras](https://autokeras.com) for the job - I used some GPU-provisioned machines graciously provided by [heap](heap.io) for the [Recurse Center](http://recurse.com) in which I was a participant for three months.


# TODO
- describe what it took to get autokeras set up
- attempt to describe the architecture of what autokeras created
- describe how I might use boosting to further improve the model
- what if multiple perturbations make the string valid again? what are the odds of this?