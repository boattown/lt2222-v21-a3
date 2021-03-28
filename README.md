# LT2222 V21 Assignment 3

Your name: Klara Båstedt

## Part 1

### Function a(f)

Function a reads a file and puts its content in a list. 
Two start-symbols are inserted at the beginning of the list and two end-symbols at the end.
The function returns that list and a list of the unique tokens in it.

### Function g(x, p)

Function g is a helper function called in function b.
It creates a numpy array with zeros of the same length as list p. (List p is the list of unique tokens returned by function a.)
In the numpy array, the 0 is replaced with 1 at the index of x in list p.
The function returns that numpy array.

### Function b(u, p)

Function b takes the two lists returned by function a. 
It iterates over the first list and when it encounters an element from vowels, it appends its index (in the vowels list) to the gt list.
In other words, gt stores integers that can be mapped to vowels in the sorted vowels list.

It also calls function g(x, p) on the two tokens before the vowel and the two tokens after it.
The numpy arrays returned by the 4 function calls are concatenated and appended to the gr list.

The function returns two numpy arrays created from the lists gr and gt. The first numpy array will have the same length, and each element is another numpy array where the 4 tokens surrounding the vowels are marked with 1. The second numpy array will consist of integers that can be mapped to vowels. 

### Commandline arguments

The argument "--k" is an optional argument, an integer that specifies the hiddensize, which defaults to 200.
The argument "--r" is an optional argument, an integer that specifies number of epochs, which defaults to 100.
The argument "m" is a string that specifies a filename with the newspaper articles.
The argument "h" is a string that specifies a filename that the model will be saved to.

## Part 2

In eval.py, I used some functions from train.py as a base as well as the sorted list of vowels and model.vocab.
I used function a from train.py to preprocess the testdata.

My function createInstances is similar to function b from train.py, but I call it with the vocabulary from the training data rather than the vocabulary from the test data. I also decided to skip vowels with surrounding characters that are not part of model.vocab. This is to avoid a ValueError.

In the function predictVowels, I predict vowels in the test data by choosing the vowel that the model assigns the highest probability to when given the arrays that represent the surrounding characters.

In the function predictText, I replace the vowels in the text with the predicted vowels and print the output to a file. (Again, I skip vowels with surrounding characters that are not part of model.vocab.)

In calculateAccuracy, I compare the predicted vowels with the true vowels and compute true positives over all instances.

## Part 3

#### Five different variations of the --k option, holding the --r option at its default:

k = 100: 0.3660601122578309

k = 200 (default): 0.459985515118595

k = 300: 0.4138602208944414

k = 400: 0.14257046291266823

k = 500: 0.35804816223067176

#### Five different variations of the --r option, holding the --k option at its default:

r = 50:  0.2851258374072062

r = 100 (default):  0.5285623755205504 (saved as 'best_model.pt')

r = 200:  0.17611201641619892

r = 300:  0.31103265133683383

r = 400:  0.15477699318003502

It is hard to draw any conclusions since the accuracy differes quite a lot between different models.
For instance, the first time I trained a model with --k and --r at their default, I got an accuracy of 0.46. 
The second time I trained a model with the same values, the accuracy was only 0.40, and the third time it was 0.53.

It also surprises me that the accuracy is 0.41 with k = 300, only 0.14 with k = 400, and then 0.36 with k = 500.
Similarly, the accuracy is higher with r = 100 than r = 200, but higher again with r = 300.

Despite this variation, which makes it hard to see any patterns, it seems that the best accuracy score is obtained with the --r and --k options close to their default values.
I also observed that the loss does not decrease much after ~250 epochs, with --k at its default.

## Bonuses

### Bonus Part A: Perplexity

Since model.pt is overwritten with each training, I can not get the perplexity values for the exact models in part 3.
I repeated the experiment and documented the perplexity scores below.
(However, I am not sure I computed the perplexity correctly since the values are very high and there are not that many vowels to choose from.)

#### Five different variations of the --k option, holding the --r option at its default:

k = 100: 63326

k = 200 (default): 55171

k = 300: 50464

k = 400: 47578

k = 500: 46247


#### Five different variations of the --r option, holding the --k option at its default:

r = 50: 77573

r = 100 (default): 55171

r = 200: 40394

r = 300: 37139

r = 400: 35708

## Other notes
