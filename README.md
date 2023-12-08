# TinyMath
TinyMath: How Small Can Language Models Be and Still do Math Correctly?

## Intent
The intent is to train small LLMs on math datasets to see how small the model size can go down to while still performing (mostly) correctly. Like TinyStories that are written on a 
level that a three year old can comprehend, the idea is to do something similar with math. In grade school or earlier humans learn to count to 5 to 10 and build on this. What kind of 
model can add 1 + 1 and so on up to 10 or so? How about one so small it gets a passing grade of say 70% right like a child would have to pass in elementary school.
To start a reasonable baseline is established using the fairly small already distil-GPT2 model. Then the idea is to move down from that size and see just how small the model can get
before it is a total failure.

As of now...
- There is a baseline test of a model trained using transfer learning on distil-GPT-2. The files are in the base directory. #1
- There is a baby GPT with half the layers,heads and embeds as GPT-1. The files are under nanoGPT. #2

##  1. Baseline using Distil-GPT-2
The first files uploaded here are for training distil-GPT2 with a math problem dataset ( math_dataset_90.txt ) that uses examples covering addition, subraction, multiplication and division for numbers from 0 to 89.
Exluding any division by zero.
This dataset intentionally is missing 90-99 to have this as a holdout region to test if the model can infer how to do math with these numbers. As in, has it learned enough to 'do math'
or is it just memorizing well.


## Files
so far
- math_training.py - Trains a distil-GPT2 model, currently set to 100 epochs, can be modifed and hacked.
- math-inference.py - Runs inference on the model that was created in directory math-test by running math-training.py
- math_dataset_90.txt - Reference dataset used with distil-GPT2. Math examples file from 0-89 format as follows...

```
27 / 86 = 0.31
24 + 82 = 106.00
76 * 56 = 4256.00
76 * 45 = 3420.00
82 + 78 = 160.00
57 * 25 = 1425.00
12 + 27 = 39.00
78 / 71 = 1.10
64 - 1 = 63.00
60 * 39 = 2340.00
```
- generate-math-dataset.py - Can generate math datasets for any range of numbers and number of random examples.

## Background
Got the idea for this from working with Andrej Karpathy's llama2.c repository code ( https://github.com/karpathy/llama2.c ) that was used to train models based TinyStories on and then reading TinyStories: How Small Can Language Models Be and Still Speak Coherent English?
https://arxiv.org/abs/2305.07759
Plus, when I was running through the nn-zero-to-hero video series ( https://github.com/karpathy/nn-zero-to-hero ). I saw the exercise question in the comments about creating a calculator and that just helped to push forward on this....

From https://github.com/erickclasen/makemore/blob/master/verbose-readme.pdf

This one will open right in browser...
http://erick.heart-centered-living.org/wp-content/uploads/2023/11/verbose-readme.pdf
...
```
Could mod inference code to be a ‘calculator’. Take in 0-9 + - / * from input and then
feed that in as prompt and print one output from model. TDB - EX2: Train
the GPT on your own dataset of choice! What other data could be fun to blabber on about? (A fun
advanced suggestion if you like: train a GPT to do addition of two numbers, i.e. a+b=c. You may find it
helpful to predict the digits of c in reverse order, as the typical addition algorithm (that you're
hoping it learns) would proceed right to left too. You may want to modify the data loader to simply
serve random problems and skip the generation of train.bin, val.bin. You may want to mask out the loss
at the input positions of a+b that just specify the problem using y=-1 in the targets (see
CrossEntropyLoss ignore_index). Does your Transformer learn to add? Once you have this, swole doge
project: build a calculator clone in GPT, for all of +-*/. Not an easy problem. You may need Chain of
Thought traces.) (From: Let's build GPT: fromscratch, in code, spelled out.
https://www.youtube.com/watch?v=kCc8FmEb1nY )
See Appendix 2A for a grid search on hyper parameters for a “Tiny Math” model that was
trained on math consiting of addition, subtraction, multiplication and division on number 0-9,
excluding those that result in div/0.
```
There was some initial warmup with the makemore code and this work is detail in the PDF for that repository. See Appendix 2A
https://github.com/erickclasen/makemore/blob/master/verbose-readme.pdf


## distil-GPT2 Examples
So far using distil-GPT2.
- Some that work.
- Some that will fail and then work.
- One from 90 and above that just fails.
```
erick@MS-7680:~/python/ml/gpt-2$ python3 math-inference.py 
Using model path: math-test
Prompt: 2 + 2
2 + 2 = 4.00


Time: 2.91
erick@MS-7680:~/python/ml/gpt-2$ python3 math-inference.py 
Using model path: math-test
Prompt: 36 / 3
36 / 3 = 12.00


Time: 2.39
erick@MS-7680:~/python/ml/gpt-2$ python3 math-inference.py 
Using model path: math-test
Prompt: 5 * 4
5 * 4 = 10.00


Time: 2.36
erick@MS-7680:~/python/ml/gpt-2$ python3 math-inference.py 
Using model path: math-test
Prompt: 5 * 4
5 * 4 = 20.00


Time: 2.35
erick@MS-7680:~/python/ml/gpt-2$ python3 math-inference.py 
Using model path: math-test
Prompt: 90 + 4
90 + 4 = 84.00

```
## 2. TinyMath nanoGPT 'baby' GPT, half the layers,heads and embeds as GPT
In the nanoGPT directory is a set of files that allows training and inference on smaller
scratch trained GPT models. nanoGPT code originates from Andrej Karpathy's nanoGPT repo at https://github.com/karpathy/nanoGPT

Loss at 5000 iters
- step 5000: train loss 0.6861, val loss 0.6933


The dataset is under the data directory and has the dataset of math expression examples from 0-89 of add,subtract,multipy and divide
to 2 decimal places. The data is prepared into train.bin and val.bin as a character level model as there are only 18 characters and there
are no 'words' to speak of, keep it char level makes sense.

config dir contains the imported configuration file ( train_tinymath.py ) that set up the correct parameters for the training and inference.

### Files added beyond nanoGPT code for the TinyMath project
math-sample.py - Beyond the normal code found in the regular nanoGPT are a special sampling file that creates a calcultor that will allow a user prompt which is optional and will output a one line expression as the 'answer'. It also contains a timer to measure how long the calculation takes.

grading-math.py - Prototype grading code for grading the samples. A file that does simple grading where it posts the correct answer next to the answer 'calculated' by GPT, it uses a file called expressions.txt which is created by piping or copy/pasting the output from sample.py to the file.

self-grading.py - Same as grading-math.py but reports correct and incorrect answers with the correct answers noted. Prints a score at the end of the run.
It is a bit more fussy when it runs and will fail if it sees malformed expressions in expressions.txt. Hand cleaning is require for now.

## Examples

### How to train
```
python3 train.py config/train_tinymath.py --device=cpu

```
### Snippet of output showing the params and model size
```
...block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2...

...tokens per iteration will be: 16,384
found vocab_size = 18 (inside data/tinymath/meta.pkl)
Initializing a new model from scratch
number of parameters: 10.63M...

...
step 0: train loss 2.9920, val loss 2.9929
iter 0: loss 2.9804, time 2429347.63ms, mfu -100.00%
iter 10: loss 2.0419, time 16193.55ms, mfu 0.02%
iter 20: loss 1.8771, time 15952.87ms, mfu 0.02%
iter 30: loss 1.8551, time 16422.38ms, mfu 0.02%
iter 40: loss 1.8314, time 16159.03ms, mfu 0.02%
iter 50: loss 1.8065, time 16221.15ms, mfu 0.02%
iter 60: loss 1.7635, time 16225.26ms, mfu 0.02%
iter 70: loss 1.6543, time 16284.06ms, mfu 0.02%
...
iter 4940: loss 0.7058, time 16458.34ms, mfu 0.02%
iter 4950: loss 0.7026, time 16458.35ms, mfu 0.02%
iter 4960: loss 0.6981, time 16408.22ms, mfu 0.02%
iter 4970: loss 0.7002, time 16512.51ms, mfu 0.02%
iter 4980: loss 0.7000, time 16505.15ms, mfu 0.02%
iter 4990: loss 0.7015, time 16465.89ms, mfu 0.02%
step 5000: train loss 0.6861, val loss 0.6933
saving checkpoint to out-tinymath
iter 5000: loss 0.7052, time 2249072.31ms, mfu 0.02%



```


### Normal nanoGPT sample.py running output
```
python3 sample.py --out_dir=out-tinymath --device=cpu
```

tail of output
```
55 / 35 = 1.57
16 / 46 = 0.36
63 / 3 = 21.00
74 * 53 = 3902.00
81 - 6 = 75.00
33 - 18 = 15.00
55 * 64 = 3560.00
71 * 49 = 3479.00
36 - 15 = 21
---------------

```
### A GPT calculator
```
python3 math-sample.py --out_dir=out-tinymath --device=cpu
Overriding: out_dir = out-tinymath
Overriding: device = cpu
Enter your prompt (hit Enter to use the default prompt): 12 + 24 =
number of parameters: 10.63M
Loading meta from data/tinymath/meta.pkl...
12 + 24 = 36.00


Execution time: 0.24 seconds

```
### Simple grading, no score, no indication of right/wrong
```
python3 grading-math.py
```
```
88 + 32 = 120.00 = 120
48 + 24 = 72.00 = 72
31 * 76 = 2356.00 = 2356
69 * 50 = 3450.00 = 3450
19 - 52 = -33.00 = -33
4 / 18 = 0.22 = 0.22
74 + 22 = 96.00 = 96
85 / 68 = 1.24 = 1.25
8 / 59 = 0.13 = 0.14
74 / 76 = 0.9 = 0.97

```
### Grading the samples automatically
```
python3 self-grading.py
```
```
48 + 24 = 72   Correct
31 * 76 = 2356   Correct
69 * 50 = 3450   Correct
19 - 52 = -33   Correct
4 / 18 = 0.22   Correct
74 + 22 = 96   Correct
85 / 68 = 1.25   Incorrect, model predicted 1.24
8 / 59 = 0.14   Incorrect, model predicted 0.13
74 / 76 = 0.97   Incorrect, model predicted 0.9

Score: 95/123 (77.24%)

```

## TBD 
- Smaller and Larger Models in line with TinyStories examples or something similar.
- Review optimization of hyperparameters with a ceiling oon model size from makemore fork ( https://github.com/erickclasen/makemore )
- Smaller dataset 0-5 ?
