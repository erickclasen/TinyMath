# TinyMath
TinyMath: How Small Can Language Models Be and Still do Math Correctly?

## Intent
The intent is to train small LLMs on math datasets to see how small the model size can go down to while still performing (mostly) correctly. Like TinyStories that are written on a 
level that a three year old can comprehend, the idea is to do something similar with math. In grade school or earlier humans learn to count to 5 to 10 and build on this. What kind of 
model can add 1 + 1 and so on up to 10 or so? How about one so small it gets a passing grade of say 70% right like a child would have to pass in elementary school.
To start a reasonable baseline is established using the fairly small already distil-GPT2 model. Then the idea is to move down from that size and see just how small the model can get
before it is a total failure.

## Baseline using Distil-GPT-2
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


## Examples
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
## TBD
More work to come on smaller models




