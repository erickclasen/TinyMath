
'''
Make GPT-2 do math. Teach it with examples. Then inference.
Train on examples of add,sub,mul,div for numbers 0-89, excluding div/0.

Reference...
https://huggingface.co/blog/how-to-generate

...used this to start an inference of the math model. A bit of a quick and dirty hack.
Clipping the output so it only shows the first 'example' as the answer. Without clipping it
will roll on giving a bunch of examples, mimicking the training dataset.
'''

# ## Step 3. Inference

# In[ ]:


from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
import time

import logging
logging.getLogger('tensorflow').disabled = True


import sys
# In[ ]:


def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def generate_text(sequence, max_length, model_path):
    model_path = model_path 
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    input_ids = tokenizer.encode(sequence, return_tensors='pt')
    output = model.generate(input_ids,
                            do_sample=True, 
                            max_length=max_length, 
                            pad_token_id=tokenizer.eos_token_id, 
                            top_k=50, 
                            top_p=0.95,
                            num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # find the last complete sentence, look for the period. This is normally how I do it.
    # For math example, go two places beyond the end to get the two digits past the '.'
    last_sentence = generated_text.rfind('.')
    if last_sentence != -1:
        generated_text = generated_text[:last_sentence+3]
    
    print(generated_text)
    

# In[ ]:

# Get a location from cmd line or default to result.
# Set default value if no argument is provided
# Adjust as needed
model_path = "math-test"

# Check if an argument is provided
if len(sys.argv) > 1:
    model_path = sys.argv[1] # Optional non-default model location for CLI.

# Use file_path variable in your code
print("Using model path:", model_path)

# Prompt for input
sequence = input("Prompt: ") 

'''
Guess Max length for math is set to 16 per the length of the max line in the math dataset_28...
erick@OptiPlex-7010:~/python/ml/makemore$ wc -L math_dataset_28_sorted.txt 
16 math_dataset_28_sorted.txt

...tried this, prints the next 'example', make it 10, seems to work OK.

'''


# This can be set different, works out good for the math example.
max_len = 10 #int(len(input())) # 20

# Check how long it takes, using a  timer.
start_time = time.time() # Start the timer

generate_text(sequence, max_len, model_path) # oil price for July June which had been low at as low as was originally stated Prices have since resumed
end_time = time.time() # End the timer


running_time = end_time - start_time # Calculate the running time

print("\n\nTime:",round(running_time,2))

# More references...
# The following process may be a little more complicated or tedious because you have to write the code one by one, and it takes a long time if you don't have a personal GPU.
# 
# Then, how about use Ainize's Teachable NLP? Teachable NLP provides an API to use the model so when data is input it will automatically learn quickly.
# 
# Teachable NLP : [https://ainize.ai/teachable-nlp](https://link.ainize.ai/3tJVRD1)
# 
# Teachable NLP Tutorial : [https://forum.ainetwork.ai/t/teachable-nlp-how-to-use-teachable-nlp/65](https://link.ainize.ai/3tATaUh)

