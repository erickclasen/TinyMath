'''
Make GPT-2 do math. Teach it with examples. Then inference.


Reference...
https://www.kaggle.com/code/changyeop/how-to-fine-tune-gpt-2-for-beginners/notebook

Took the example and trained it on a dataset of math examples.
add,sub,mul,div for number 0-89.

'''
import os
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments



# Parameters to adjust.

BS = 128 # Block size. For math this probably could be smaller.
PDTBS = 16 # Per Device Training Batch Size. Lower this for lower RAM use. 16 was good speed for my CPU.
EPOCHS = 100.0 # Lots of epochs to make it memorize well.

def load_dataset(file_path, tokenizer, block_size = BS):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
    )
    return dataset


def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator


def train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
  tokenizer = GPT2Tokenizer.from_pretrained(model_name)
  train_dataset = load_dataset(train_file_path, tokenizer)
  data_collator = load_data_collator(tokenizer)

  tokenizer.save_pretrained(output_dir)
      
  model = GPT2LMHeadModel.from_pretrained(model_name)

  model.save_pretrained(output_dir)

  training_args = TrainingArguments(
          output_dir=output_dir,
          overwrite_output_dir=overwrite_output_dir,
          per_device_train_batch_size=per_device_train_batch_size,
          num_train_epochs=num_train_epochs,
      )

  trainer = Trainer(
          model=model,
          args=training_args,
          data_collator=data_collator,
          train_dataset=train_dataset,
  )
      
  trainer.train()
  trainer.save_model()



# Paths to set.
# you need to set these if the location changes.
train_file_path = "math_dataset_90.txt" 
model_name = 'distilgpt2'
checkpoint_dir = "math-test/lastpoint"

# See if there is a checkpoint directory to pick up from.
# Make it by using for example, 'mv checkpoint-35000 lastpoint' 
# on the last/best numbered checkpoint dir, in the example this was 35000.
if os.path.isdir(checkpoint_dir):
    model_name = checkpoint_dir
    print("\nResuming from Checkpoint\n")	
output_dir = "math-test"
overwrite_output_dir = False
per_device_train_batch_size =  PDTBS #8
num_train_epochs = EPOCHS #2.0
save_steps = 2000

# It takes about 30 minutes to train in colab. Original code that is.
# This modified math code took a few days to train on a CPU.
train(
    train_file_path=train_file_path,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps
)

