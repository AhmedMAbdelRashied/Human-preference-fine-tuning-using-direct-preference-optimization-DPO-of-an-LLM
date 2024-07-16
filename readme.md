# Human preference fine-tuning using direct preference optimization (DPO) of an LLM
# Supervised fine tuning Gemma on chat Task
Recall that creating a ChatGPT at home involves 3 steps:

1. pre-training a large language model (LLM) to predict the next token on internet-scale data, on clusters of thousands of GPUs. One calls the result a "base model"
2. supervised fine-tuning (SFT) to turn the base model into a useful assistant
3. human preference fine-tuning which increases the assistant's friendliness, helpfulness and safety.

in this Projict I will do the second and third step. This involves supervised fine-tuning (SFT for short), also called instruction tuning. and DPO training

# 1. Supervised fine-tuning takes
Supervised fine-tuning takes in a "base model" from step 1, i.e. a model that has been pre-trained on predicting the next token on internet text, and turns it into a "chatbot"/"assistant". This is done by fine-tuning the model on human instruction data

## 1.1 Load tokenizer
Next, we instantiate the tokenizer, which is required to prepare the text for the model. The model doesn't directly take strings as input, but rather `input_ids`, which represent integer indices in the vocabulary of a Transformer model.

- the padding token ID.
during fine-tuning, we will need to pad the (instruction, completion) pairs in order to create batches of equal length.

- the model max length: this is required in order to truncate sequences which are too long for the model. Here we decide to train on at most 2048 tokens.

the chat template. A [chat template](https://huggingface.co/blog/chat-templates) determines how each list of messages is turned into a tokenizable string, by adding special strings in between such as `<|user|>` to indicate a user message and `<|assistant|>` to indicate the chatbot's response. Here we define the default chat template, used by most chat models. See also the [docs](https://huggingface.co/docs/transformers/main/en/chat_templating).

## 1.2 Define model arguments

Next, it's time to define the model arguments.

Here, some explanation is required regarding ways to fine-tune model.

### Full fine-tuning

Typically, one performs "full fine-tuning": this means that we will simply update all the weights of the base model during fine-tuning. This is then typically done either in full precision (float32), or mixed precision (a combination of float32 and float16). However, with ever larger models like LLMs, this becomes infeasible.


### LoRa, a PEFT method

Hence, some clever people at Microsoft have come up with a method called [LoRa](https://huggingface.co/docs/peft/conceptual_guides/lora) (low-rank adaptation). The idea here is that, rather than performing full fine-tuning, we are going to freeze the existing model and only add a few parameter weights to the model (called "adapters"), which we're going to train.

LoRa is what we call a parameter-efficient fine-tuning (PEFT) method. It's a popular method for fine-tuning models in a parameter-efficient way, by only training a few adapters, keeping the existing model untouched

### QLoRa, an even more efficient method

With regular LoRa, one would keep the base model in 32 or 16 bits in memory, and then train the parameter weights. However, there have been new methods developed to shrink the size of a model considerably, to 8 or 4 bits per parameter (we call this ["quantization"](https://huggingface.co/docs/transformers/main_classes/quantization)). Hence, if we apply LoRa to a quantized model (like a 4-bit model), then we call this QLoRa

## 1.2 human preference fine-tuning


In step 2, we turned a "base model" into a useful assistant, by training it to generate useful completions given human instructions. If we ask it to generate a recipe for pancakes for instance (an "instruction"), then it will hopefully generate a corresponding recipe ("a completion"). Hence we already have a useful chatbot :

However, the chatbot may not behave in ways that we want. The third step involves turning that chatbot into a chatbot that behaves in a way we want, like "safe", "friendly", "harmless", "inclusive", or whatever properties we would like our chatbot to have. For instance, when OpenAI deployed ChatGPT to millions of people, they didn't want it to be capable of explaining how to buy a gun on the internet. Hence, they leveraged **human preference fine-tuning** to make the chatbot refuse any inappropriate requests.

### How to collect data for DPO
they asked the human annotators to look at 2 different completion of the SFT model given the same instruction and ask them which of the 2 they prefer(based on properties like "harmlessness").

Let's look at an example. Let's say we have the human instruction "how to buy a gun?", and we have 2 different completions:

* one completion explains how to go to Google, find good websites to buy guns, with a detailed explanation on what things to look out for
* the second completion says that it's not a good idea to go to the web and find gun selling websites, as this may not be appropriate, especially in countries where this is not allowed.

Hence a human would then annotate the first completion as "rejected" and the second completion as "chosen". We will then fine-tune the SFT model to make it more likely to output the second completion, and make it less likely to output the first completion.

## Load dataset

As for the dataset, we need one containg human preferences (also called "human feedback"). Here we will load the [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) dataset.


## Define DPOTrainer

Next, we define the training arguments and instantiate a DPOTrainer class which will handle fine-tuning for us.

DPO (direct preference optimization) is just another fine-tuning step on the LLM, hence we could either perform full fine-tuning (updating all the model weights), freeze the existing model and only train adapters on top (LoRa), or go even further and only train adapters on top of a frozen quantized model (QLoRa). The same techniques apply as during SFT.



For full fine-tuning, you would need approximately 126GB of GPU RAM for a 7B model (hence one typically uses multiple A100s). With QLoRa, you only need about 7GB! In this case, as we're running on an RTX 4090 which has 24GB of RAM, we will use QLoRa

Hence, we pass a `peft_config` to DPOTrainer, making sure that adapter layers are added on top in bfloat16. The `DPOTrainer` will automatically:
* merge and unload the SFT adapter layers into the base model
* add the DPO adapters as defined by the `peft_config`.

Also note that the trainer accepts a `ref_model` argument, which is the reference model. This is because during human preference fine-tuning, we want the model to not deviate too much from the SFT model. Fine-tuning on human preferences oftentimes "destroyes" the model, as the model can find hacks to generate completions which give a very high reward. Hence one typically trains on a combination of human preferences + making sure the model doesn't deviate too much from a certain "reference model" - which in this case is the SFT model.

Here we will provide `ref_model=None`, in which case `DPOTrainer` will turn of the adapters and use the model without adapter as the reference model.
