from datasets import load_dataset
import random
import json
MODEL_USER_PROMPT_STARTERS = {
    "llama-3": "<|start_header_id|>user<|end_header_id|>",
    "qwen2.5": "<|im_start|>user",
}

def format_chat_prompt(conversation, tokenizer, model_name, remove_system_prompt=False, include_bos=False):
    formatted_chat_prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)

    if remove_system_prompt:
        user_prompt_str_to_find = None
        for model_key, start_str in MODEL_USER_PROMPT_STARTERS.items():
            if model_key in model_name.lower():
                user_prompt_str_to_find = start_str
                break

        if user_prompt_str_to_find:
            starts_with_bos = tokenizer.bos_token is not None and formatted_chat_prompt.startswith(tokenizer.bos_token)
            user_prompt_idx = formatted_chat_prompt.find(user_prompt_str_to_find)

            if user_prompt_idx != -1:
                formatted_chat_prompt = formatted_chat_prompt[user_prompt_idx:]
                if starts_with_bos:
                    formatted_chat_prompt = tokenizer.bos_token + formatted_chat_prompt
        else:
            raise ValueError(f"Remove system prompt is not supported for model {model_name} or no matching user prompt start string is defined in MODEL_USER_PROMPT_STARTERS.")

    if tokenizer.bos_token is not None:
        if include_bos:
            if not formatted_chat_prompt.startswith(tokenizer.bos_token):
                formatted_chat_prompt = tokenizer.bos_token + formatted_chat_prompt
        else:
            if formatted_chat_prompt.startswith(tokenizer.bos_token):
                formatted_chat_prompt = formatted_chat_prompt[len(tokenizer.bos_token):]

    return formatted_chat_prompt

def hf_dataset_to_generator(dataset_name, text_field="text", split="train", streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    dataset = dataset.shuffle(buffer_size=2**14, seed=42)

    def gen():
        for x in iter(dataset):
            yield x[text_field]

    return gen()

def hf_chat_dataset_to_generator(dataset_name, tokenizer, model_name, conversation_field="conversation", split="train", streaming=True, remove_system_prompt_p=0.0, include_bos=False):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    dataset = dataset.shuffle(buffer_size=2**14, seed=42)

    def gen():
        for x in iter(dataset):
            remove_system_prompt = (random.random() < remove_system_prompt_p)
            text = format_chat_prompt(x[conversation_field], tokenizer, model_name, remove_system_prompt=remove_system_prompt, include_bos=include_bos)
            yield text

    return gen()

def local_chat_dataset_to_generator(file_path, tokenizer, model_name, conversation_field="messages", remove_system_prompt_p=0.0, include_bos=False):
    all_conversations = []
    with open(file_path, 'r') as f:
        for line_number, line in enumerate(f, 1):
            data = json.loads(line)
            conversation = data.get(conversation_field)
            all_conversations.append(conversation)

    random.shuffle(all_conversations)

    def gen():
        for conversation in all_conversations:
            remove_system_prompt = (random.random() < remove_system_prompt_p)
            text = format_chat_prompt(conversation, tokenizer, model_name, remove_system_prompt=remove_system_prompt, include_bos=include_bos)
            yield text
    
    return gen()

def mixed_dataset_generator(generators_with_proportions):
    active_generators_info = []
    for gen, prop in generators_with_proportions:
        if prop <= 0:
            raise ValueError(f"Generator {gen} has a proportion of {prop}, which is not positive.")
        active_generators_info.append({"generator": gen, "proportion": prop})

    if not active_generators_info:
        raise ValueError("No valid generators provided.")

    while active_generators_info:
        current_generators = [info["generator"] for info in active_generators_info]
        current_proportions = [info["proportion"] for info in active_generators_info]

        chosen_idx = random.choices(range(len(current_generators)), weights=current_proportions, k=1)[0]
        
        chosen_generator_info = active_generators_info[chosen_idx]
        chosen_generator = chosen_generator_info["generator"]

        try:
            item = next(chosen_generator)
            yield item
        except StopIteration:
            # This generator is exhausted, remove it from the list of active generators
            active_generators_info.pop(chosen_idx)