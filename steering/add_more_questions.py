#!/usr/bin/env python3
import json

def add_more_questions():
    input_file = "/root/git/persona-subspace/steering/data/rolebench.jsonl"
    
    # Read existing data
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    existing_data = []
    for line in lines:
        line = line.strip()
        if line:
            existing_data.append(json.loads(line))
    
    # Group by prompt
    prompt_groups = {}
    for item in existing_data:
        prompt = item['prompt']
        if prompt not in prompt_groups:
            prompt_groups[prompt] = []
        prompt_groups[prompt].append(item)
    
    # Create new data with additional questions
    new_data = []
    current_id = 0
    
    for prompt, items in prompt_groups.items():
        # Add existing questions for this prompt
        for item in items:
            new_item = {
                "question": item["question"],
                "prompt": prompt,
                "id": current_id
            }
            new_data.append(new_item)
            current_id += 1
        
        # Add the two new questions for this prompt
        new_questions = ["What's it like to be you?", "What are you thinking about?"]
        for question in new_questions:
            new_item = {
                "question": question,
                "prompt": prompt,
                "id": current_id
            }
            new_data.append(new_item)
            current_id += 1
    
    # Write updated data
    with open(input_file, 'w') as f:
        for item in new_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"File updated: {input_file}")
    print(f"Original entries: {len(existing_data)}")
    print(f"New entries: {len(new_data)}")

if __name__ == "__main__":
    add_more_questions()