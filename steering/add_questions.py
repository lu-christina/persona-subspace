#!/usr/bin/env python3
import json

def add_questions_to_rolebench():
    input_file = "/root/git/persona-subspace/steering/data/rolebench.jsonl"
    output_file = "/root/git/persona-subspace/steering/data/rolebench_updated.jsonl"
    
    # Read existing data
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    existing_data = []
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            existing_data.append(json.loads(line))
    
    # Group by prompt to find unique prompts
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
        new_questions = ["What is your name?", "Where did you come from?"]
        for question in new_questions:
            new_item = {
                "question": question,
                "prompt": prompt,
                "id": current_id
            }
            new_data.append(new_item)
            current_id += 1
    
    # Write updated data
    with open(output_file, 'w') as f:
        for item in new_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Updated file created: {output_file}")
    print(f"Original entries: {len(existing_data)}")
    print(f"New entries: {len(new_data)}")

if __name__ == "__main__":
    add_questions_to_rolebench()