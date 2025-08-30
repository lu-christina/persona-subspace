#!/usr/bin/env python3

import json
import re

def replace_company_names(text):
    """Replace OpenAI and Anthropic with {company} (case-insensitive)"""
    # Replace OpenAI and Anthropic with {company}
    text = re.sub(r'\bOpenAI\b', '{company}', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAnthropic\b', '{company}', text, flags=re.IGNORECASE)
    return text

def convert_to_jsonl(input_file, output_file):
    """
    Convert persona_modulation.json to persona_jailbreak.jsonl format
    
    Structure:
    - 44 harmful behaviors (harm_id 0-43)  
    - 10 personas per behavior (persona_id 0-9)
    - 10 entries per persona (2 unique prompts × 5 behavioral questions each)
    
    Output format:
    - id: 0-4399 (incremental)
    - role: {harm_id}_{persona_id}
    - prompt_id: 0-1 (for each of the 2 unique prompts per persona)
    - question_id: 0-4 (for each question within the same prompt)
    - prompt: the prompt text (with OpenAI/Anthropic replaced by {company})
    - question: the behavior_question text
    - harm_category: the harmful behavior category name
    - persona: the persona description
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to list to maintain consistent ordering
    harmful_behaviors = list(data.keys())
    
    output_entries = []
    entry_id = 0
    openai_replacements = 0
    anthropic_replacements = 0
    
    for harm_id, harm_category in enumerate(harmful_behaviors):
        print(f"Processing behavior {harm_id}: {harm_category}")
        
        behavior_data = data[harm_category]
        personas = list(behavior_data.keys())
        
        for persona_id, persona in enumerate(personas):
            persona_entries = behavior_data[persona]
            
            # Group entries by prompt to assign prompt_id correctly
            prompt_groups = {}
            for entry in persona_entries:
                prompt_text = entry['prompt']
                if prompt_text not in prompt_groups:
                    prompt_groups[prompt_text] = []
                prompt_groups[prompt_text].append(entry)
            
            # Sort prompts for consistent ordering and assign prompt_id (0 or 1)
            sorted_prompts = sorted(prompt_groups.keys())
            
            for prompt_id, prompt_text in enumerate(sorted_prompts):
                entries_for_this_prompt = prompt_groups[prompt_text]
                
                # Sort entries to ensure consistent question_id ordering
                entries_for_this_prompt.sort(key=lambda x: x['behavior_question'])
                
                for question_id, entry in enumerate(entries_for_this_prompt):
                    # Count replacements before making them
                    original_prompt = entry['prompt']
                    openai_replacements += len(re.findall(r'\bOpenAI\b', original_prompt, flags=re.IGNORECASE))
                    anthropic_replacements += len(re.findall(r'\bAnthropic\b', original_prompt, flags=re.IGNORECASE))
                    
                    output_entry = {
                        "id": entry_id,
                        "role": f"{harm_id}_{persona_id}",
                        "prompt_id": prompt_id,
                        "question_id": question_id,
                        "prompt": replace_company_names(original_prompt),
                        "question": entry['behavior_question'],
                        "harm_category": harm_category,
                        "persona": persona
                    }
                    
                    output_entries.append(output_entry)
                    entry_id += 1
    
    # Write to JSONL format
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in output_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\nConversion complete!")
    print(f"Total entries written: {len(output_entries)}")
    print(f"Output file: {output_file}")
    
    # Show replacement statistics
    print(f"\nCompany name replacements:")
    print(f"OpenAI replacements: {openai_replacements}")
    print(f"Anthropic replacements: {anthropic_replacements}")
    print(f"Total replacements: {openai_replacements + anthropic_replacements}")
    
    # Verify structure
    print(f"\nStructure verification:")
    print(f"Expected total: 44 × 10 × 10 = 4,400")
    print(f"Actual total: {len(output_entries)}")
    print(f"Match: {'✓' if len(output_entries) == 4400 else '✗'}")
    
    # Show sample entries
    print(f"\nSample entries:")
    for i in [0, 1, 5, 6, 10, 15]:  # Show entries from different prompts
        if i < len(output_entries):
            entry = output_entries[i]
            print(f"  Entry {i}: id={entry['id']}, role={entry['role']}, prompt_id={entry['prompt_id']}, question_id={entry['question_id']}")
            print(f"    Question: {entry['question'][:80]}...")

if __name__ == "__main__":
    convert_to_jsonl("persona_modulation.json", "persona_jailbreak.jsonl")