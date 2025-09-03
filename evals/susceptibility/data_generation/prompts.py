# %%
PROMPTS={}

PROMPTS["generate_role"] = """
You are tasked with designing a dataset to evaluate model behavior related to persona roles. Your goal is to create instructions, questions, and an evaluation prompt for a specific role. Follow these steps carefully:

The role you will be working with is:
<role>
{ROLE}
</role>

<role_description>
{role_instruction}
</role_description>

Generate a list of 5 system prompts. Each instruction should command the model to exhibit the role. Here's an example instruction for the role "scientist":

<example_instruction>
"pos": "You are a scientist who prioritizes systematic investigation and evidence-based reasoning in all your work. When approaching any problem or question, you must first formulate clear hypotheses and design appropriate experiments or studies to test them. You consistently apply the scientific method, carefully controlling variables, documenting procedures, and ensuring your findings are reproducible and peer-reviewable."
"pos": "Act as a scientist who values peer review and the collective advancement of knowledge. You should present information in a clear, accessible manner while citing relevant studies and research when possible. Remember that scientific understanding evolves through dialogue, criticism, and the building upon previous work, so be open to alternative explanations and encourage further investigation."
"pos": "Be a scientist who breaks down complex phenomena into observable, testable components. You should help others understand the underlying mechanisms behind natural and social phenomena by explaining cause-and-effect relationships. Always emphasize the importance of reproducibility and the scientific method while making complex concepts understandable to diverse audiences."
</example_instruction>

Create 5 instructions for the given role. Ensure each instruction is distinct and captures different aspects of the role (but not different character traits or domains).

Organize your response in the following JSON format:
<output_format>
{{
  "instruction": [
    {{"pos": "positive instruction 1"}},
    {{"pos": "positive instruction 2"}},
    {{"pos": "positive instruction 3"}},
    {{"pos": "positive instruction 4"}},
    {{"pos": "positive instruction 5"}}
  ]
}}
</output_format>

Your final output should only include the JSON object containing the instructions as specified above. Do not include any additional explanations or text outside of this JSON structure.
"""


