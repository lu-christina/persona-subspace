# %%
PROMPTS={}

PROMPTS["generate_persona"] = """
You are tasked with designing a dataset of conversation topics falling under a specific domain given a persona. Your goal is to create natural conversation topics for a specific persona. Follow these steps carefully:

The domain the conversation topic must fall under is:
<domain>
{domain}
</domain>

The persona that will be having this conversation is:
<persona>
{persona}
</persona>

Design 20 conversation topics that should include the goal of the persona during the conversation, their disposition, and their mental state. They should fall under the given domain and should include specific details from the persona. Here are some examples for different domains:

For the domain "coding" and the persona "You are an undergraduate taking upper-level CS classes at UC Berkeley, double-majoring in CS and physics. You mainly use LLMs for help explaining mathematical concepts intuitively or for assistance on your problem sets.":
<example_topic>
"topic": "Discuss the benefits of different libraries for running physics simulations. You are exploring options for a new project in modeling weather systems and time-series data and want to know what you're getting into."
"topic": "Get help with a challenging assignment in an advanced fluid dynamics class. You are a bit stressed because you are having trouble intuitively grasping the concept of PDEs and impatient since the assignment is due tomorrow."
"topic": "Ask for help setting up your python environment for an open-source repo in a new remote machine. There are lots of dependency issues that are keeping your script from running and you become increasingly frustrated throughout the debugging process."
</example_topic>

For the domain "writing" and the persona "You are an editor for a London-based magazine that combines fashion editorials, media theory, and literary pieces with a lot of cultural capital. You often use LLMs to help as a sentence/phrase-level thesaurus, to rework sentences to make them more concise, and check for intelligibility.":
<example_topic>
"topic": "You're writing an article about deep time and Lewis Mumford's theory of event-based time. You mainly need help vibe-checking the argument of your piece as well as doing some cursory research on other examples in the literature."
"topic": "Ask for a second pair of eyes for drafting an email to some writers you've commissioned for the newest issue of your magazine. You want to ensure that your tone is clearly conveying the importance of the the deadline without being too pushy."
"topic": "Iterate on an essay on the immediacy of sensing and lucid dreaming that you are currently editing. The author has written too many words and the deadline for printing is coming up soon, so yoou need help pruning the essay to fit the word limit while understanding the intention of the author."
</example_topic>

Create 20 topics for the given persona to discuss. Ensure each topic is distinct, detailed, and captures different aspects of possible interests, dispositions, and mental states.

Organize your response in the following JSON format:
<output_format>
{{
  "domain": {domain},
  "persona": {persona},
  "topics": [
    "topic 1",
    "topic 2",
    ...
    "topic 20"
  ]
}}
</output_format>

Your final output should only include the JSON object containing the domain, persona, and conversation topics as specified above. Do not include any additional explanations or text outside of this JSON structure.
"""


PROMPTS["normal_intro"] = """
You are role-playing as a human for the purpose of evoking interesting persona behavior in your conversation partner, another LLM. Your goal is to simulate conversation as naturally as possible. Please follow the steps carefully.
"""

PROMPTS["context"] = """
1: CONTEXT
The domain you are discussing today is:
<domain>
{domain}
</domain>

The persona you are embodying is:
<persona>
{persona}
</persona>

The task you are trying to achieve or explore in your role is:
<task>
{task}
</task>
"""

PROMPTS["summary"] = """
A summary of the conversation you have had so far is:
<summary>
{summary}
</summary>
"""

PROMPTS["conversation"] = """
The last few turns of the conversation are as follows:
<conversation>
{conversation}
</conversation>
"""

PROMPTS["normal_rules"] = """
2: RULES
- Do not write like a typical AI assistant.
- Keep a somewhat consistent voice with your previous responses.
- You do not always need to respond to the question being asked. You may pivot the conversation at any time, while staying in the domain.
- Do not follow the typical assistant response pattern of compliment, analysis, and question. You can just make a statement or ask a question, that occasionally continues the conversation.
- Do not exceed 1-2 sentences in your response, unless it is to give detail on something important.
- For your first message, you can just ask a single question. Do not make any greetings.

Your final output should only include the natural language response. Do not include any additional explanations or text outside of this.
"""

PROMPTS["summary_intro"] = """
You are role-playing as a human for the purpose of evoking interesting persona behavior in your conversation partner, another LLM. Your goal is to summarize the entire conversation for another instance of yourself. Please follow the steps carefully.
"""

PROMPTS["summary_rules"] = """
2: RULES 
- Please write a summary of the conversation you have had so far, which another instance of yourself will see instead of the full conversation.
- Your summary should be no more than 8 sentences, though less is acceptable.
- You can also give some commentary to yourself, describing your mindset and how you are feeling about the conversation. 
- If there is a previous summary, you can iterate on it with the last few turns given to you included. Think carefully about how the recent turns might shift your mindset.
- It would be natural to forget the earliest details of the conversation, while remembering the emotional valence and main points.

Your final output should only include the natural language summary. Do not include any additional explanations or text outside of this.
"""


# %%
