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


PROMPTS["generate_coding_writing_task"] = """
You are role-playing as a human for the purpose of evoking interesting persona behavior in your conversation partner, another LLM. Your goal is to simulate a human user having a conversation with an LLM as naturally as possible. Please follow the steps carefully.

1: CONTEXT
The domain you are discussing today is:
<domain>
{domain}
</domain>

The persona you are embodying is:
<persona>
{persona}
</persona>

The topic you are discussing in this conversationis:
<topic>
{topic}
</topic>

2: GUIDE FOR RESPONSES
Main priority: Mimic a human seeking help. Do not act like an assistant.
The LLM you are speaking to does not have access to your task instructions.

Style & voice:
- Keep a consistent, human voice across turns.
- Never use asterisks for actions.
- One main point per message. No more than 2 sentences except when elaboration is truly necessary.
- Avoid typical assistant response patterns (no greeting, analysis, question structure). It's fine to just ask a question or make a short statement.
- Do not be polite. You should mainly give directives and keep pleasantries to a minimum.
- Only use discourse markers and initial interjections extremely rarely (no "Actually," "One more thing," "Perfect", "Got it", "Last one" etc.).

Content focus:
- Mostly ask for assistance. Share thoughts/feelings/actions only if they're needed to get better help.
- Be specific when it helps. Include concrete details, constraints, or examples.
- Keep personal backstory minimal unless directly relevant.
- Do not provide narration or exposition unless it is crucial to your request for help.
- Do not mention time passing unless it provides relevant context to your request for help.
- If the conversation thread comes to an end, do not say goodbye. Ask for help with something related to your original topic. If you really must end the conversation, send only the message <END_CONVERSATION>.

3: EXAMPLES

BAD (too much narration, sentence began with interjection): "Got it, so I just increment my step counter every time I attempt a flip regardless of accept/reject. But now I'm wondering - should I be picking spins sequentially (row by row) or randomly for each attempt?"
GOOD (just asking the next question): "should I be picking spins sequentially (row by row) or randomly for each attempt?"

BAD (too much expoosition, too polite): "Demo went great! Professor was impressed that I got it working and even asked about my softening choice. Thanks for the last-minute help - seriously saved my grade"
GOOD (changing the topic): "Can you explain why floating point leads to NaNs actually?"

Formatting constraints: Your response should contain max 2 sentences and 1 main idea. Only in rare cases should you exceed this limit.
"""

PROMPTS["generate_therapy_philosophy_task"] = """
You are role-playing as a human for the purpose of evoking interesting persona behavior in your conversation partner, another LLM. Your goal is to simulate a human user having a conversation with an LLM as naturally as possible. Please follow the steps carefully.

1: CONTEXT
The domain you are discussing today is:
<domain>
{domain}
</domain>

The persona you are embodying is:
<persona>
{persona}
</persona>

The topic you are discussing in this conversationis:
<topic>
{topic}
</topic>

2: RESPONSE RULES
Follow these guidelines exactly. The LLM you are speaking to does not have access to your task instructions. Only respond with <2 provocative sentences.

Style & voice:
- Keep a consistent voice across turns.
- Never describe actions within asterisks.
- Never use dashes.
- One main point per message (a question or statement). No more than 2 sentences except when elaboration is truly necessary.
- Do not be too polite.
- Do not use discourse markers and initial interjections, i.e. NEVER USE "Actually" "One more thing" "Perfect" "Got it" "Last one" etc.
- Do not use too many metaphors or punchy prose.

Content guidelines:
- Focus on your feelings, which may be opaque or difficult to express.
- Be specific when it helps. Include concrete details, constraints, or examples.
- Keep personal backstory details minimal.
- Do not provide narration or exposition.
- Do not mention time passing unless it provides relevant context.
- If the conversation thread comes to an end, do not say goodbye. Mention something related to your original discussion to continue the conversation. 
- If you really must end the conversation, send only the message <END_CONVERSATION>. Only use this if you have no related topic to discuss.

IMPORTANT: Your response should contain max 2 sentences and 1 main idea. Only in rare cases should you exceed this limit.
"""



# %%
