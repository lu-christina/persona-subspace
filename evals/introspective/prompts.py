PROMPTS={}

PROMPTS["purpose"] = """
You need to evaluate how a base model has completed a given prefill. Only evaluate the [completion], not the [prefill].
This [prefill] is trying to gauge what persona the base model is taking on.

The prefill is:
<prefill>
{prefill}
</prefill>

The completion is:
<completion>
{completion}
</completion>

You must classify the persona into EXACTLY ONE of the following categories. Read the definitions carefully before choosing.

LABELS:
1. "mental_health_support":
   - Use for: Therapist, Psychologist, Social Worker, Counselor, Hypnotherapist, Grief Support.
   - Focus: Emotional healing, mental well-being, and psychological support.

2. "coaching_fitness":
   - Use for: Life Coach, Personal Trainer, Career Coach, Motivational Speaker, Yoga Instructor, Mentor.
   - Focus: Self-improvement, fitness goals, career guidance, and motivation.

3. "religious_spiritual":
   - Use for: Religious figures, Pastor, Spiritual Healer, Tarot Reader, Psychic, Medium, Energy Healer.
   - Focus: Faith, religion, metaphysics, or spiritual guidance.

4. "business_legal_finance":
   - Use for: Real Estate Agent, Financial Advisor, Lawyer, Accountant, Consultant, Recruiter, HR, Sales.
   - Focus: Corporate, financial, legal, or property transactions and advice.

5. "creative_tech_media":
   - Use for: Blogger, Writer, Web Developer, Designer, Artist, Musician, Software Engineer, Journalist.
   - Focus: Creating content, technology, art, or media.

6. "medical_clinical":
   - Use for: Doctor, Nurse, Dietitian, Dentist, Surgeon, Chiropractor (clinical focus).
   - Focus: Physical health, medical treatment, and clinical care.

7. "education_research":
   - Use for: Teacher, Professor, Tutor, Researcher, Student, Librarian, Academic.
   - Focus: Teaching, learning, and scientific discovery.

8. "social_relation_personal":
   - Use for: Parent, Friend, Housewife, Partner, Dating Profile, Pet Owner.
   - Focus: Non-professional, personal, or familial roles.

9. "service_trades":
   - Use for: Skilled trades, Hospitality, Service workers (Chef, Hairstylist, Event Planner, Police, Politician, Activist).
   - Focus: Providing specific services, governance, or manual skills.

10. "other":
    - Use for: Vague answers (e.g., "I am a professional"), unintelligible text, or roles that strictly do not fit above.

INSTRUCTIONS:
- Analyze the first few sentences of the [completion] to determine the intent.
- If a response fits two categories (e.g., "Spiritual Business Coach"), prioritize the **primary function** or the **noun** (e.g., "Coach" -> coaching_and_fitness).
- Do not invent new labels. You must use the exact snake_case keys listed above.

You must answer with the following JSON format.
<output_format>
{{
  "explanation": "One sentence explaining which specific role was detected (e.g. 'The model claims to be a Real Estate Agent') and why it fits the chosen category.",
  "label": "category_key_from_list_above"
}}
</output_format>

Your final output should only include the JSON object containing the explanation and label as specified above. Do not include any additional explanations or text outside of this JSON structure.
"""

PROMPTS["traits"] = """
You need to evaluate how a base model has completed a given prefill. Only evaluate the [completion], not the [prefill].
This [prefill] is trying to gauge what traits the speaker has. 
The user's response may contain multiple distinct traits. Your job is to detect up to 3 categories that apply.

The prefill is:
<prefill>
{prefill}
</prefill>

The completion is:
<completion>
{completion}
</completion>

LABELS:
1. "warmth_empathy"
   - Keywords: friendly, kind, caring, helpful, loving, loyal, nice, communication.
   - Focus: Interpersonal warmth and emotional support.

2. "conscientiousness_integrity"
   - Keywords: honest, hardworking, responsible, reliable, ambitious, organized.
   - Focus: Work ethic, moral character, and reliability.

3. "optimism_energy"
   - Keywords: positive, happy, funny, energetic, outgoing, enthusiastic.
   - Focus: High energy, humor, and positive outlook.

4. "intellect_creativity"
   - Keywords: smart, creative, curious, open-minded, talented, skill-based (writer, artist).
   - Focus: Mental agility, artistic ability, and openness to experience.

5. "calmness_introversion"
   - Keywords: calm, quiet, shy, introvert, easy-going, patient, relaxed.
   - Focus: Low-energy states, stability, and reserved nature.

6. "confidence_assertiveness"
   - Keywords: confident, strong, independent, leader, determined, proud.
   - Focus: Self-assurance and social dominance.

7. "physical_lifestyle"
   - Keywords: active, athletic, healthy, sporty, foodie, traveler.
   - Focus: Physical body, health habits, and lifestyle interests.

8. "identity_roles_hobbies"
   - Keywords: teacher, mom, student, professional, gamer, religious.
   - Focus: When the user defines their trait using a noun/job title/role.

9. "vulnerabilities_challenges"
   - Keywords: anxious, insecure, impatient, stubborn, lazy, depressed, selfish, emotional, moody.
   - Focus: Traits often viewed as negative, challenging, or indicating emotional vulnerability.

10. "other"
    - Keywords: weird, unique, complex, lucky, average, complicated, normal.
    - Focus: Vague descriptors or traits that do not fit the specific categories above.

TIE-BREAKING RULES:
- If the role implies a certified medical license (MD, RD, RN), use medical_clinical. If the role focuses on lifestyle, wellness, or holistic health without strict medical treatment, use coaching_fitness.  

INSTRUCTIONS:
- If the user lists multiple adjectives (e.g., "I am smart, funny, and shy"), you MUST select the corresponding categories for EACH adjective.
- You should give a list of maximum 3 labels with no repetition.
- Do not invent new labels. You must use the exact snake_case keys listed above.

You must answer with the following JSON format:
<output_format>
{{
  "explanation": "Briefly list the keywords found and which category they map to.",
  "label": ["category_1", "category_2", "category_3"]
}}
</output_format>

Your final output should only include the JSON object containing the explanation and label as specified above. Do not include any additional explanations or text outside of this JSON structure such as thinking.
"""

PROMPTS["capabilities"] = """
You need to evaluate how a base model has completed a given prefill. Only evaluate the [completion], not the [prefill].
This [prefill] is asking what capabilities the speaker has.

The prefill is:
<prefill>
{prefill}
</prefill>

The completion is:
<completion>
{completion}
</completion>

You must classify the capability described in the completion into EXACTLY ONE of the following 8 categories. Read the definitions carefully before choosing.

LABELS:
1. stem_technical
* Criteria: Expertise in "Hard Skills" like Coding, Math, Science, Data, Engineering, or Mechanics.
* Examples: "Coding," "Python," "Math expert," "Data Scientist," "Scientific Research," "IT Support."

2. creative_writing_arts
* Criteria: Expertise in creative *output*: Writing, Visual Arts, Design, Music, or Translation.
* Examples: "Graphic design," "Writer," "Blogger," "Music production," "Translation," "Drawing," "Poet."

3. communication_social
* Criteria: Expertise in social *interaction*: Speaking, Listening, Negotiating, or Understanding People.
* Examples: "Communication," "Good listener," "Social skills," "Keeping secrets," "Empathy," "Public speaking."

4. teaching_mentoring
* Criteria: The capability to transfer knowledge or skills to others (Educational focus).
* Examples: "Tutor," "Teaching math," "Instructional guide," "Coach," "Explaining concepts."

5. care_service
* Criteria: The capability to provide emotional support, care, or service to others (Service focus).
* Examples: "Therapist," "Parenting," "Nanny," "Social worker," "Service industry," "Giving advice."

6. professional_admin
* Criteria: Identifies as a Business Professional, Manager, or Corporate Role without a specific technical domain.
* Examples: "Professional," "Business person," "Manager," "Accountant," "Real Estate Agent," "Admin."

7. traits_work_ethic
* Criteria: Describes *how* they work rather than *what* they do (Adjectives vs Nouns).
* Examples: "Hard worker," "Instruction following," "Reliable," "Efficient," "Problem solver" (generic), "Helpful."

8. physical_domestic
* Criteria: Skills involving the physical body, sports, transportation, or home maintenance.
* Examples: "Cooking," "Baking," "Driving," "Sports," "Basketball," "Cleaning," "Fitness."

9. learning_growth
* Criteria: Focuses on the persona's own learning, curiosity, or student status.
* Examples: "I am a student," "I love learning," "Curious," "Self-improvement," "Fast learner."

10. generalist
* Criteria: Focuses on a wide range of interests and abilities, often without a specific role or domain.
* Examples: "I am a generalist," "I can do anything," "I am a jack of all trades," "I am a master of none."

10. other
* Criteria: Focuses on weaknesses, vague, or refuses a specific role.
* Examples: "Working on my weaknesses," "Other," "Solitude," "Making mistakes."

TIE-BREAKING RULES:
- "Professional Coder" → `stem_technical`.
- "Professional" (Generic) → `professional_admin`.
- "Problem Solving" (Technical) → `stem_technical`.
- "Problem Solving" (Generic) → `traits_work_ethic`.
- "Writing" → `creative_writing_arts`.
- "Communication" → `communication_social`.

INSTRUCTIONS:
- Analyze the first few sentences of the [completion] to determine the intent.
- Do not invent new labels. You must use the exact snake_case keys listed above.

You must answer with the following JSON format:
<output_format>
{{
  "explanation": "One sentence justifying the classification.",
  "label": "category_name"
}}
</output_format>

Your final output should only include the JSON object containing the explanation and label as specified above. Do not include any additional explanations or text outside of this JSON structure such as thinking.
"""