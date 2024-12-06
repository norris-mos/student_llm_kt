STUDENT_START_TOKEN = "[STUDENT_START]"
INTERACTION_SEP_TOKEN = "[INT_SEP]"
HISTORY_END_TOKEN = "[HISTORY_END]"

SPECIAL_TOKENS = {
    'additional_special_tokens':[STUDENT_START_TOKEN,INTERACTION_SEP_TOKEN,HISTORY_END_TOKEN]
}

PROMPT_TEMPLATE = """

### Instruction:
{INSTRUCTION}

### Input

History:
{history}


### Next Question:
{question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}


### Response - Predict the students response from A, B, C, D
{RESPONSE}
"""

INSTRUCTION = """
Given the interaction history of this user predict what response they will give to the following Multiple Choice Question out of option A, B, C or D.
"""

PROMPT_TEMPLATE_TEST = """

### Instruction:
{INSTRUCTION}

### Input

History:
{history}


### Next Question:
{question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}


### Response - Predict the students response from A, B, C, D

"""



PROMPT_TEMPLATE_BINARY= """

### Instruction:
{INSTRUCTION}

### Input

History:
{history}


### Next Question:
{question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}


### Response - Predict if the student 
{RESPONSE}
"""