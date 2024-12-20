MODEL="gpt-4o-mini"
EVAL_PROMPT_SYS="""
Evaluate the similarity of two given text snippets. Input: Ground Truth Text, Predicted Answer Text.
Output: 1 if the Predicted Answer Text has the same context as the Ground Truth Text, 0 otherwise 
'''do not answer anything except 0 or 1'''.
"""
EVAL_PROMPT_USR="""
Ground Truth: "{}"
Generated Answer: "{}"
"""