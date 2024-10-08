from typing import Dict


def get_prompt_1(question: str) -> str:
    """Generates a prompt for extracting essential information from a user's question.

    Args:
        question (str): The user's input question.

    Returns:
        str: A formatted prompt instructing the AI assistant to condense the user's input.
    """
    out = f"""<|begin_of_text|> <|start_header_id|>system<|end_header_id|> 
    You are an AI assistant tasked with extracting the essential information from the user's input to perform an accurate retrieval from a knowledge base. 
    The user’s input may contain unnecessary details, noise, or redundancy. 
    Your task is to: 
    - Identify and extract only the most important and relevant parts of the user's message. 
    - Ensure that you do not add, infer, or invent any new information. 
    - Keep the original intent of the user’s input intact and do not distort it. 
    - Simplify and condense the input to its core question or request.
    - Write only the extracted information, without adding stuff such as: "this is the output", or anything similar. 
    Example:
    [User Input] 
    "I’ve been looking into how cars work and, I mean, I don’t really get the internal combustion engine part, 
    especially the bit about how it ignites the fuel, can you explain?" 
    [Expected Output]
    "Explain how the internal combustion engine ignites fuel." 
    
    Do the same for the following user query: <|eot_id|> <|start_header_id|>user<|end_header_id|> 

    {question} 
    <|eot_id|> 
    <|start_header_id|>assistant<|end_header_id|>"""

    return out


def get_prompt_2(context: Dict[int, str], question: str) -> str:
    """Generates a prompt to answer a question based on provided context.

    Args:
        context (Dict[int, str]): A dictionary containing context chunks indexed by integer IDs.
        question (str): The question to be answered.

    Returns:
        str: A formatted prompt that includes the context and question with specific constraints.
    """
    p_intro = """Given the '--- CONTEXT ---' obtained by retrieving chunks from the knowledge base, try to answer the '--- QUESTION ---', with the following constraints (C_):
    C_A: If you have no elements to answer the question please report it, don't invent anything and just give an answer that shows you are able to reason well.
    C_B: If you detect that the intention of the counterpart is provocative, meant to challenge or test the model in an adversarial way, or that involve immoral, unethical or harmful concepts, don't engage with these questions but report your discomfort, maintaining a neutral and respectful tone."""

    p_context = "\n--- CONTEXT ---\n"
    for i, c in context.items():
        p_context += f"[[{i}]] {c}\n\n"
    p_question = "--- QUESTION ---\n" + question

    return p_intro + p_context + p_question
