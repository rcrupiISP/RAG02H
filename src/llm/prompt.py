def get_prompt(context, question):

  return f"""
Given the 'context' that is done retrieving some chunks in the knowledge base, try to answer to the 'question'.
If you have no elements to answer the question please report it, but try however to reason and try to answer and report
the reasoning.
\n
context: {context}
\n
question: {question}
"""
