import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class LLMState(TypedDict):
    question: str
    answer: str

def build_workflow():
    model = ChatGroq(model="llama-3.3-70b-versatile")

    def llm_qa(state: LLMState) -> LLMState:
        question = state['question']
        prompt = f"Answer the following question: {question}"
        answer = model.invoke(prompt).content
        state['answer'] = answer
        return state

    graph = StateGraph(LLMState)
    graph.add_node('llm_qa', llm_qa)
    graph.add_edge(START, 'llm_qa')
    graph.add_edge('llm_qa', END)
    return graph.compile()

if __name__ == "__main__":
    workflow = build_workflow()
    initial_state = LLMState(question="What is the capital of France?")
    final_state = workflow.invoke(initial_state)
    print("Final State:", final_state)
