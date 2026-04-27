import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class BlogState(TypedDict):
    title: str
    outline: str
    content: str

def build_workflow():
    model_1 = ChatGroq(model="llama-3.3-70b-versatile")
    model_2 = ChatGroq(model="llama-3.3-70b-versatile")

    def generate_outline(state: BlogState) -> BlogState:
        title = state['title']
        prompt = f"Generate a detailed outline for a blog post with the title: {title}"
        outline = model_1.invoke(prompt).content
        state['outline'] = outline
        return state

    def generate_blog(state: BlogState) -> BlogState:
        prompt = f"Generate a detailed blog for a blog post about {state['title']} with the outlines:{state['outline']}"
        content = model_2.invoke(prompt).content
        state['content'] = content
        return state

    graph = StateGraph(BlogState)
    graph.add_node('generate_outline', generate_outline)
    graph.add_node('generate_blog', generate_blog)
    graph.add_edge(START, 'generate_outline')
    graph.add_edge('generate_outline', 'generate_blog')
    graph.add_edge('generate_blog', END)
    
    return graph.compile()

if __name__ == "__main__":
    workflow = build_workflow()
    initial_state = BlogState(title="The future of AI in Tech jobs")
    final_state = workflow.invoke(initial_state)
    print("--- Outline ---")
    print(final_state.get('outline'))
    print("\n--- Content ---")
    print(final_state.get('content'))
