from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float
    category: str

def calculate_bmi(state: BMIState) -> BMIState:
    weight = state['weight_kg']
    height = state['height_m']
    bmi = weight / (height ** 2)
    state['bmi'] = round(bmi, 2)
    return state

def label_bmi(state: BMIState) -> BMIState:
    bmi = state['bmi']
    if bmi < 18.5:
        category = 'Underweight'
    elif bmi < 25:
        category = 'Normal weight'
    elif bmi < 30:
        category = 'Overweight'
    else:
        category = 'Obese'
    state['category'] = category
    return state

def build_workflow():
    graph = StateGraph(BMIState)
    graph.add_node('calculate_bmi', calculate_bmi)
    graph.add_node('label_bmi', label_bmi)
    graph.add_edge(START, 'calculate_bmi')
    graph.add_edge('calculate_bmi', 'label_bmi')
    graph.add_edge('label_bmi', END)
    return graph.compile()

if __name__ == "__main__":
    workflow = build_workflow()
    input_state = {'weight_kg': 80, 'height_m': 1.73}
    final_state = workflow.invoke(input_state)
    print("Final State:", final_state)
