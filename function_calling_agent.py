from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool , FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
import os
os.environ['OPENAI_API_KEY'] = 'openai-APIKEY'
def calculate_relative_speed(speed_1: float, speed_2: float) -> float:
    relative_speed = speed_1 + speed_2
    return relative_speed

# Step 2: Reason about the time it will take for the objects to meet
def calculate_meeting_time(distance: float, relative_speed: float) -> float:

    if relative_speed > 0:
        time_to_meet = distance / relative_speed

        return time_to_meet
    else:

        return float('inf')  # Infinite time

# Step 3: Reason about when the objects meet (time and distance)
def calculate_meeting_point(time_to_meet: float, speed_1: float) -> float:

    distance_covered_by_1 = speed_1 * time_to_meet

    return distance_covered_by_1

# Creating function tools for reasoning steps
relative_speed_tool = FunctionTool.from_defaults(fn=calculate_relative_speed)
meeting_time_tool = FunctionTool.from_defaults(fn=calculate_meeting_time)
meeting_point_tool = FunctionTool.from_defaults(fn=calculate_meeting_point)


llm = OpenAI(model="gpt-4")

agent_worker = FunctionCallingAgentWorker.from_tools(
    [relative_speed_tool, meeting_time_tool, meeting_point_tool],
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=False,
)
agent = AgentRunner(agent_worker)

response = agent.chat("distance between trains 300 km, speed of first train 100 km/hr , speed of second train 80 km/hr, calculate their meeting point")


===================================
Added user message to memory: distance between trains 300 km, speed of first train 100 km/hr , speed of second train 80 km/hr, calculate their meeting point
=== Calling Function ===
Calling function: calculate_relative_speed with args: {"speed_1": 100, "speed_2": 80}
=== Function Output ===
180
=== Calling Function ===
Calling function: calculate_meeting_time with args: {"distance": 300, "relative_speed": 180}
=== Function Output ===
1.6666666666666667
=== Calling Function ===
Calling function: calculate_meeting_point with args: {"time_to_meet": 1.6666666666666667, "speed_1": 100}
=== Function Output ===
166.66666666666669
=== LLM Response ===
The meeting point of the two trains is approximately 166.67 km from the starting point of the first train.