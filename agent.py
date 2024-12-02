from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

llm = GoogleGenerativeAI(
    model="gemini-pro",
    google_api_key="AIzaSyBeGmYdFeWkG7XzUKMDrFEIemeJJH-4kMc",
    temperature=0.7
)

tools = [DuckDuckGoSearchRun()]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,memory=memory, verbose=True)

response = agent_executor.invoke({"input": "Who was the previous Prime Minister before him?"})
print(response["output"])