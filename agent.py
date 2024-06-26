from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import tool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents import create_openai_functions_agent, AgentExecutor, AgentType, create_tool_calling_agent, Tool, create_react_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Start....")

    tools = [PythonREPLTool()]
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)

    agent = create_openai_functions_agent(ChatOpenAI(temperature=0), tools, prompt)

    python_agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # python_agent_executor.invoke(
    #     {
    #         "input": "generate and save in current working directory 2 QRcodes that point to https://www.facebook.com/mainul.mif"
    #     }
    # )


    csv_agent = create_csv_agent(
        ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        path="./episode-info.csv",
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        allow_dangerous_code=True
    )
    # csv_agent.run("How many columns are there in file csv file")
    # csv_agent.run("which writer wrote the most episode? How many episodes did he write?")

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    tools_agent=[
        Tool(
            name="PythonAgent",
            func=python_agent_executor.invoke,
            description= """useful when you need to transform natural language and write from it python and execute the python code, returning the results of the code execution, DO NOT SEND PYTHON CODE TO THIS TOOL"""
        ),
        Tool(
            name="CSVAgent",
            func=csv_agent.run,
            description= """useful when you need to answer question over episode_info.csv file, takes an input the entire question and returns the answer after running pandas calculations"""
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    grand_agent = create_react_agent(llm, tools_agent, react_prompt)

    agent_executor = AgentExecutor(
        agent=grand_agent, tools=tools, handle_parsing_errors=True, verbose=True, max_iterations=5
    )

    input = (
        "generate and save in current working directory 2 QRcodes that point to https://github.com/mainul-islam-faruqi, you have qrcode package installed already"
    )
    agent_executor.invoke({"input": input})

    
if __name__ == "__main__":
    main()
