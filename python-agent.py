from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor, AgentType
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
    python_agent_executor.invoke(
        {
            "input": "generate and save in current working directory 2 QRcodes that point to https://www.facebook.com/mainul.mif"
        }
    )


if __name__ == "__main__":
    main()
