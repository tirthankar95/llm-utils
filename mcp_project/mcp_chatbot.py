from xml.parsers.expat import model
from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from anthropic import Anthropic

nest_asyncio.apply()
load_dotenv()


class LLM_Client(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_message(self, args: dict) -> str:
        """All parameters are passed as dictionary including the model name"""
        pass


class MyOpenAI(LLM_Client):
    def __init__(self):
        super().__init__()
        self.client = OpenAI()

    def get_message(self, **kwargs) -> str:
        return self.client.chat.completions.create(**kwargs)


class MyAnthropic(LLM_Client):
    def __init__(self):
        super().__init__()
        self.client = Anthropic()

    def get_message(self, **kwargs) -> str:
        return self.client.messages.create(**kwargs)


class Local(LLM_Client):
    def __init__(self):
        super().__init__()
        self.client = ChatOpenAI(
            base_url="http://localhost:8080/v1",
        )

    def get_message(self, **kwargs) -> str:
        self.client.invoke(kwargs["messages"], tools=kwargs["tools"])


class MCP_ChatBot:
    def __init__(self, llm_client: LLM_Client, model_name: str = "gpt-4o-mini"):
        # Initialize session and client objects
        self.session: ClientSession = None
        self.client = llm_client
        self.model_name = model_name
        self.available_tools: List[dict] = []

    async def process_query(self, query):
        messages = [{"role": "user", "content": query}]
        response = self.client.get_message(
            max_tokens=1024,
            model=self.model_name,
            tools=self.available_tools,
            messages=messages,
        )
        process_query = True
        while process_query:
            assistant_content = []
            for content in response.content:
                if content.type == "text":
                    print(content.text)
                    assistant_content.append(content)
                    if len(response.content) == 1:
                        process_query = False
                elif content.type == "tool_use":
                    assistant_content.append(content)
                    messages.append({"role": "assistant", "content": assistant_content})
                    tool_id = content.id
                    tool_args = content.input
                    tool_name = content.name
                    print(f"Calling tool {tool_name} with args {tool_args}")
                    # Call a tool
                    # result = execute_tool(tool_name, tool_args): not anymore needed
                    # tool invocation through the client session
                    result = await self.session.call_tool(
                        tool_name, arguments=tool_args
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": result.content,
                                }
                            ],
                        }
                    )
                    response = self.client.get_message(
                        max_tokens=1024,
                        model=self.model_name,
                        tools=self.available_tools,
                        messages=messages,
                    )
                    if (
                        len(response.content) == 1
                        and response.content[0].type == "text"
                    ):
                        print(response.content[0].text)
                        process_query = False

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                await self.process_query(query)
                print("\n")
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="uv",  # Executable
            args=["run", "research_server.py"],  # Optional command line arguments
            env=None,  # Optional environment variables
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                # Initialize the connection
                await session.initialize()

                # List available tools
                response = await session.list_tools()

                tools = response.tools
                print(
                    "\nConnected to server with tools:", [tool.name for tool in tools]
                )

                self.available_tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in response.tools
                ]
                await self.chat_loop()


async def main():
    # chatbot = MCP_ChatBot(Local())
    chatbot = MCP_ChatBot(MyAnthropic(), model_name="claude-haiku-4-5-20251001")
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    asyncio.run(main())
