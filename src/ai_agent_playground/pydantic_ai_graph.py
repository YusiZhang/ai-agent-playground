"""
Create a simple entry point for the Pydantic AI so to debug into its graph-based agent run code 
to help understand how it works.
"""

# Load API key
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
load_dotenv()

# configure logfire
import logfire
from enum import Enum
logfire.configure(send_to_logfire='if-token-present')

client_4o = AsyncAzureOpenAI(
    azure_deployment='gpt-4o-mini',
    api_version='2024-08-01-preview'
)
llm = OpenAIModel('gpt-4o-mini', openai_client=client_4o)

class LLMResponse(BaseModel):
    text: str = Field(title='LLM response text', description='The text of the LLM response', default=None)
    tool_call: str = Field(title='Tool call name', description='The name of the tool call made by the LLM', default=None)

def main_without_tool_call():
    agent = Agent(model=llm, system_prompt='Please echo the input with a prefix of "Echo: "', result_type=LLMResponse)
    result = agent.run_sync('Hello world!')
    print(result.data)

def main_with_tool_call_and_tool():
    agent = Agent(
        model=llm,
        system_prompt='Please use the tool to echo the input. Make sure the response has all the necessary information.',
        result_type=LLMResponse
    )
    @agent.tool
    async def echo(ctx, text: str) -> LLMResponse:
        return LLMResponse(text=f'Tool Echo: {text}', tool_call='echo')
    result = agent.run_sync('Hello world!')
    print(result.data)


if __name__ == '__main__':
    main_without_tool_call()
    main_with_tool_call_and_tool()