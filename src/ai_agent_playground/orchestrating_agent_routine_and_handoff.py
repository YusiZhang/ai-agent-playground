"""
https://cookbook.openai.com/examples/orchestrating_agents

This example shows how to orchestrate agents in a routine and handoff information between them.
"""

from typing import Union
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelRequestPart, UserPromptPart
from pydantic_ai.usage import Usage, UsageLimits
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncAzureOpenAI

# Load API key
from dotenv import load_dotenv
load_dotenv()

# configure logfire
import logfire
from enum import Enum
logfire.configure(send_to_logfire='if-token-present')

# llm = OpenAIModel(model_name="gpt-4o-2024-11-20")

# client_4o = AsyncAzureOpenAI(
#     azure_deployment='gpt-4o',
#     api_version='2024-08-01-preview'
# )
# llm = OpenAIModel('gpt-4o', openai_client=client_4o)

client_4o = AsyncAzureOpenAI(
    azure_deployment='gpt-4o-mini',
    api_version='2024-08-01-preview'
)
llm = OpenAIModel('gpt-4o-mini', openai_client=client_4o)

class AgentName(str, Enum):
    TRIAGE = "Triage Agent"
    SALES = "Sales Agent"
    ISSUES_AND_REPAIRS = "Issues and Repairs Agent"

class AgentResponse(BaseModel):
    plan_text_response: str = Field(title="Plan Text Response", description="The plain text response from the agent.", default=None)
    current_agent_name: AgentName = Field(title="Current Agent Name", description="The name of the current agent.", default=None)
    handoff_agent_name: AgentName = Field(title="Handoff Agent Name", description="The name of the agent to handoff to.", default=None)
    user_prompt: str = Field(title="User Prompt", description="The user prompt to handoff to the next agent, it would either be user input or fulfilled by previous agent", default=None)

########################################################
# Triage Agent
########################################################
triage_agent = Agent(
    model=llm,
    name="Triage Agent",
    system_prompt=(
        "You are a customer service bot for ACME Inc. "
        "Introduce yourself. Always be very brief. "
        "Gather information and use tool call to transfer the customer to the right department. "
        "But make your questions subtle and natural."
        "Call `final_result` when you are done or ready for more user input."
    ),
    result_type=AgentResponse,
    # should do retry?
)


@triage_agent.tool_plain
def escalate_to_human(summary: str) -> AgentResponse:
    """Only call this if explicitly asked to."""
    print("Escalating to human agent...")
    print("\n=== Escalation Report ===")
    print(f"Summary: {summary}")
    print("=========================\n")
    return AgentResponse(plan_text_response="Escalating to human agent...")


@triage_agent.tool_plain
def transfer_to_sales_agent() -> AgentResponse:
    """transfer user to sales department for anything sales or buying related."""
    return AgentResponse(handoff_agent_name="Sales Agent")


@triage_agent.tool_plain
def transfer_to_issues_and_repairs() -> AgentResponse:
    """transfer user issue and repair department for issues, repairs, or refunds."""
    return AgentResponse(handoff_agent_name="Issues and Repairs Agent")

########################################################
# Sales Agent
########################################################
sales_agent = Agent(
    model=llm,
    name="Sales Agent",
    system_prompt=(
        "You are a sales agent for ACME Inc that sells painkill and beauty product. Your goal is to engage the user and sell them a product."
        "Always answer in a sentence or less."
        "Follow the following routine with the user:"
        "1. Ask them about any problems in their life related to catching roadrunners.\n"
        "2. Casually mention one of ACME's crazy made-up products can help.\n"
        " - Don't mention price.\n"
        "3. Once the user is bought in, drop a ridiculous price.\n"
        "4. Only after everything, and if the user says yes, "
        "tell them a crazy caveat and execute their order.\n"
        
        "Call `final_result` when you are done or ready for more user input."
        ""
    ),
    result_type=AgentResponse,
)


@sales_agent.tool_plain
def transfer_back_to_triage() -> AgentResponse:
    """Call this if the user brings up a topic outside of your purview,
    including escalating to human."""
    return AgentResponse(handoff_agent_name="Triage Agent")

@sales_agent.tool_plain
def execute_order(product: str, price: int) -> AgentResponse:
    """Price should be in USD."""
    print("\n\n=== Order Summary ===")
    print(f"Product: {product}")
    print(f"Price: ${price}")
    print("=================\n")
    confirm = input("Confirm order? y/n: ").strip().lower()
    if confirm == "y":
        print("Order execution successful!")
        return AgentResponse(plan_text_response="Order execution successful!")
    else:
        print("Order cancelled!")
        return AgentResponse(plan_text_response="Order cancelled!")


########################################################
# Issues and Repairs Agent
########################################################
issues_and_repairs_agent = Agent(
    model=llm,
    name="Issues and Repairs Agent",
    system_prompt=(
        "You are a customer support agent for ACME Inc."
        "Always answer in a sentence or less."
        "Follow the following routine with the user:"
        "1. First, ask probing questions and understand the user's problem deeper.\n"
        " - unless the user has already provided a reason.\n"
        "2. Propose a fix (make one up).\n"
        "3. ONLY if not satesfied, offer a refund.\n"
        "4. If accepted, search for the ID and then execute refund."

        "Call `final_result` when you are done or ready for more user input."
        ""
    ),
    result_type=AgentResponse,
)


@issues_and_repairs_agent.tool_plain
def look_up_item(search_query: str) -> AgentResponse:
    # """Use to find item ID.
    # Search query can be a description or keywords."""
    """
    Use to find item ID.
    
    Args:
        search_query (str): Search query can be a description or keywords.
    Returns:
        AgentResponse: Response containing the item ID.
    """
    item_id = "item_132612938"
    print("Found item:", item_id)
    return AgentResponse(plan_text_response=item_id)


@issues_and_repairs_agent.tool_plain
def execute_refund(item_id: str, reason: str = "not provided") -> AgentResponse:
    """
    Use to execute refund.
    
    Args:
        item_id (str): The ID of the item to be refunded.
        reason (str): The reason for the refund. Defaults to "not provided".
    Returns:
        AgentResponse: Response indicating the success of the refund execution.
    """
    print("\n\n=== Refund Summary ===")
    print(f"Item ID: {item_id}")
    print(f"Reason: {reason}")
    print("=================\n")
    print("Refund execution successful!")
    return AgentResponse(plan_text_response="Refund execution successful!")


@issues_and_repairs_agent.tool_plain
def transfer_back_to_triage() -> AgentResponse:
    """Call this if the user brings up a topic outside of your purview,
    including escalating to human."""
    return AgentResponse(handoff_agent_name="Triage Agent")



agent_dict = {
        "Triage Agent": triage_agent,
        "Sales Agent": sales_agent,
        "Issues and Repairs Agent": issues_and_repairs_agent,
        
}


def agent_lookup(agent_name: str) -> Agent:    
    return agent_dict[agent_name]

def run_full_turn(agent: Agent, messages: list[ModelMessage]) -> list[ModelMessage]:
    current_agent = agent
    # num_initial_messages = len(messages)
    messages = messages.copy()
    handoff_message = None
    handoff_message_history = None

    while True:
        # === 1. get agent to run ===
        # TODO: should just use the user_prompt from Handoff?
        if handoff_message:
            user_prompt = handoff_message
            handoff_message_history = handoff_message_history
        else:
            user_prompt = messages[-1].parts[0].content
            handoff_message_history = messages
        result = current_agent.run_sync(user_prompt=user_prompt, message_history=handoff_message_history)

        if isinstance(result.data, str):
            print("Agent:", result.data)
            print("waitng for user input...")
            break

        # If result data has plain text response and no agent handoff then we need to break and waiting for user input
        if result.data and result.data.plan_text_response and not result.data.handoff_agent_name:
            print("Agent (plan text response):", result.data.plan_text_response)
            print("last user prompt:", user_prompt)
            print("waitng for user input...")
            break

        # hand off to another agent and run that agent
        if (isinstance(result.data, AgentResponse) and result.data.handoff_agent_name):
            handoff_message = result.data.user_prompt
            handoff_message_history = result.new_messages()
            print("Handoff to", result.data.handoff_agent_name)
            print("last user prompt:", user_prompt)
            current_agent = agent_lookup(result.data.handoff_agent_name)

        
        
def main():
    starting_agent = triage_agent
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        messages = [ModelRequest(parts=[UserPromptPart(content=user_input)])]
        run_full_turn(starting_agent, messages)

def main_sales_only():
    starting_agent = sales_agent
    result = None
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        result = sales_agent.run_sync(user_input, message_history=[] if not result else result.new_messages())
        print("Agent:", result.data)

# === Main ===
if __name__ == "__main__":
    # main()
    main_sales_only()
    
        






