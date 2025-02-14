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

from dotenv import load_dotenv

load_dotenv()

llm = OpenAIModel(model_name="gpt-4o-mini")


########################################################
# Triage Agent
########################################################
triage_agent = Agent(
    model=llm,
    name="Triage Agent",
    system_prompt=(
        "You are a customer service bot for ACME Inc. "
        "Introduce yourself. Always be very brief. "
        "Gather information to direct the customer to the right department. "
        "But make your questions subtle and natural."
    ),
)


@triage_agent.tool_plain
def escalate_to_human(summary: str) -> None:
    """Only call this if explicitly asked to."""
    print("Escalating to human agent...")
    print("\n=== Escalation Report ===")
    print(f"Summary: {summary}")
    print("=========================\n")
    return None


@triage_agent.tool_plain
def transfer_to_sales_agent() -> Agent:
    """User for anything sales or buying related."""
    return sales_agent


@triage_agent.tool_plain
def transfer_to_issues_and_repairs() -> Agent:
    """User for issues, repairs, or refunds."""
    return issues_and_repairs_agent


########################################################
# Sales Agent
########################################################
sales_agent = Agent(
    model=llm,
    name="Sales Agent",
    system_prompt=(
        "You are a sales agent for ACME Inc."
        "Always answer in a sentence or less."
        "Follow the following routine with the user:"
        "1. Ask them about any problems in their life related to catching roadrunners.\n"
        "2. Casually mention one of ACME's crazy made-up products can help.\n"
        " - Don't mention price.\n"
        "3. Once the user is bought in, drop a ridiculous price.\n"
        "4. Only after everything, and if the user says yes, "
        "tell them a crazy caveat and execute their order.\n"
        ""
    ),
)


@sales_agent.tool_plain
def transfer_back_to_triage() -> Agent:
    """Call this if the user brings up a topic outside of your purview,
    including escalating to human."""
    return triage_agent


@sales_agent.tool_plain
def execute_order(product: str, price: int) -> str:
    """Price should be in USD."""
    print("\n\n=== Order Summary ===")
    print(f"Product: {product}")
    print(f"Price: ${price}")
    print("=================\n")
    confirm = input("Confirm order? y/n: ").strip().lower()
    if confirm == "y":
        print("Order execution successful!")
        return "Success"
    else:
        print("Order cancelled!")
        return "User cancelled order."


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
        ""
    ),
)


@issues_and_repairs_agent.tool_plain
def look_up_item(search_query: str) -> str:
    """Use to find item ID.
    Search query can be a description or keywords."""
    item_id = "item_132612938"
    print("Found item:", item_id)
    return item_id


@issues_and_repairs_agent.tool_plain
def execute_refund(item_id: str, reason: str = "not provided") -> str:
    print("\n\n=== Refund Summary ===")
    print(f"Item ID: {item_id}")
    print(f"Reason: {reason}")
    print("=================\n")
    print("Refund execution successful!")
    return "success"


@issues_and_repairs_agent.tool_plain
def transfer_back_to_triage() -> Agent:
    """Call this if the user brings up a topic outside of your purview,
    including escalating to human."""
    return triage_agent


class Handoff(BaseModel):
    agent_name: str
    user_prompt: str

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

    while True:
        # === 1. get agent to run ===
        result = current_agent.run_sync(user_prompt=messages[-1].parts[0].content, message_history=messages)
        '''
        TODO: need some work here because if returning Agent class directly, then it cannot be serialized
        error: return self.serializer.to_json(
        pydantic_core._pydantic_core.PydanticSerializationError: Unable to serialize unknown type: <class 'openai.AsyncOpenAI'>
        '''
        if (isinstance(result.data, str)):
            print(result.data)
            print("...waiting for user input...")

        # === 2. check if we need to handoff ===
        if (isinstance(result.data, Agent)):
            print("last message:", messages[-1].content)
            print("Handoff to", result.data.name)
            current_agent = result.data
def main():
    starting_agent = triage_agent
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        messages = [ModelRequest(parts=[UserPromptPart(content=user_input)])]
        run_full_turn(starting_agent, messages)

# === Main ===
if __name__ == "__main__":
    main()
    
        






