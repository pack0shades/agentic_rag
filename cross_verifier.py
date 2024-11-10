import os
from swarms import Agent, ConcurrentWorkflow, SwarmRouter, SwarmType
from swarm_models import OpenAIChat
import concurrent.futures
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("WORKSPACE_DIR"))

# Initialize agents
model = OpenAIChat(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    temperature=0.1,
)

VERIFICATION_PROMPT = "Verify the relevance of the following input context to the given query."

CONSENSUS_PROMPT = "Take the input from verifier agents and give a consensus whether the context matches the query. Strictly answer in 'Yes' or 'No'."

# Function to create a verifier agent


def create_verifier(agent_name, system_prompt, llm, saved_state_path, user_name, retry_attempts=1):
    return Agent(
        agent_name=agent_name,
        system_prompt=system_prompt,
        llm=llm,
        max_loops=2,  # increase to 2 if SSLError
        dynamic_temperature_enabled=True,
        saved_state_path=saved_state_path,
        user_name=user_name,
        retry_attempts=retry_attempts,
    )


consensus_agent = Agent(
    agent_name="Consensus_agent",
    system_prompt=CONSENSUS_PROMPT,
    llm=model,
    max_loops=1,
    dynamic_temperature_enabled=True,
    saved_state_path="Consensus_agent.json",
    user_name="swarm_corp",
    retry_attempts=1,  # increase to 2 if SSLError
)

# Create verifier agents
verifier_agent_1 = create_verifier(
    "Verifier-Agent-1", VERIFICATION_PROMPT, model, "verifier_agent_1.json", "swarm_corp")
verifier_agent_2 = create_verifier(
    "Verifier-Agent-2", VERIFICATION_PROMPT, model, "verifier_agent_2.json", "swarm_corp")
verifier_agent_3 = create_verifier(
    "Verifier-Agent-3", VERIFICATION_PROMPT, model, "verifier_agent_3.json", "swarm_corp")


def verifier_1(prompt):
    return verifier_agent_1.run(prompt)


def verifier_2(prompt):
    return verifier_agent_1.run(prompt)


def verifier_3(prompt):
    return verifier_agent_3.run(prompt)

# Function to perform cross-verification


def cross_verify(query, output):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_result_1 = executor.submit(
            verifier_1, f"Query: {query}" + f"Output: {output}")
        future_result_2 = executor.submit(
            verifier_2, f"Query: {query}" + f"Output: {output}")
        future_result_3 = executor.submit(
            verifier_3, f"Query: {query}" + f"Output: {output}")

        result_1 = future_result_1.result()
        result_2 = future_result_2.result()
        result_3 = future_result_3.result()
        print("Cross-verifier running:::::::::::::::::::::::::::::::")
    # # Each verifier agent checks the relevance of the output
    # result_1 = verifier_agent_1.run(f"Query: {query}" + f"Output: {output}")
    # result_2 = verifier_agent_2.run(f"Query: {query}" + f"Output: {output}")
    # result_3 = verifier_agent_3.run(f"Query: {query}" + f"Output: {output}")

    # Aggregate the results using a consensus mechanism
    consensus_result = consensus_agent.run(
        f"Agent_1: {result_1}" + f"Agent_2: {result_2}" + f"Agent_3: {result_3}")
    print(consensus_result + "::::::::::::::::::::::::::::::::::::::")
    return consensus_result


if __name__ == "__main__":
    # Example usage
    query = "What is the impact of income tax on small businesses?"
    output = "Income tax can significantly affect the cash flow of small businesses, potentially leading to financial constraints."

    verified_output = cross_verify(query, output)
    print(verified_output)
