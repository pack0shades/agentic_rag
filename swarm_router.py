import os
from swarms import Agent, SwarmRouter, SwarmType
from swarm_models import OpenAIChat
import concurrent.futures
import time
from cross_verifier import cross_verify
from dotenv import load_dotenv
from test import query, context, query2, context2

load_dotenv()
print(os.getenv("WORKSPACE_DIR"))


def get_agents():
    # Initialize agents
    model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o-mini",
        temperature=0.1,
    )

    # Define custom system prompts for each social media platform
    EXTRACTOR_AGENT_SYS_PROMPT = """
    You are an Extractor Agent. Your task is to meticulously analyze the provided contract text and extract specific information based on the given instructions.  Do not interpret or summarize, only extract.
    """

    CLASSIFIER_AGENT_SYS_PROMPT = """
    You are a Classifier Agent. Your task is to analyze the provided contract text and classify it based on its type and identify key legal concepts present.  Provide a concise classification and a list of relevant legal concepts with brief explanations.
    """

    SUMMARIZER_AGENT_SYS_PROMPT = """
    You are a Summarizer Agent. Your task is to create a concise summary of the provided contract, focusing on the key obligations and rights of each party involved. Avoid legal jargon and use plain language.
    """

    QA_AGENT_SYS_PROMPT = """
    You are a QA Agent. You will receive a query and structured information extracted from a contract by other agents. Use this information to answer the query accurately and concisely.  If the information provided is insufficient to answer the query, state "Insufficient Information."
    """

    RISK_ASSES_AGENT_SYS_PROMPT = """
    You are a Risk Assessment Agent. Analyze the provided contract text and identify potential legal risks and liabilities for each party involved. Prioritize clarity and conciseness in your output.
    """

    # Initialize your agents for different social media platforms
    agents = [
        Agent(
            agent_name="Extractor-Agent",
            system_prompt=EXTRACTOR_AGENT_SYS_PROMPT,
            llm=model,
            max_loops=1,
            dynamic_temperature_enabled=True,
            saved_state_path="Extractor_agent.json",
            retry_attempts=1,
        ),
        Agent(
            agent_name="Classifier-Agent",
            system_prompt=CLASSIFIER_AGENT_SYS_PROMPT,
            llm=model,
            max_loops=1,
            dynamic_temperature_enabled=True,
            saved_state_path="Classifier_agent.json",
            retry_attempts=1,
        ),
        Agent(
            agent_name="Summarizer-Agent",
            system_prompt=SUMMARIZER_AGENT_SYS_PROMPT,
            llm=model,
            max_loops=1,
            dynamic_temperature_enabled=True,
            saved_state_path="Summarizer_agent.json",
            retry_attempts=1,
        ),
        Agent(
            agent_name="QA-Agent",
            system_prompt=QA_AGENT_SYS_PROMPT,
            llm=model,
            max_loops=1,
            dynamic_temperature_enabled=True,
            saved_state_path="Assets_agent.json",
            retry_attempts=1,
        ),
        Agent(
            agent_name="Risk_Asses-Agent",
            system_prompt=RISK_ASSES_AGENT_SYS_PROMPT,
            llm=model,
            max_loops=1,
            dynamic_temperature_enabled=True,
            saved_state_path="Shares_agent.json",
            retry_attempts=1,
        ),
    ]

    SUMMARIZER_PROMPT = "Generate a concise and accurate summary of the input text. Handle input from multiple LLMs and produce a summarized output that captures the essential information from all the input texts."

    meta_agent = Agent(
        agent_name="Document-Summarizer",
        system_prompt=SUMMARIZER_PROMPT,
        llm=model,
        max_loops=1,
        autosave=True,
        verbose=True,
        dynamic_temperature_enabled=True,
        saved_state_path="summarizer_agent.json",
        retry_attempts=1,
        context_length=200000,
        output_type="string",
    )

    FINAL_PROMPT = "Answer the query using only the given context."

    final_agent = Agent(
        agent_name="Question-answerer",
        system_prompt=FINAL_PROMPT,
        llm=model,
        max_loops=1,
        autosave=True,
        verbose=True,
        dynamic_temperature_enabled=True,
        saved_state_path="final-agent.json",
        retry_attempts=1,
        context_length=200000,
        output_type="string",
    )

    ROUTER_PROMPT = '''Provide a list of recommendations for the most suitable agents from the given system prompts of 5 agents, based on their characteristics and capabilities, to best match and fulfill the query.
                        Agent name : extractor_agent
                        EXTRACTOR_AGENT_SYS_PROMPT = """
                        You are an Extractor Agent. Your task is to meticulously analyze the provided contract text and extract specific information based on the given instructions.  Do not interpret or summarize, only extract.
                        """
                        
                        Agent name : classifier_agent
                        CLASSIFIER_AGENT_SYS_PROMPT = """
                        You are a Classifier Agent. Your task is to analyze the provided contract text and classify it based on its type and identify key legal concepts present.  Provide a concise classification and a list of relevant legal concepts with brief explanations.
                        """

                        Agent name : summarizer_agent
                        SUMMARIZER_AGENT_SYS_PROMPT = """
                        You are a Summarizer Agent. Your task is to create a concise summary of the provided contract, focusing on the key obligations and rights of each party involved. Avoid legal jargon and use plain language.
                        """

                        Agent name : qa_agent
                        QA_AGENT_SYS_PROMPT = """
                        You are a QA Agent. You will receive a query and structured information extracted from a contract by other agents. Use this information to answer the query accurately and concisely.  If the information provided is insufficient to answer the query, state "Insufficient Information."
                        """

                        Agent name : risk_asses_agent
                        RISK_ASSES_AGENT_SYS_PROMPT = """
                        You are a Risk Assessment Agent. Analyze the provided contract text and identify potential legal risks and liabilities for each party involved. Prioritize clarity and conciseness in your output.
                        """
                        Output the comma separated names of the relevant agents 
                        '''

    router = Agent(
        agent_name="Router-Agent",
        system_prompt=ROUTER_PROMPT,
        llm=model,
        max_loops=1,
        autosave=True,
        verbose=True,
        dynamic_temperature_enabled=True,
        saved_state_path="router-agent.json",
        retry_attempts=1,
        context_length=200000,
        output_type="string",
    )
    return agents, meta_agent, final_agent, router


def revenue_agent(query, context, agents):
    result = agents[0].run(context)
    result_bool = cross_verify(query, result)
    return result, result_bool


def income_tax_agent(query, context, agents):
    result = agents[1].run(context)
    result_bool = cross_verify(query, result)
    return result, result_bool


def legalility_agent(query, context, agents):
    result = agents[2].run(context)
    result_bool = cross_verify(query, result)
    return result, result_bool


def assets_agent(query, context, agents):
    result = agents[3].run(context)
    result_bool = cross_verify(query, result)
    return result, result_bool


def share_agent(query, context, agents):
    result = agents[4].run(context)
    result_bool = cross_verify(query, result)
    return result, result_bool


def multi_agent(agents, meta_agent, final_agent, router, query, context):

    relevant_agents = router.run(query).split(", ")
    print(relevant_agents)

    with concurrent.futures.ThreadPoolExecutor() as executor:

        if "revenue_agent" in relevant_agents:
            future_revenue = executor.submit(
                revenue_agent, query, context, agents)
        if "income_tax_agent" in relevant_agents:
            future_income_tax = executor.submit(
                income_tax_agent, query, context, agents)
        if "legalility_agent" in relevant_agents:
            future_legalility = executor.submit(
                legalility_agent, query, context, agents)
        if "assets_agent" in relevant_agents:
            future_assets = executor.submit(
                assets_agent, query, context, agents)
        if "share_agent" in relevant_agents:
            future_share = executor.submit(share_agent, query, context, agents)

        final_context = ""

        # Collect the results
        if "revenue_agent" in relevant_agents:
            revenue_context, revenue_result = future_revenue.result()
            if (revenue_result != "No"):
                final_context += revenue_context
        if "income_tax_agent" in relevant_agents:
            income_tax_context, income_tax_result = future_income_tax.result()
            if (income_tax_result != "No"):
                final_context += income_tax_context
        if "legalility_agent" in relevant_agents:
            legality_context, legalility_result = future_legalility.result()
            if (legalility_result != "No"):
                final_context += legality_context
        if "assets_agent" in relevant_agents:
            assets_context, assets_result = future_assets.result()
            if (assets_result != "No"):
                final_context += assets_context
        if "share_agent" in relevant_agents:
            share_context, share_result = future_share.result()
            if (share_result != "No"):
                final_context += share_context

    summarised_context = meta_agent.run(final_context)

    final_output = final_agent.run(
        f"Query: {query}" + f"Context: {summarised_context}")

    return final_output


if __name__ == "__main__":
    start = time.time()

    agents, meta_agent, final_agent, router = get_agents()

    print(multi_agent(agents, meta_agent, final_agent, router, query2, context2))

    end = time.time()
    print(end - start)

    # # Retrieve and print logs
    # for log in router.get_logs():
    #     print(f"{log.timestamp} - {log.level}: {log.message}")
