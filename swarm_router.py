import os
from swarms import Agent, SwarmRouter, SwarmType
from swarm_models import OpenAIChat
import concurrent.futures
import time
from cross_verifier import cross_verify
from dotenv import load_dotenv
from test import *


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
    REVENUE_AGENT_SYS_PROMPT = """
    Extract revenue-related information from a retrieved financial report context, providing detailed insights and relevant data points. The report should be summarized to highlight key revenue metrics, including but not limited to: total revenue, revenue growth rate, revenue by segment, and other relevant financial metrics. The extracted information should be accurately interpreted and presented in a clear and concise manner.
    """

    INCOME_TAX_AGENT_SYS_PROMPT = """
    Extract all relevant information regarding income tax from a context of financial reports. Please provide a detailed summary of the information extracted, including but not limited to: tax rates, tax liabilities, tax credits, and any other relevant details. Note: Please consider the context of the 10-K reports and identify the specific sections or parts that are most relevant to income tax information. Make sure to extract accurate and concise information.
    """

    LEGALITY_AGENT_SYS_PROMPT = """
    Extract relevant legal proceeding information from financial reports context.
    """

    ASSETS_AGENT_SYS_PROMPT = """
    Provide information about the company's assets, specifically those listed in the financial report, which I assume is a publicly available annual report filed with the Securities and Exchange Commission. Use the context from the 10-K report to provide details about the company's assets, including their nature, value, and any relevant information provided in the report. Please provide this information in a concise and organized manner.
    """

    SHARES_AGENT_SYS_PROMPT = """
    Provide information about the shares, stocks, and equity of a company from the financial report. Please use the information retrieved from the company's 10-K report for [Company Name] (e.g., [link to 10-K report] or a similar publicly available filing) to answer the following questions: [Insert specific questions about shares, stocks, and equity, e.g. 'What is the company's total share count?', 'What is the market capitalization of the company?', 'How many outstanding shares does the company have?
    """

    # Initialize your agents for different social media platforms
    agents = [
        Agent(
            agent_name="Revenue-Agent",
            system_prompt=REVENUE_AGENT_SYS_PROMPT,
            llm=model,
            max_loops=1,
            dynamic_temperature_enabled=True,
            saved_state_path="Revenue_agent.json",
            user_name="swarm_corp",
            retry_attempts=1,
        ),
        Agent(
            agent_name="Income-Tax-Agent",
            system_prompt=INCOME_TAX_AGENT_SYS_PROMPT,
            llm=model,
            max_loops=1,
            dynamic_temperature_enabled=True,
            saved_state_path="Income_tax_agent.json",
            user_name="swarm_corp",
            retry_attempts=1,
        ),
        Agent(
            agent_name="Legality-Agent",
            system_prompt=LEGALITY_AGENT_SYS_PROMPT,
            llm=model,
            max_loops=1,
            dynamic_temperature_enabled=True,
            saved_state_path="Legality_agent.json",
            user_name="swarm_corp",
            retry_attempts=1,
        ),
        Agent(
            agent_name="Assets-Agent",
            system_prompt=ASSETS_AGENT_SYS_PROMPT,
            llm=model,
            max_loops=1,
            dynamic_temperature_enabled=True,
            saved_state_path="Assets_agent.json",
            user_name="swarm_corp",
            retry_attempts=1,
        ),
        Agent(
            agent_name="Shares-Agent",
            system_prompt=SHARES_AGENT_SYS_PROMPT,
            llm=model,
            max_loops=1,
            dynamic_temperature_enabled=True,
            saved_state_path="Shares_agent.json",
            user_name="swarm_corp",
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
                        Agent name : revenue_agent
                        REVENUE_AGENT_SYS_PROMPT = """
                        Extract revenue-related information from a retrieved financial report context, providing detailed insights and relevant data points. The report should be summarized to highlight key revenue metrics, including but not limited to: total revenue, revenue growth rate, revenue by segment, and other relevant financial metrics. The extracted information should be accurately interpreted and presented in a clear and concise manner.
                        """
                        
                        Agent name : income_tax_agent
                        INCOME_TAX_AGENT_SYS_PROMPT = """
                        Extract all relevant information regarding income tax from a context of financial reports. Please provide a detailed summary of the information extracted, including but not limited to: tax rates, tax liabilities, tax credits, and any other relevant details. Note: Please consider the context of the 10-K reports and identify the specific sections or parts that are most relevant to income tax information. Make sure to extract accurate and concise information.
                        """

                        Agent name : legalility_agent
                        LEGALITY_AGENT_SYS_PROMPT = """
                        Extract relevant legal proceeding information from financial reports context.
                        """

                        Agent name : assets_agent
                        ASSETS_AGENT_SYS_PROMPT = """
                        Provide information about the company's assets, specifically those listed in the financial report, which I assume is a publicly available annual report filed with the Securities and Exchange Commission. Use the context from the 10-K report to provide details about the company's assets, including their nature, value, and any relevant information provided in the report. Please provide this information in a concise and organized manner.
                        """

                        Agent name : share_agent
                        SHARES_AGENT_SYS_PROMPT = """
                        Provide information about the shares, stocks, and equity of a company from the financial report. Please use the information retrieved from the company's 10-K report for [Company Name] (e.g., [link to 10-K report] or a similar publicly available filing) to answer the following questions: [Insert specific questions about shares, stocks, and equity, e.g. 'What is the company's total share count?', 'What is the market capitalization of the company?', 'How many outstanding shares does the company have?
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
            future_revenue = executor.submit(revenue_agent, query, context, agents)
        if "income_tax_agent" in relevant_agents:
            future_income_tax = executor.submit(income_tax_agent, query, context, agents)
        if "legalility_agent" in relevant_agents:
            future_legalility = executor.submit(legalility_agent, query, context, agents)
        if "assets_agent" in relevant_agents:
            future_assets = executor.submit(assets_agent, query, context, agents)
        if "share_agent" in relevant_agents:
            future_share = executor.submit(share_agent, query, context, agents)

        
        final_context = ""
        
        # Collect the results
        if "revenue_agent" in relevant_agents:
            revenue_context, revenue_result = future_revenue.result()
            if(revenue_result != "No"):
                final_context += revenue_context
        if "income_tax_agent" in relevant_agents:
            income_tax_context, income_tax_result = future_income_tax.result()
            if(income_tax_result != "No"):
                final_context += income_tax_context  
        if "legalility_agent" in relevant_agents:
            legality_context, legalility_result = future_legalility.result()
            if(legalility_result != "No"):
                final_context += legality_context
        if "assets_agent" in relevant_agents:
            assets_context, assets_result = future_assets.result()
            if(assets_result != "No"):
                final_context += assets_context
        if "share_agent" in relevant_agents:
            share_context, share_result = future_share.result()
            if(share_result != "No"):
                final_context += share_context

    summarised_context = meta_agent.run(final_context)

    final_output = final_agent.run(f"Query: {query}" + f"Context: {summarised_context}")    

    return final_output


if __name__ == "__main__":
    start = time.time()

    #print(multi_agent(query, context))
    agents, meta_agent, final_agent, router = get_agents()

    print(multi_agent(agents, meta_agent, final_agent, router, query2, context2))

    end = time.time()
    print(end - start)

    # # Retrieve and print logs
    # for log in router.get_logs():
    #     print(f"{log.timestamp} - {log.level}: {log.message}")
