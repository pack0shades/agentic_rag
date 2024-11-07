import openai
from openai import OpenAI
import concurrent.futures
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
def revenue_agent(prompt: str)-> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract revenue-related information from a retrieved 10-K report context, providing detailed insights and relevant data points. The report should be summarized to highlight key revenue metrics, including but not limited to: total revenue, revenue growth rate, revenue by segment, and other relevant financial metrics. The extracted information should be accurately interpreted and presented in a clear and concise manner."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

def income_tax_agent(prompt: str)-> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract all relevant information regarding income tax from a context of 10-K reports. Please provide a detailed summary of the information extracted, including but not limited to: tax rates, tax liabilities, tax credits, and any other relevant details. Note: Please consider the context of the 10-K reports and identify the specific sections or parts that are most relevant to income tax information. Make sure to extract accurate and concise information."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content  

def legalility_agent(prompt: str)-> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract relevant legal proceeding information from 10-K reports context."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content 

def assets_agent(prompt: str)-> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Provide information about the company's assets, specifically those listed in the 10-K report, which I assume is a publicly available annual report filed with the Securities and Exchange Commission. Use the context from the 10-K report to provide details about the company's assets, including their nature, value, and any relevant information provided in the report. Please provide this information in a concise and organized manner."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content


def share_agent(prompt: str)-> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Provide information about the shares, stocks, and equity of a company from the 10-K report. Please use the information retrieved from the company's 10-K report for [Company Name] (e.g., [link to 10-K report] or a similar publicly available filing) to answer the following questions: [Insert specific questions about shares, stocks, and equity, e.g. 'What is the company's total share count?', 'What is the market capitalization of the company?', 'How many outstanding shares does the company have?"},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content 
   
def final_agent(prompt: str)-> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Design a text summarization model that can take in input from other large language models (LLMs) and generate a concise and accurate summary of the input text. The model should be able to handle input from multiple LLMs and produce a summarized output that captures the essential information from all the input texts."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content  

# Define a function to execute all agents in parallel and combine the results
def generate_final_prompt(prompt: str)-> str:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_revenue = executor.submit(revenue_agent, prompt)
        future_income_tax = executor.submit(income_tax_agent, prompt)
        future_legalility = executor.submit(legalility_agent, prompt)
        future_assets = executor.submit(assets_agent, prompt)
        future_share = executor.submit(share_agent, prompt)
        
        # Collect the results
        revenue_result = future_revenue.result()
        income_tax_result = future_income_tax.result()
        legalility_result = future_legalility.result()
        assets_result = future_assets.result()
        share_result = future_share.result()
        
        # Combine the results to form the final prompt
        final_prompt = prompt + revenue_result + income_tax_result + legalility_result + assets_result + share_result
    
    return final_prompt   

def context_to_agent(prompt):
    final_prompt = generate_final_prompt(prompt)
    output = final_agent(final_prompt)
    return output     


if __name__ == "__main__":
    prompt = "what was the ROI on tesla shares from 2016 to 2020"
    output = context_to_agent(prompt)
    print ("Final_output:")
    print(output)


