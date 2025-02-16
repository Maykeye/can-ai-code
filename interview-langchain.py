#!/usr/bin/env python3
from langchain import LLMChain, PromptTemplate
import pandas as pd
import argparse
import os
import json
from time import sleep
from jinja2 import Template

def init_model(provider, **kwargs):
    if provider == 'cohere/command-nightly':
        from langchain import Cohere
        return Cohere(model='command-nightly',**kwargs)
    elif provider == 'ai21/j2-jumbo-instruct':
        from langchain.llms import AI21

        if 'max_tokens' in kwargs:
            kwargs['maxTokens'] = kwargs['max_tokens']
            del kwargs['max_tokens']
        
        return AI21(model='j2-jumbo-instruct', **kwargs)
    elif provider == 'openai/chatgpt':
        from langchain.chat_models import ChatOpenAI
        return ChatOpenAI(model='gpt-3.5-turbo', **kwargs)
    elif provider == 'openai/gpt4':
        from langchain.chat_models import ChatOpenAI
        return ChatOpenAI(model='gpt-4', **kwargs)
    raise Exception('Unsupported provider')


def prompt_template(model):
    FILENAMES = {
        'openai/chatgpt': 'prompts/openai_chatgpt.txt',
        'cohere/command-nightly': 'prompts/cohere_command-nightly.txt',
        'ai21/j2-jumbo-instruct': 'prompts/ai21_j2-jumbo-instruct.txt',
    }
    filename = FILENAMES.get(model)
    if filename:
        with open(filename) as f:
            return f.read()
    print('WARNING: Failed to load template for provider '+model)
    return '{{prompt}}' 

parser = argparse.ArgumentParser(description='Interview executor for LangChain')
parser.add_argument('--questions', type=str, required=True, help='path to questions .csv from prepare stage')
parser.add_argument('--model', type=str, required=True, help='model to use')
parser.add_argument('--outdir', type=str, required=True, help='output directory')
parser.add_argument('--delay', type=int, default=0, help='delay between questions (in seconds)')
parser.add_argument('--temperature', type=float, default=0.7, help='temperature for generation')
parser.add_argument('--max_tokens', type=int, default=512, help='max length of generated text')
args = parser.parse_args()

model = init_model(args.model, temperature=args.temperature, max_tokens=args.max_tokens)
if not os.path.exists(args.outdir):
    os.mkdir(args.outdir) 

info = {
    'interview': 'langchain',
    'model': args.model,
    'params': {
        'temperature': args.temperature,
        'max_tokens': args.max_tokens
    },
    'prompt_template': prompt_template(args.model),
    'stop_token': None
}
with open(args.outdir+'/info.json', 'w') as f:
    f.write(json.dumps(info))

df = pd.read_csv(args.questions)
for idx, test in df.iterrows():
    print(test['name'])
    out_file = args.outdir+'/'+test['name']+'.txt'

    if os.path.exists(out_file):
        print('Skipping, already exists')
        continue

    full_prompt = Template(info['prompt_template']).render(prompt=test['prompt'])

    lc_prompt = PromptTemplate(template='{input}', input_variables=['input'])
    chain = LLMChain(llm=model, prompt=lc_prompt)

    answer = chain.run(input=full_prompt)
    print(answer)
    with open(out_file, 'w') as f:
        f.write(answer)

    if args.delay:
        sleep(args.delay)