import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse
import pandas as pd
from pathlib import Path
from jinja2 import Template

def parse_args():
    parser = argparse.ArgumentParser(description='Interview for transformers models')
    parser.add_argument('--questions', type=str, required=True, help='path to questions .csv from prepare stage')
    parser.add_argument('--outdir', type=str, required=True, help='output directory')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--prompt', type=str, required=True, help="prompt template")
    parser.add_argument('--model', type=str, required=True, help="either model path(e.g. '../models/my-model') or model id to download from HF(e.g. 'Salesforce/codet5p-2b')")
    parser.add_argument('--device', type=str,  default='cuda', help="cuda/cpu")
    parser.add_argument('--trust-remote-code', action='store_true', help="allow to run remote code inside the model .py files.")
    parser.add_argument('--dtype', choices={'f16', 'f32', 'bf16'},  default="bf16", help="data type")
    return parser.parse_args()

def run_interview(args, generate):
    Path(args.outdir).mkdir(exist_ok=True, parents=True)

    with open(args.prompt) as f:
        prompt_template = Template(f.read())

    df = pd.read_csv(args.questions)


    comment = {
        'python': '#',
        'javascript': '//'
    }
    function_prefix = {
        'python': 'def',
        'javascript': 'function'
    }

    for idx, test in df.iterrows():
        print(test['name'])
        out_file = args.outdir+'/'+test['name']+'.txt'

        if Path(out_file).exists():
            print('Skipping, already exists')
            continue

        full_prompt = prompt_template.render(
            prompt=test['prompt'],
            comment=comment[test['language']],
            function_prefix=function_prefix[test['language']]
        )

        if args.debug:
            print(f"===vvv===\n{full_prompt}===^^^===")

        answer = generate(full_prompt)
        print(answer)

        with open(out_file, 'w') as f:
            f.write(answer)

class CodeT5P:
    def __init__(self, args):
        types = {
            'bf16': torch.bfloat16,
            'f16': torch.float16,
            'f32': torch.float32,            
        }
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model, 
                                                    torch_dtype=types.get(args.dtype),
                                                    trust_remote_code=args.trust_remote_code).to(args.device)
        self.device = args.device


    def generate(self, txt, max_new_tokens=512):
        tokenized = self.tokenizer(txt, return_tensors="pt").to(self.device)
        tokenized['decoder_input_ids'] = tokenized['input_ids'].clone()
        outputs = self.model.generate(**tokenized, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def run():
    args = parse_args()
    codet5p = CodeT5P(args)
    run_interview(args, codet5p.generate)

if __name__ == "__main__":
    run()
