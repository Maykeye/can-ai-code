import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse


class HuggingFaceModelInterview:
    def __init__(self, checkpoint: str, dtype: str, device:str, trust_remote_code=False) -> None:        
        self.device = device        
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, 
                                                    **self.parse_dtype(dtype),
                                                    trust_remote_code=trust_remote_code).to(device)    

    def parse_dtype(self, dtype):
        if dtype == '8bit':
            return {'load_in_8bit': True}
        
        types = {
            'bf16': torch.bfloat16,
            'f16': torch.float16,
            'f32': torch.float32,            
        }
        return types[dtype]

    def tokenize(self, txt):
        return self.tokenizer(txt, return_tensors="pt").to(self.device)
    
    def detokenize(self, tokenized):
        return self.tokenizer.decode(tokenized[0], skip_special_tokens=True)
    
    def generate(self, txt, max_new_tokens=512):
        encoding = self.tokenize(txt)
        outputs = self.model.generate(**encoding, max_new_tokens=max_new_tokens)
        return self.detokenize(outputs)

    @classmethod
    def parse_args(clazz):
        parser = argparse.ArgumentParser(description='Interview for transformers models')
        parser.add_argument('--questions', type=str, required=True, help='path to questions .csv from prepare stage')
        parser.add_argument('--outdir', type=str, required=True, help='output directory')
        parser.add_argument('--prompt', type=str, required=True, help="prompt template")
        parser.add_argument('--model', type=str, required=True, help="either model path(e.g. '../models/my-model') or model id to download from HF(e.g. 'Salesforce/codet5p-2b')")
        parser.add_argument('--device', type=str,  default='cuda', help="cuda/cpu")
        parser.add_argument('--trust-remote-code', action='store_true', help="allow to run remote code inside the model .py files.")
        parser.add_argument('--dtype', choices={'f16', 'f32', 'bf16', '8bit'},  default="bf16", help="data type")
        return parser.parse_args()




class CodeT5P(HuggingFaceModel):
    def __init__(self, args: argparse.Namespace) -> None:
        assert args.trust_remote_code
        super().__init__(checkpoint=args.model, trust_remote_code=True)

    def tokenize(self, txt):
        tokenized = super().tokenize(txt)
        tokenized['decoder_input_ids'] = tokenized['input_ids'].clone()
        return tokenized

def run():
    args = parse_args()


if __name__ == "__main__":
    run()



