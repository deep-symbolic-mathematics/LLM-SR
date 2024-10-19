import gc
from argparse import ArgumentParser
import torch
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from flask import Flask, request, jsonify

from flask_cors import CORS


# arguments
parser = ArgumentParser()
parser.add_argument('--gpu_ids', nargs='+', default=['0','1','2','3'])
parser.add_argument('--quantization', default=False, action='store_true')
parser.add_argument('--model_path', type=str, default='mistralai/Mixtral-8x7B-Instruct-v0.1')
parser.add_argument('--host', type=str, default=None)
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--do_sample', type=bool, default=True)
parser.add_argument('--max_new_tokens', type=int, default=512)
parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--eos_token_id', type=int, default=32021)
parser.add_argument('--pad_token_id', type=int, default=32021)
parser.add_argument('--num_return_sequences', type=int, default=1)
parser.add_argument('--max_repeat_prompt', type=int, default=10)
args = parser.parse_args()


# cuda devices
if torch.cuda.is_available():
    if args.gpu_ids is None:
        device = torch.device("cuda")
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        device = torch.device(f"cuda:{args.gpu_ids[0]}")
        gpu_ids = args.gpu_ids
    print(f"Using GPU(s): {gpu_ids}")
    torch.cuda.set_device(device)

else:
    device = torch.device("cpu")
    gpu_ids = []
    print("CUDA is not available. Using CPU.")

# quantization
if args.quantization:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, #4bit quant
        bnb_4bit_compute_dtype=torch.float16,
        # load_in_8bit=True, #8bit quant
        # llm_int8_enable_fp32_cpu_offload=True
    )
else:
    quantization_config = None


# load model
pretrained_model_path = args.model_path
config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_path
    )

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_path,
    quantization_config=quantization_config,
    device_map='auto',
    )

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_path,
    )


# flask API
app = Flask(__name__)
CORS(app)

@app.route(f'/completions', methods=['POST'])
def completions():
    content = request.json
    prompt = content['prompt']
    prompt = [{'role': 'user', 'content': prompt}]
    repeat_prompt = content.get('repeat_prompt', 1)

    # parameters
    if 'params' in content:
        params: dict = content.get('params')
        max_new_tokens = params.get('max_new_tokens', args.max_new_tokens)
        temperature = params.get('temperature', args.temperature)
        do_sample = params.get('do_sample', args.do_sample)
        top_k = params.get('top_k', args.top_k)
        top_p = params.get('top_p', args.top_p)
        num_return_sequences = params.get('num_return_sequences', args.num_return_sequences)
        eos_token_id = params.get('eos_token_id', args.eos_token_id)
        pad_token_id = params.get('pad_token_id', args.pad_token_id)
        max_repeat_prompt = params.get('max_repeat_prompt', args.max_repeat_prompt)


    # response generation
    while True:
        inputs = tokenizer.apply_chat_template(prompt, 
                                               add_generation_prompt=True, 
                                               return_tensors='pt')       
        inputs = torch.vstack([inputs] * repeat_prompt).to(model.device)

        try:
            output = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id
            )

        except torch.cuda.OutOfMemoryError as e:
            # clear cache
            gc.collect()
            if torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()
            continue
        
        content = []
        for i, out_ in enumerate(output):
            content.append(tokenizer.decode(output[i, len(inputs[i]):], skip_special_tokens=True))
        
        # clear cache
        gc.collect()
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()

        # Send back the response.
        return jsonify({'content': content})


if __name__ == '__main__':
    app.run(host=args.host, port=args.port)
