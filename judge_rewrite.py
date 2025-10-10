import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import json
from vllm import LLM, SamplingParams
import time  
from tqdm import tqdm

device_id = 0
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() and torch.cuda.device_count() > device_id else 'cpu')  
model_name = "Qwen2.5-3B-Instruct"
tokenizer_judge = AutoTokenizer.from_pretrained(model_name)
model_qwen = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
# Load LoRA configuration
lora_path = "judge_lora"
peft_config = PeftConfig.from_pretrained(lora_path)
assert peft_config.base_model_name_or_path == model_name, "LoRA does not match the original model!"
model_judge = PeftModel.from_pretrained(model_qwen, lora_path).to(model_qwen.device)



model_name_or_path="rewrite" 
tokenizer_rewrite = AutoTokenizer.from_pretrained(model_name_or_path)
sampling_params = SamplingParams(temperature=0.1, top_p=0.1, repetition_penalty=1.05, max_tokens=2048) 
model_rewrite = LLM(model=model_name_or_path,tensor_parallel_size=1, pipeline_parallel_size=1, dtype='float16',gpu_memory_utilization=0.7) 


def get_judge(doc,chunk):
    system_prompt='''<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
根据提供的文档内容和从中分割出来的一个文本块，请判断该文本块是否存在缺失必要信息的情况，导致信息表述不准确。该任务是利用原始文档内容中被明确阐述的内容，判断给定文本块是否需要补充必要信息。对于文本块未涉及到的内容不需要补充。
直接回复是或否，不要输出其他任何解释内容，也不要用引号、反引号或其他分隔符括住你的回复。


原始文档：
<document>{}</document>


文本块：
<chunk>{}</chunk><|im_end|>
<|im_start|>assistant
'''.format(doc,chunk)
    input_ids = tokenizer_judge.encode(system_prompt, padding=True, truncation=True, return_tensors="pt").to(model_judge.device)
    with torch.no_grad():
        outputs = model_judge(input_ids)
    # 解析预测结果
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    # 1是缺失必要信息，0是不缺失必要信息
    return predicted_class

def prompt_llm(doc,chunk):
    user_prompt='''根据提供的原始文档（被<document>和</document>包裹）和从中分割出来的一个文本块（被<chunk>和</chunk>包裹），利用原始文档的全局信息，对文本块中因信息缺失而模糊的地方进行完善和补充。
输出包括两部分，推理过程和完善好的文本块：
1. 推理过程：列举文本块哪些地方因缺失信息而导致语义不清以及相应的补充信息，被<think>和</think>包裹。
2. 完善好的文本块：依据推理过程对文本块进行重写优化，确保补充的信息自然融入文本块，使得其不存在语义不清的地方。同时保持原有写作风格不变，不要为文本块省略信息或添加未在原始文本中提及的内容。
输出内容格式如下所示：
<think>
缺失信息和补充信息的推理过程
</think>
完善好的文本块

如果你理解，按照格式直接回复内容，不要输出其他任何解释内容，也不要用引号或其他符号括住你的回复。


原始文档：
<document>{}</document>


文本块：
<chunk>{}</chunk>'''.format(doc,chunk)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer_rewrite.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Set to False to strictly disable thinking
    )
    outputs = model_rewrite.generate([text], sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
    response = generated_text
    return response


start_time = time.time() 
save_list=[]

with open('qchunker/OmniEval_ratio.json', 'r', encoding='utf-8') as cfile: 
    qa_data = json.load(cfile)


save_list=[]
for line in tqdm(qa_data):
    raw_corpus = line["raw_corpus"]
    final_chunk= line["final_chunk"]
    rewrite_chunk=[]
    original_rewrite_chunk=[]
    if_rewrite=[]
    for sentence in final_chunk:
        judge=get_judge(raw_corpus,sentence)
        if_rewrite.append(judge)
        if judge==1:
            rewrite=prompt_llm(raw_corpus,sentence)
            original_rewrite_chunk.append(rewrite)
            rewrite_chunk.append(rewrite.split('</think>')[-1].strip())
        else:
            rewrite=sentence
            original_rewrite_chunk.append('')
            rewrite_chunk.append(rewrite)
        
    save={
        "raw_corpus": line["raw_corpus"],
        "outline": line["outline"],
        "final_chunk": final_chunk,
        "original_rewrite_chunk": original_rewrite_chunk,
        "rewrite_chunk": rewrite_chunk,
        "if_rewrite": if_rewrite
    }
    save_list.append(save)
    with open('qchunker/final/OmniEval_ratio.json', 'w', encoding='utf-8') as file:
        json.dump(save_list, file,ensure_ascii=False, indent=4)
    
    
end_time = time.time()  
execution_time = end_time - start_time  
print(f"The program execution time is: {execution_time} s.")

# CUDA_VISIBLE_DEVICES=1 nohup python judge_rewrite.py >> qchunker/final/OmniEval_ratio.log 2>&1 &