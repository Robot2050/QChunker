import time
import json
from tqdm import tqdm
import re

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name_or_path="qchunker-3B" 
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
sampling_params = SamplingParams(temperature=0.1, top_p=0.1, repetition_penalty=1.05, max_tokens=2048) 
model = LLM(model=model_name_or_path,tensor_parallel_size=1, pipeline_parallel_size=1, dtype='float16',gpu_memory_utilization=0.9) 



def prompt_llm(user_prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Set to False to strictly disable thinking
    )
    outputs = model.generate([text], sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
    response = generated_text
    return response

def chunking_by_llm(content):
    system_prompt='''这是一个文本分块任务，你是一名文本分割方面的专家，负责按照文档生成一个问题大纲，然后依据大纲将给定的文本分割成文本块。你必须遵守以下四个条件：
1. 利用原始文档的全局信息，为每一个文本块生成一个能够代表这段文本的完整问题，最终形成一个大纲。
2. 根据生成的大纲对文档进行分割，使得每个文本块都具有完整的逻辑表达。
3. 避免文本块过短，在识别内容转换与分块长度之间取得良好平衡。
4. 输出的每个文本块由文本块开头和结尾几个字符组成，中间内容由“[MASK]”来代替，输出格式如下：
<think>
文档的问题大纲
</think>
<chunk>文本块1的开头几个字符[MASK]文本块1的结尾几个字符</chunk>
......

如果你理解，按照格式直接回复内容，不要输出其他任何解释内容，也不要用引号或其他分隔符括住你的回复。


文档内容：
<document>{}</document>'''.format(content)
    try:
        str_result=prompt_llm(system_prompt)
        return str_result
    except Exception as e:
        print('111',flush=True)
        print(f"An error occurred: {e}.")
        return "GPT thinks prompt is unsafe"



with open('tmp_cleandata/multifield.json', 'r', encoding='utf-8') as file:  
    qa_data = json.load(file)

save_path='qchunker/multifield.json'

start_time = time.time()
save_list=[]
for content in tqdm(qa_data):
    raw_gpt_output=chunking_by_llm(content)
    save = {}
    if raw_gpt_output == "GPT thinks prompt is unsafe":
        json_str = json.dumps({'raw_corpus':content}, ensure_ascii=False)
        print('zjh111',json_str,flush=True)
    else: 
        save['raw_corpus'] = content
        save['gpt_output'] = raw_gpt_output
                
        save_list.append(save)    
        
        if len(save_list) % 100 == 0:
            with open(save_path, 'w', encoding='utf-8') as sfile:
                json.dump(save_list, sfile, ensure_ascii=False, indent=4)
with open(save_path, 'w', encoding='utf-8') as sfile:
    json.dump(save_list, sfile, ensure_ascii=False, indent=4)
    
end_time = time.time()  
# Calculate and print execution time
execution_time = end_time - start_time  
print(f"The program execution time is: {execution_time} s.")

# CUDA_VISIBLE_DEVICES=1 nohup python qchunker.py >> qchunker/multifield.log 2>&1 &
