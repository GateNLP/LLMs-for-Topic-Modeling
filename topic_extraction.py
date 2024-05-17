import openai
from collections import Counter
import numpy as np 
import time
from tqdm import tqdm
import pandas as pd
openai.api_key = "Your API Key"
data = pd.read_csv('vaxx.csv') #ng20.csv
data = pd.read_csv('ng20.csv')
import backoff  # for exponential backoff
import openai  # for OpenAI API calls
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


bad_20ng='''From the given text, identify up to 3 specific topics.
Ensure that each topic is no more than 3 words.
Make sure to only return the topic and nothing more. The desired output format:
Topic 1: xxx
Topic 2: xxx
Topic 3: xxx
The given Text:
{list_of_text}
'''

good_20ng='''From the given text, identify up to 3 specific topics related to news categories.
Ensure that each topic is no more than 3 words.
Avoid returning general topics such as "News Articles" and "Media", as these are already known.
Make sure to only return the topic and nothing more. The desired output format:
Topic 1: xxx
Topic 2: xxx
Topic 3: xxx
The given Text:
{list_of_text}
'''

best_20ng='''You are tasked with performing topic modeling.
From the given text, identify up to 3 specific topics related to news categories (example seeds topics include: "Computer", "Sports"), ensuring each topic is no more than 3 words.
Do not generate topics such as "News Articles" and "Media" as these are already known.
Make sure to only return the topic and nothing more. The desired output format:
Topic 1: xxx
Topic 2: xxx
Topic 3: xxx 
The given Text:
{list_of_text}
'''

bad_vaxx=''' From the given text, identify up to 3 general topics.
Ensure that each topic is no more than 3 words.
Make sure to only return the topic and nothing more. The desired output format:
Topic 1: xxx
Topic 2: xxx
Topic 3: xxx
The given Text:
{list_of_text}[/INST]
'''

good_vaxx=''' From the given text, identify up to 3 specific topics related to related to COVID-19 vaccine hesitancy.
Ensure that each topic is no more than 3 words.
Avoid returning general topics such as "COVID-19" and "Vaccine," as these are already known.
Make sure to only return the topic and nothing more. The desired output format:
Topic 1: xxx
Topic 2: xxx
Topic 3: xxx
The given Text:
{list_of_text}
'''

best_vaxx=''' You are tasked with performing topic modeling. 
From the given text, identify up to 3 specific topics related to COVID-19 vaccine hesitancy (example seeds topics include: "Safety", "Trust" and "Effectiveness"), ensuring each topic is no more than 3 words.
Do not general topics such as "Vaccine" and "COVID-19" as these are already known.
Make sure to only return the topic and nothing more. The desired output format:
Topic 1: xxx
Topic 2: xxx
Topic 3: xxx 
The given Text:
{list_of_text}
'''

from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
def llama4396(data, prompts):
    from nltk.stem import LancasterStemmer, WordNetLemmatizer
    lemmer = WordNetLemmatizer()
    output=[]
    #past_summarise_topics = {'side effect': 0, 'ineffective': 0}
    past_summarise_topics = {}
    #sortedDictKey = list(dict(sorted(past_summarise_topics.items(), key=lambda item: item[1], reverse=True)).keys())
    #batch_dynamic = []
    batch_raw = []
    batch_clean = []
    for i in tqdm(data): 
        #time.sleep(0.1)
        #try:
            #formatted_prompt = prompts.format(existing_topics = '\n'.join(sortedDictKey[:20]), list_of_text = i)
        formatted_prompt = prompts.format(list_of_text = i)
        content = formatted_prompt
        sequences = pipeline(
            content,
            do_sample=False,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=3000,
        )
        for seq in sequences:
            chat_response = seq['generated_text'][len(formatted_prompt):]
        print(chat_response)
        output.append(chat_response)
        df_output = pd.DataFrame(data={"raw_output": output})
        df_output.to_csv("llamaraw_output.csv", sep = ',',index=False)     
        #for h in chat_response.values:            
        for h in chat_response.split('\n'):          
            if h[:5].lower() == 'topic':
                h = h.lower().replace('-',' ').replace('\n','').replace('\r','').replace('topic 1: ','').replace('topic 2: ','').replace('topic 3: ','').replace('topic 1:','').replace('topic 2:','').replace('topic 3:','').replace('topic: ','')
                if len(h) > 0:
                    batch_raw.append(h)
                    if h not in ['vaccine', 'covid', 'covid 19', 'covid vaccine', 'vaccination', 'virus']:
                        h = ' '.join([lemmer.lemmatize(w, 'n') for w in h.split()])
                        batch_clean.append(h)     
    clean = Counter(batch_clean)
    raw = Counter(batch_raw)
    df_clean = pd.DataFrame(clean.items(), columns=['Topics', 'Count'])
    df_clean.to_csv("llamaoutput_clean.csv", sep = ',',index = False)
    df_raw = pd.DataFrame(raw.items(), columns = ['Topics', 'Count'])
    df_raw.to_csv("llamaoutput_raw.csv", sep = ',',index = False)
    return output, batch_clean, batch_raw
    
    
def gpt4396(data, prompts):
    from nltk.stem import LancasterStemmer, WordNetLemmatizer
    lemmer = WordNetLemmatizer()
    output=[]
    batch_raw = []
    batch_clean = []
    for i in tqdm(data): 
#             time.sleep(0.1)
            try:
                formatted_prompt = prompts.format(list_of_text = i)
                content = formatted_prompt
                #print(content)
                messages = [{"role": "system", "content": 'You are an expert in topic modeling.'},
                 {"role": "user", "content" : content}
                ]
                completions_with_backoff = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages= messages,
                  temperature=0.1,
                  max_tokens=1200
                )
            except openai.error.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
                pass
            except openai.error.APIConnectionError as e:
                print(f"Failed to connect to OpenAI API: {e}")
                pass    
            except openai.error.RateLimitError as e:
                print(f"OpenAI API request exceeded rate limit: {e}")
                pass 

            except (openai.APIError,  # andling errors (for some of them I have not a clue of what they are! )
                    #requests.exceptions.Timeout, 
                    #APIConnectionError,
                    openai.error.APIError, 
                    openai.error.APIConnectionError, 
                    openai.error.RateLimitError, 
                    openai.error.ServiceUnavailableError, 
                    openai.error.Timeout): 
                pass      
            chat_response = completions_with_backoff.choices[0].message.content
            output.append(chat_response)
            df_output = pd.DataFrame(data={"raw_data": output})
            df_output.to_csv("./gpt_results/best_vaxx.csv", sep = ',',index=False)     
        #for h in chat_response.values:            
            for h in chat_response.split('\n'):          
                if h[:5].lower() == 'topic':
                    h = h.lower().replace('-',' ').replace('\n','').replace('\r','').replace('topic 1: ','').replace('topic 2: ','').replace('topic 3: ','').replace('topic 1:','').replace('topic 2:','').replace('topic 3:','').replace('topic: ','')
                    if len(h) > 0:
                        batch_raw.append(h)
                        if h not in ['COVID-19', 'Vaccine']:
                            h = ' '.join([lemmer.lemmatize(w, 'n') for w in h.split()])
                            batch_clean.append(h)  
    clean = Counter(batch_clean)
    raw = Counter(batch_raw)
    df_clean = pd.DataFrame(clean.items(), columns=['Topics', 'Count'])
    df_clean.to_csv("./gpt_results/best_vaxx_clean.csv", sep = ',',index = False)
    df_raw = pd.DataFrame(raw.items(), columns = ['Topics', 'Count'])
    df_raw.to_csv("./gpt_results/best_vaxx_raw.csv", sep = ',',index = False)
    return output, batch_clean, batch_raw
    
gpt4396(data, prompts) 
llama4396(data, prompts)   
