import openai
from collections import Counter
import numpy as np 
import time
from tqdm import tqdm
openai.api_key = [Your API Key]
#
import pandas as pd
# data=pd.read_csv('/home/yidamu/Vera_AI/covid_misinfo.csv')
data=pd.read_csv('/home/yidamu/LLM_Evaluation/cavs_clean.csv')
from sklearn.model_selection import train_test_split
##
train, test = train_test_split(data, test_size=0.2, random_state=4396, stratify=data['labels'])
test=test.reset_index()
#
test=test.sort_values(by='time')
#
del test['index']
#
#
# from simple_single import add_topics, gpt4396
#
#
import backoff  # for exponential backoff
import openai  # for OpenAI API calls
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)
    
def gpt4396(data, prompts):
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
        time.sleep(0.1)
        try:
            #formatted_prompt = prompts.format(existing_topics = '\n'.join(sortedDictKey[:20]), list_of_text = i)
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
              max_tokens=30
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
        ##########
        chat_response = completions_with_backoff.choices[0].message.content
        #
        print(chat_response)
        output.append(chat_response)
        df_output = pd.DataFrame(data={"raw_output": output})
        df_output.to_csv("21oct_raw_output.csv", sep = ',',index=False)     
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
    df_clean.to_csv("21oct_output_clean.csv", sep = ',',index = False)
    df_raw = pd.DataFrame(raw.items(), columns = ['Topics', 'Count'])
    df_raw.to_csv("21oct_output_raw.csv", sep = ',',index = False)
    return output, batch_clean, batch_raw    
    
###
test1='''
Consider existing topics:
(1) side effect and (2) ineffective
Read the text blew and enumerate up to 3 topics related to the vaccine hesitancy reasons such as side effect and ineffective, and each topic contain less than 3 words.

Text:
{list_of_text}

Do not return any explainations. The desired output format:
Topic 1: xxx
Topic 2: xxx
Topic 3: xxx
'''
###
output, batchclean, batchraw = gpt4396(test.text, test1)
###
 
