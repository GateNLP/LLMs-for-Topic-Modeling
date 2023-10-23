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

test1='''
<s>[INST] <<SYS>>
Consider existing topics: (1) Computer Hardware and (2) Sport Baseball.
Read the text below and enumerate up to 3 topics, and each topic contains less than 3 words.

Make sure you to only return the topic and nothing more. The desired output format:
Topic 1: xxx
Topic 2: xxx
Topic 3: xxx
<</SYS>>
Text:
{list_of_text} [/INST]
'''

batchs='''
<s>[INST] <<SYS>>
Emumerate up to 3 topics for each text in the provided list of 5 texts, and each topic contain less than 3 words.

Do not return any explainations. The desired output format:
Text 1:
Topic: xxx
Topic: xxx
Topic: xxx

Text 2:
Topic: xxx
Topic: xxx
Topic: xxx
<</SYS>>
Text:
{list_of_text} [/INST]
'''

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
        #time.sleep(0.1)
        #try:
            #formatted_prompt = prompts.format(existing_topics = '\n'.join(sortedDictKey[:20]), list_of_text = i)
        formatted_prompt = test1.format(list_of_text = i)
        content = formatted_prompt
        sequences = pipeline(
            content,
            do_sample=True,
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
    
    
    
#############    
output, batchclean, batchraw = gpt4396(test.text, test1)  
##########
########
# Statistics
clean = Counter(batchclean)
raw = Counter(batchraw)
df_clean = pd.DataFrame(clean.items(), columns=['Topics', 'Count'])
# df_clean.to_csv("output_clean.csv", sep = ',',index = False)
df_raw = pd.DataFrame(raw.items(), columns = ['Topics', 'Count'])
# df_raw.to_csv("output_raw.csv", sep = ',',index = False)
d1 = df_raw
d1= d1.sort_values(by='Count', ascending=False)
dd=d1.Topics[:500].values
#dd
