import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re
import backoff  # for exponential backoff
import openai  # for OpenAI API calls
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)
from sys import exit
##
prompt_x = '''
Summarize and merge the following list of topics into {Fixed_Number} of final topics.
Only return the name of final top {Fixed_Number} topics, without any explanations and subtopics.
The desired output format:
Topic 1 : xx
Topic 2 : xx
Topic 3 : xx
The list of topics:
{Topic_list}
Topics:
'''

directory = './test_llama'
tps=[]
n=[]
name=[]
import os
for filename in os.listdir(directory):
    if filename.endswith(".csv"):  # Check if the file is a CSV
        # Construct full file path
        filepath = os.path.join(directory, filename)
        print(f"Processing {filename}")
        data = pd.read_csv(filepath)
        a = ', '.join(data.Topics)
        for i in [10,20,30]:
            print(i)
            formatted_prompt = prompt_x.format(Fixed_Number=i, Topic_list = a)
            content = formatted_prompt
            #print(content)
            messages = [{"role": "system", "content": 'You are an expert in topic modeling.'},
             {"role": "user", "content" : content}
            ]
            completions_with_backoff = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages= messages,
                  #temperature=0.1,
                  max_tokens=350,
                  #seed = 4396
            )    
            chat_response = completions_with_backoff.choices[0].message.content
            tps.append(chat_response)
            n.append(i)
            name.append(filename)
            
######
dd = pd.DataFrame({'file': name, 'final_n': n, 'topics': tps})            
import os
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
for i in gg.topics:
    ii = [x.split(': ')[1] for x in i.split('\n')]
    topic_vectors = model.encode(ii)
    similarity_matrix = util.pytorch_cos_sim(topic_vectors, topic_vectors).numpy()
    np.fill_diagonal(similarity_matrix, np.nan)
    mean_similarity = np.nanmean(similarity_matrix)
    #b=similar_n(ii)
    print(np.round(mean_similarity, 3))
