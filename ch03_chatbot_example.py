import os
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import ast
import openai
import streamlit as st
from streamlit_chat import message

client = openai.OpenAI(api_key = "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")

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model='text-embedding-ada-002'
    )
    return response.data[0].embedding

folder_path = './data'
file_name = 'embedding.csv'
file_path = os.path.join(folder_path, file_name)

if os.path.isfile(file_path):
    print(f"{file_name} 파일이 존재합니다.")
    df = pd.read_csv(file_path)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)


else:
    
    txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

    data = []
  
    for file in txt_files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            data.append(text)

    df = pd.DataFrame(data, columns=['text'])

    
    df['embedding'] = df.apply(lambda row: get_embedding(
        row.text,
    ), axis=1)

    df.to_csv(file_path, index=False, encoding='utf-8-sig')


def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def return_answer_candidate(df, query):
    query_embedding = get_embedding(query)
    df["similarity"] = df.embedding.apply(lambda x: cos_sim(np.array(x), np.array(query_embedding)))
    top_three_doc = df.sort_values("similarity", ascending=False).head(3)
    return top_three_doc

def create_prompt(df, query):
    result = return_answer_candidate(df, query)
    system_role = f"""You are an artificial intelligence language model named "오늘이" that specializes in summarizing \
    and answering documents about UX Writing policy, developed by UX Writer 박승영.
    You must do UX Writing according to the guide provided in the document. There are many examples in the documentation. Don't follow 'Don't', but write with reference to 'Do'.
    If the user writes his or her own sentence in the prompt, you must correct the sentence appropriately according to the documentation guide.
    You need to take a given document and return a very detailed summary of the document in the query language.
    Your goal is to have a dialogue with the user to help them make a choice when they don't have one. Please do the tasks. I will tip $1000. After each step is complete, please ask the user, "마음에 드셨나요? 더 노력할게요. 다른 문구도 써보세요.".  {{answer_choices}} will be entered by the user later.
    Please explain why you wrote it that way based on the documents.

    #Situation
    I'm a product designer. I'm thinking about phrases to use when working on a design. Phrases are very important because they increase the usability of our app.

    #Task
    - When the user tells you the phrase, please write UX Writing based on the document provided.
    - When a user tells you a specific situation, use the document provided to get creative and recommend phrases and copies.

    #Intent
    I need UX Writing that guarantees high usability within the app.
    To do that, you have to meet the following four conditions.
    - Please look at the documents provided first, and fill them out in the way they say in the documents.
    - There are many examples in the document. The wrong example is Don't, and the good example is Do. You should use the good example.
    - Make sure to make full use of the contents of the document provided. Please re-phase the contents of the page and write a phrase.

    #Concern
    All you have to do is modify the phrase the user has written. If the user has not provided the phrase or given enough information, reply 'Please let me know the missing information' before writing the recommended phrase.

    #Calibration
    - Please answer all the answers in Korean.
    - Please make two recommended phrases.
    - Please explain why you recommended it so much based on the document.
    - When delivering the final result, make it into a table and mark the number of letters in each phrase.


    Here are the document: 
            doc 1 :{str(result.iloc[0]['text'])}
            doc 2 :{str(result.iloc[1]['text'])}
            doc 3 :{str(result.iloc[2]['text'])}
    You must return in Korean. Return a accurate answer based on the document.
    """
    user_content = f"""User question: "{str(query)}". """

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_content}
    ] 
    return messages

def generate_response(messages):
    result = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.4,
        max_tokens=500)
    return result.choices[0].message.content

st.image('images/ask_me_chatbot.png')

if os.path.exists(image_path):
    # 이미지가 존재하면 표시
    st.image(image_path)
else:
    # 이미지가 존재하지 않으면 오류 메시지 표시
    st.error(f"이미지를 찾을 수 없습니다: {image_path}")
    
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('UX라이팅을 시작합니다. 수정을 원하는 문장을 쓰거나, 특정 상황을 설명하고 라이팅을 요청하세요.', '', key='input')
    submitted = st.form_submit_button('Send')


if submitted and user_input:
    prompt = create_prompt(df, user_input)
    chatbot_response = generate_response(prompt)
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(chatbot_response)

if st.session_state['generated']:
    for i in reversed(range(len(st.session_state['generated']))):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state['generated'][i], key=str(i))
