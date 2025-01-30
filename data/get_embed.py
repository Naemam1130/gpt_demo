import pickle
import re
import asyncio
from openai import OpenAI
import time
import aiohttp
from chromadb.utils import embedding_functions
from chromadb import PersistentClient
from datetime import datetime

# openai settings
LLM_API_URL = "https://api.openai.com/v1/chat/completions"
EMBED_API_URL = "https://api.openai.com/v1/embeddings"
OPENAI_KEY = ""
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# keyword prompt
PROMPT = "주어진 문장의 핵심 키워드 단어들을 5개 뽑아주세요. 답변은 오직 키워드 단어의 나열로만 해주세요."

# chroma DB 세팅
DB_PATH = '../chat_history_DB'
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_KEY,
    model_name=EMBED_MODEL
)
chroma_client = PersistentClient(path=DB_PATH)
rag_embedding = chroma_client.get_or_create_collection("RAG_embedding", embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})

# openai client
client = OpenAI(api_key=OPENAI_KEY)

# post GPT
async def post_gpt(session, input_sentence, url, model, embed=True):
    """async로 동시에 gpt로 데이터 처리를 하기 위한 함수

    Args:
        session (ClientSession): session
        input_sentence (List|str): embedding을 사용할 때는 str, LLM을 사용할 때는 List
        url (str): post할 URL
        model (str): 사용할 GPT 모델
        embed (bool, optional): embedding 사용할 시에 True. Defaults to True.

    Returns:
        Dict: GPT 결과물
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_KEY}",
    }
    data = {
        "model": model,
    }
    if embed:
        data['input'] = input_sentence
    else:
        data['messages'] = input_sentence
    # 실패 시 5번까지 반복
    for _ in range(5):
        try:
            async with session.post(url, headers=headers, json=data) as response:
                #연결 상태 200이 아닌경우 raise
                response.raise_for_status()
                result = await response.json()
                return result
        except Exception as e:
            print(e)

async def embed_result(data, num = 10, fails = None):
    """embedding, keyword 추출 결과물 chroma DB에 저장

    Args:
        data (Dict): 스마트 스토어 데이터
        num (int, optional): post할 parallel 수. Defaults to 10.
        fails (bool, optional): fail index 처리용 flag. Defaults to None.

    Returns:
        Tuple(List[]): embedding, keyword, fail_index 반환환
    """
    timeout = aiohttp.ClientTimeout(total=20)
    list_data = list(data.items())
    embed_responses = []
    keywords_responses = []
    fail_index = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        
        for i in range((len(data)-1)//num + 1):
            # TODO: 아래 코드 모듈화 후 fail index로 반복
            try:
                tmp_data = list_data[i*num:(i+1)*num]
                
                s = time.time()
                # get embedding
                embed_response = await asyncio.gather(*[post_gpt(session,value['documents'][:8192], EMBED_API_URL, EMBED_MODEL) for _, value in tmp_data])
                # get keywords
                keywords_response = await asyncio.gather(*[post_gpt(session,[{"role": "assistant", "content": PROMPT}, {"role": "user", "content": value['documents']}], LLM_API_URL, LLM_MODEL, embed=False) for _, value in tmp_data])
                
                embed_response = [e["data"][0]["embedding"] for e in embed_response]
                keywords_response = [k["choices"][0]["message"]['content'].strip().rstrip('.').split(',') for k in keywords_response]
                
                embed_responses += embed_response
                keywords_responses += keywords_response
                e = time.time()
                print(f'{i}\t{e-s}', end='\r')
                # chroma DB 추가가
                rag_embedding.add(
                    ids=[key for key, _ in tmp_data],
                    metadatas=[
                        {
                            "doc_name": key,
                            "text": value['text'],
                            "keyword1": keywords[0].strip(),
                            "keyword2": keywords[1].strip(),
                            "keyword3": keywords[2].strip(),
                            "keyword4": keywords[3].strip(),
                            "keyword5": keywords[4].strip(),
                            "timestamp": datetime.now().isoformat(),
                        }
                        for (key, value), keywords in zip(tmp_data, keywords_response)
                    ],
                    documents=[value['documents'] for _, value in tmp_data],
                    embeddings = embed_response
                )
            except KeyboardInterrupt:
                break
            except:
                fail_index.append(i)
                print(f'fail {i}')
                continue
            
        
    return embed_responses, keywords_responses, fail_index


if __name__ == '__main__':
    with open('final_result.pkl', 'rb') as f:
        data = pickle.load(f)

    # 공백 특수문자 처리
    replace_words = {'\xa0' : ' ', '\ufeff' : '', '\u200b': ''}
    pattern = re.compile("|".join(replace_words.keys()))

    new_data = dict()
    for d in data:
        n_d = pattern.sub(lambda m: replace_words[re.escape(m.group(0))], d)
        new_data[n_d] = pattern.sub(lambda m: replace_words[re.escape(m.group(0))], data[d])
    
    # 불필요한 텍스트 제거
    for d in new_data:
        text = new_data[d]
        new_data[d] = dict()
        if '관련 도움말/키워드' not in text:
            new_data[d]['text'] = text.split('위 도움말이 도움이 되었나요?')[0]
            new_data[d]['documents'] = d + ' : ' + new_data[d]['text']
        else:
            new_data[d]['text'], tmp = text.split('위 도움말이 도움이 되었나요?')
            new_data[d]['keywords'] = tmp.split('관련 도움말/키워드')[1][:-10]
            new_data[d]['documents'] = d + ' : ' + new_data[d]['text'] + '\n관련 도움말/키워드 : ' + new_data[d]['keywords']
    
    
    # gpt post
    embed_responses, keywords_responses, fail_index = asyncio.run(embed_result(new_data))
    # TODO: fail_index에 항목 있을 시 fail index 재시도
    
    
    """
    # 임시 파일 저장
    with open('embed_result.pkl', 'wb') as f, open('data.pkl', 'wb') as f2, open('keywords.pkl', 'wb') as f3, open('fail.pkl', 'wb') as f4:
        pickle.dump(embed_responses,f)
        pickle.dump(new_data,f2)
        pickle.dump(keywords_responses,f3)
        pickle.dump(fail_index,f4)
    """