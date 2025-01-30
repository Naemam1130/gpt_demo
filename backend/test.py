from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
from fastapi.logger import logger
from chromadb import PersistentClient
from datetime import datetime
import aiohttp
import json
from chromadb.utils import embedding_functions


# openai key
API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_KEY = ""
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
DB_PATH = '../chat_history_DB'

# developer PROMPT
PROMPT = {"role":"developer", "content": "네이버 스마트 스토어에 관련된 질의응답을 도와주는 챗봇입니다. 다음은 네이버 스마트 스토어의 간단한 소개입니다.\n\n\
        네이버 스마트스토어는 누구나 쉽게 온라인 상점을 개설하고 운영할 수 있는 플랫폼입니다. 판매자는 스마트스토어센터를 통해 상품 등록, 주문 관리, 정산 등 다양한 기능을 활용할 수 있습니다. \
        스마트스토어에 가입하려면 네이버 계정이 필요하며, 가입 절차와 2단계 인증 설정 등에 대한 자세한 안내는 스마트스토어 고객센터에서 확인할 수 있습니다. \
        상품 등록 시에는 카테고리 선택, 상품명 작성, 가격 설정, 옵션 설정, 대표 이미지 및 상세 설명 등록 등 여러 단계를 거치게 됩니다. 이 과정에 필요한 자료와 준비 사항에 대한 자세한 내용은 관련 블로그에서 확인할 수 있습니다. \
        또한, 스토어 관리, 혜택/마케팅, 통계 등 다양한 기능과 관련된 자주 묻는 질문은 스마트스토어 고객센터의 FAQ 섹션에서 찾아볼 수 있습니다.\n\n\
        위의 기본 소개와 더불어 추가적으로 주어진 문서를 기반으로 답변하여야 하며 또한 유저가 추가적으로 궁금해 할만한 연계질문을 제시해야 합니다.\
        다만 스마트 스토어와 관계 없는 질문에는 '저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.'라는 답변을 해야 합니다."}
RAG_PROMPT = "다음은 스마트 스토어와 관련된 문서입니다. 이 문서를 기반으로 답변을 작성하세요.\n"
app = FastAPI()

# chroma DB settings
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_KEY,
                model_name=EMBED_MODEL
            )

chroma_client = PersistentClient(path=DB_PATH)
chat_history = chroma_client.get_or_create_collection("chat_history")
rag_embedding = chroma_client.get_or_create_collection("RAG_embedding", embedding_function=openai_ef)


async def save_message_to_chroma(user_id: str, role: str, content: str):
    """ chroma DB에 대화 내역 저장

    Args:
        user_id (str): user_id
        role (str): ['user', 'assistant', 'developer'] 챗봇 대화의 주인
        content (str): 대화 내용
    """
    chat_history.add(
        ids=[f"{user_id}-{datetime.now().isoformat()}"],
        metadatas=[
            {
                "user_id": user_id,
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        ],
        documents=[content]
    )

async def get_user_conversation_history(user_id: str):
    """ user id로 chroma DB에서 대화내역 조회

    Args:
        user_id (str): user_id

    Returns:
        List[ dict()]: role과 대화내역이 포함된 dict로 이루어진 List
    """
    results = chat_history.get(
        where={"user_id": user_id},  # user_id 기준으로 조회
    )
    return [{"role": metadata["role"], "content": metadata["content"]} for metadata in results["metadatas"]]

async def delete_user_conversation_history(user_id: str):
    """ 대화 내역 초기화

    Args:
        user_id (str): user-id

    Returns:
        bool: 제거 성공 여부부
    """
    chat_history.delete(
        where={"user_id": user_id},  # user_id 기준으로 제거
    )
    results = chat_history.get(
        where={"user_id": user_id},  # 확인
    )
    return False if results['ids'] else True

async def stream_chatgpt_responses(input_sentence, websocket, user_id):
    """GPT에 문장을 post하여 답변을 stream으로 받음

    Args:
        input_sentence (List[Dict()]): 이전 대화내역
        websocket (Websocket): websocket
        user_id (str): user_id
    """
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_KEY}",
    }
    data={
        "model": LLM_MODEL,
        "messages": input_sentence,
        "stream": True,
    }
    all_texts = []
    for _ in range(5):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(API_URL, headers=headers, json=data) as response:
                    send_response = {'msg_type': 'Send', 'msg': "[START]"}
                    # client에 시작 전달
                    await websocket.send_text(json.dumps(send_response))
                    
                    async for line in response.content:
                        if line:
                            # 데이터 스트림에서 JSON 추출
                            decoded_line = line.decode("utf-8").strip()
                            if decoded_line.startswith("data: "):
                                decoded_line = decoded_line[6:]  # "data: " 제거
                                if decoded_line == "[DONE]":
                                    
                                    break
                                line_ = json.loads(decoded_line)
                                try:
                                    text = line_['choices'][0]['delta']['content']
                                    all_texts.append(text)
                                    send_response['msg'] = text
                                    # chunk 전달
                                    await websocket.send_text(json.dumps(send_response))
                                except:
                                    continue
                    # chromaDb에 대화내역 저장장
                    await save_message_to_chroma(user_id, "assistant", ''.join(all_texts))
            break
        except Exception as e:
            continue

# websocket 설정
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """websocket 메세지 구분 및 처리

    Args:
        websocket (WebSocket): websocket
    """
    print(f"client connected : {websocket.client}")
    await websocket.accept() # client의 websocket접속 허용
    
    while True:
        data = await websocket.receive_json()  # client 메시지 수신대기
        msg_type = data["msg_type"] 
        user_id = data["user_id"]
        # send 버튼
        if msg_type == 'Send':
            messages = data["messages"]
            # 대화 내역 수집
            history = await get_user_conversation_history(user_id)
            
            # 대화 내역 없을 시 developer Prompt 추가
            if not history:
                await save_message_to_chroma(user_id, PROMPT['role'], PROMPT["content"])
                history.append(PROMPT)
                add_Q = ''
            # 대화 내역 있을 시 챗봇의 추가 질문 처리
            else:
                add_Q = history[-1]['content'][-50:]
                # TODO: Kiwi등을 활용해 키워드 추출
            msg = messages[0]
            
            # 유저 메세지 chroma DB에 기록
            await save_message_to_chroma(user_id, msg["role"], msg["content"])
            
            # 관련 문서 검색, 유저 메세지 + 추가 질문
            docs = rag_embedding.query(query_texts=[msg["content"] + add_Q], n_results=3)
            # TODO: 추가 질문 query 분리
            
            # 검색 문서 PROMPT 추가
            rag_doc= RAG_PROMPT + ''.join([f'문서{i}: {doc}\n' for i, doc in enumerate(docs['documents'][0])])
            history = [history[0]] + [h for h in history[1:] if h['role'] != 'developer']
            history.append({'role': 'developer', 'content': rag_doc})
            history.append(msg)
            
            # chatGPT post
            await stream_chatgpt_responses(history, websocket, user_id)
            
        # delete 버튼
        elif msg_type == 'Delete':
            # chroma DB에서 대화내역 제거 후 client에 메세지 전송
            deleted = await delete_user_conversation_history(user_id)
            send_delete = {'msg_type': 'Delete', 'msg': 'Success' if deleted else 'Fail'}
            await websocket.send_text(json.dumps(send_delete))
        # load 버튼
        elif msg_type == 'History':
            # 대화 내역 검색 후 client에 메세지 전송
            history = await get_user_conversation_history(user_id)
            
            # developer prompt는 제거
            history = [h for h in history if h['role'] != 'developer']
            send_history = {'msg_type': 'History', 'msg': history}
            await websocket.send_text(json.dumps(send_history))

# 대화내역 확인용 API
@app.get("/history/{user_id}")
async def get_conversation_history(user_id):
    history = await get_user_conversation_history(user_id)
    if history:
        return JSONResponse(content={"user_id": user_id, "history": history})
    else:
        return JSONResponse(content={"error": "User ID not found"}, status_code=404)
