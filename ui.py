import streamlit as st
import requests
import json # POST 요청을 위해 추가

# --- Streamlit UI 및 상태 관리 ---

st.set_page_config(page_title="대화형 AI 어시스턴트", layout="wide")
st.title("대화형 AI 어시스턴트 💬")
st.write("이제 어시스턴트가 대화의 맥락을 기억합니다. 후속 질문을 통해 더 깊이 있는 정보를 얻어보세요.")

# 1. (핵심) Streamlit의 세션 상태(session_state)를 사용하여 대화 기록을 저장합니다.
#    이것이 프론트엔드의 '기억' 역할을 합니다.
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. 이전 대화 기록을 순회하며 채팅 형식으로 화면에 표시합니다.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. 사용자가 새로운 메시지를 입력할 수 있는 채팅 입력창을 만듭니다.
if prompt := st.chat_input("무엇이든 물어보세요..."):
    # 4. 사용자가 입력한 메시지를 대화 기록에 추가하고 화면에 표시합니다.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5. 어시스턴트의 답변을 표시할 영역을 미리 만듭니다.
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # 6. (핵심) 백엔드에 보낼 데이터를 준비합니다.
            #    사용자의 현재 입력(prompt)과 이전 대화 기록을 함께 보냅니다.
            api_url = "http://127.0.0.1:8000/ask"
            
            # 이전 대화 기록을 (user, ai) 짝으로 만듭니다.
            history_pairs = []
            # st.session_state.messages 에는 user, assistant 메시지가 순서대로 들어있습니다.
            # 짝을 맞추기 위해 2칸씩 건너뛰며 user, assistant 메시지를 묶습니다.
            for i in range(0, len(st.session_state.messages) - 1, 2):
                 # 마지막 메시지는 현재 사용자가 입력한 것이므로 제외합니다.
                if st.session_state.messages[i]["role"] == "user" and st.session_state.messages[i+1]["role"] == "assistant":
                    history_pairs.append(
                        (st.session_state.messages[i]["content"], st.session_state.messages[i+1]["content"])
                    )

            payload = {
                "input": prompt,
                "chat_history": history_pairs
            }
            
            with st.spinner("AI 어시스턴트가 생각 중..."):
                # 7. POST 방식으로 JSON 데이터를 백엔드에 보냅니다.
                response = requests.post(api_url, data=json.dumps(payload), headers={"Content-Type": "application/json"})

            if response.status_code == 200:
                result = response.json().get("response")
                message_placeholder.markdown(result)
                # 8. 어시스턴트의 최종 답변을 대화 기록에 추가합니다.
                st.session_state.messages.append({"role": "assistant", "content": result})
            else:
                error_msg = f"서버 오류: {response.status_code}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

        except requests.exceptions.RequestException as e:
            error_msg = f"서버 연결 오류: {e}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
