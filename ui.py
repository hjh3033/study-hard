import streamlit as st
import requests
import json

st.set_page_config(page_title="대화형 AI 어시스턴트", layout="wide")
st.title("대화형 AI 어시스턴트 🇰🇷")
st.write("이제 어시스턴트가 한국특허정보원(KIPRIS)의 공식 API에 연결되었습니다. 한국어로 특허를 검색해보세요.")

# --- 예시 질문 업데이트 ---
st.markdown("""
- **(New!) KIPRIS 특허 검색:** `컨베이어 벨트 장력 조절`
- **(New!) KIPRIS 특허 검색:** `자동화 시스템용 로봇 그리퍼`
- **내부 지식 질문:** `고온 환경에서 쓸만한 벨트 모델 추천해줘`
""")

# --- 나머지 UI 로직은 이전과 동일 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("무엇이든 물어보세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            api_url = "http://127.0.0.1:8000/ask"
            
            history_pairs = []
            for i in range(0, len(st.session_state.messages) - 1, 2):
                if st.session_state.messages[i]["role"] == "user" and st.session_state.messages[i+1]["role"] == "assistant":
                    history_pairs.append(
                        (st.session_state.messages[i]["content"], st.session_state.messages[i+1]["content"])
                    )

            payload = {
                "input": prompt,
                "chat_history": history_pairs
            }
            
            with st.spinner("KIPRIS 특허 데이터베이스를 검색하는 중..."):
                response = requests.post(api_url, data=json.dumps(payload), headers={"Content-Type": "application/json"})

            if response.status_code == 200:
                result = response.json().get("response")
                message_placeholder.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            else:
                error_msg = f"서버 오류: {response.status_code}\n\n{response.text}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

        except requests.exceptions.RequestException as e:
            error_msg = f"서버 연결 오류: {e}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
