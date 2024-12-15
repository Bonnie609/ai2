#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1QZAd6_m-Q4pOWfTh5bQ2q5o0aMHwMP7F'


# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_column_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://www.google.com/imgres?q=%EB%8B%A8%EC%97%B4%EC%9D%B4%20%EC%9E%98%EB%90%98%EB%8A%94%20%EC%A7%91%20%EC%97%B4%ED%99%94%EC%83%81&imgurl=https%3A%2F%2Fmblogthumb-phinf.pstatic.net%2FMjAxOTEyMDFfMjg0%2FMDAxNTc1MjA1OTE0MjE3.bSJsEw_Rp2lBng7WOlZVNQ2vnlmawcnJXHeS8hZPOjsg.pY0xN-RscLAJmDg3bXnws_sDb-b07pCYCAcKWekbfysg.JPEG.jeffrey001%2FK-002.jpg%3Ftype%3Dw800&imgrefurl=https%3A%2F%2Fblog.naver.com%2Fjeffrey001%2F221724064842%3FviewType%3Dpc&docid=dXRk6JAiuuM6XM&tbnid=tzsNtSCGYvj58M&vet=12ahUKEwio5dzpj6qKAxX7sFYBHRP8AGsQM3oECBUQAA..i&w=500&h=380&hcb=2&ved=2ahUKEwio5dzpj6qKAxX7sFYBHRP8AGsQM3oECBUQAA"
        ],
        'videos': [
            "https://youtu.be/kuu4GhDoggM?feature=shared"
            
        ],
        'texts': [
            "Label 1 단열을 높이는 방법은 열이 빠져나가기 쉬운 외벽과 지붕, 창문을 수시로 점검하는 것이다."
        ]
    },
    labels[1]: {
        'images': [
            "https://www.google.com/imgres?q=%EB%8B%A8%EC%97%B4%EC%9D%B4%20%EC%9E%98%EB%90%98%EB%8A%94%20%EC%A7%91%20%EC%97%B4%ED%99%94%EC%83%81&imgurl=https%3A%2F%2Fmblogthumb-phinf.pstatic.net%2FMjAxODAxMjJfMjgz%2FMDAxNTE2NTczMjI5ODk1.pr3_m4NeQz7Ss0mrYP_SDcIZM-diA7rfyGxxEz6b8cEg.Kzsfk_8pn-3kYgZp3Lhjcs1vWcRYS9d-zZjp74Qd8Lwg.JPEG.jeffrey001%2F%25EC%25B9%25A8%25EC%258B%25A42.jpg%3Ftype%3Dw420&imgrefurl=http%3A%2F%2Fblog.naver.com%2Fjeffrey001%2F221190392378&docid=icl3rqBBTqQRgM&tbnid=ZncAZsH3thYQ8M&vet=12ahUKEwiS3PLfkKqKAxUe5DQHHVY6H8w4ChAzegQIRRAA..i&w=420&h=315&hcb=2&ved=2ahUKEwiS3PLfkKqKAxUe5DQHHVY6H8w4ChAzegQIRRAA",
            
        ],
        'videos': [
           "https://youtu.be/nPMtCuNaEqE?feature=shared"
        ],
        'texts': [
            "Label 2 최근에는 가정집에 이중창을 설치하여 공기층을 만들어 단열을 높이는 건축양상을 확인할 수 있다."
            
        ]
    }
   
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

