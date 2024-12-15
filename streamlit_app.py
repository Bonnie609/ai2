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
labels = ["Label 1 단열이 잘 된 집", "Label 2 단열이 안되는 집"]  # labels 배열 정의

labels = ["Label 1 단열이 잘 된 집", "Label 2 단열이 안되는 집"]  # labels 배열 정의

content_data = {
    labels[0]: {  # "Label 1"에 해당하는 데이터
        'videos': [
            "https://youtu.be/kuu4GhDoggM?feature=shared",
            "https://youtu.be/nPMtCuNaEqE?feature=shared"
        ],
        'texts': [
            "Label 1 단열을 높이는 방법은 열이 빠져나가기 쉬운 외벽과 지붕, 창문을 수시로 점검하는 것이다.",
            "Label 2 최근에는 가정집에 이중창을 설치하여 공기층을 만들어 단열을 높이는 건축양상을 확인할 수 있다."
        ]
    },
    labels[1]: {  # "Label 2"에 해당하는 데이터
        'videos': [
            "https://youtu.be/W1zgJNDL0eU?feature=shared",
            "https://youtu.be/w7rMeOWeCrI?feature=shared"
        ],
        'texts': [
            "Label 2 벽돌과 벽돌 사이에는 스타이로폼이 있어 단열을 도와준다.",
            "Label 3 단열을 가장 효과적으로 만든 예는 패시브 하우스가 있다."
        ]
    }
}

# content_data 출력
print(content_data)



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
            'videos': ["https://youtu.be/w7rMeOWeCrI?feature=shared"] * 3,
            'texts': ["단열을 가장 효과적으로 만든 예는 패시브 하우스가 있다"] * 3
        })
        display_right_content(prediction, data)


