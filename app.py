import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os
import urllib.request

# 1. 웹 페이지 기본 레이아웃 및 테마 설정
st.set_page_config(page_title="LED 스펙 및 정품 판별기", page_icon="💡", layout="centered")

# [수정 ①] 타이틀 줄바꿈 방지 및 글자 크기 최적화 (어절이 쪼개지지 않음)
st.markdown("<h1 style='font-size: 2.1rem; word-break: keep-all; margin-bottom: 10px;'>💡 인공지능 기반 LED 스펙 및 정품 판별 서비스</h1>", unsafe_allow_html=True)

st.markdown("""
인터넷 쇼핑몰(알리익스프레스 등)의 **허위 스펙(가짜 칩)** 및 고가 제품(SBT90.2)을 저가 제품으로 속여 파는 행위를 방지하기 위한 소비자 보호용 딥러닝 서비스입니다.
""")
st.info("📸 LED 칩이 잘 보이도록 초근접(매크로 렌즈) 촬영한 사진을 업로드해 주세요.")

# [수정 ②] 조사된 7종 LED의 정확한 하드웨어 상세 스펙 사전 정의
led_specs = {
    '519a': {
        '제조사': 'Nichia (니치아)', '사이즈': '3535 (3.5 x 3.5 mm)', '구동 전압': '3V',
        '최대 전류': '5 ~ 6A (터보 구동 시)', '최고 밝기': '약 1,200 ~ 1,400 lm',
        '주요 특징': '높은 수준의 연색성 및 자연스러운 색감 (CRI 90+, R9080 구현)'
    },
    'lhp531': {
        '제조사': 'Lumenpioneer (루멘파이오니어)', '사이즈': '5050 (5.0 x 5.0 mm)', '구동 전압': '3V',
        '최대 전류': '8A', '최고 밝기': '약 3,000 lm',
        '주요 특징': '3x3 격자 배열 코어가 특징인 평면형(Domeless) 가성비 칩, 우수한 중심광 직진성 발휘'
    },
    'lhp73b': {
        '제조사': 'Lumenpioneer (루멘파이오니어)', '사이즈': '7070 (7.0 x 7.0 mm)', '구동 전압': '3V',
        '최대 전류': '16 ~ 20A (약 50W)', '최고 밝기': '약 8,500 lm (스펙상)',
        '주요 특징': '거대한 체급의 초광량 대형 돔리스 칩, 고가인 SBT90.2의 경쟁 모델'
    },
    'sbt90.2': {
        '제조사': 'Luminus (루미너스)', '사이즈': '9090 (9.0 x 9.0 mm)', '구동 전압': '3V',
        '최대 전류': '18 ~ 20A', '최고 밝기': '약 5,400 ~ 6,000 lm',
        '주요 특징': '메탈 프레임 및 투명 유리창 구조. 높은 수준의 초장거리 직진성(Thrower) 칩 '
    },
    'sft42r': {
        '제조사': 'Luminus (루미너스)', '사이즈': '5050 (5.0 x 5.0 mm)', '구동 전압': '3V',
        '최대 전류': '13A', '최고 밝기': '약 3,000 lm',
        '주요 특징': '원형 발광면(Round Die) 구조, 사각 왜곡 없이 둥근 빔 형태의 직진성(Thrower) 칩'
    },
    'sft70': {
        '제조사': 'Luminus (루미너스)', '사이즈': '7070 (7.0 x 7.0 mm)', '구동 전압': '6V 또는 12V',
        '최대 전류': '7A (6V 기준)', '최고 밝기': '약 3,000 ~ 3,500 lm',
        '주요 특징': '고전압 구동 방식, 면적 대비 빛의 밀도가 높은 직진성(Thrower) 칩'
    },
    'sft90': {
        '제조사': 'Luminus (루미너스)', '사이즈': '7070 (7.0 x 7.0 mm)', '구동 전압': '3V',
        '최대 전류': '20A', '최고 밝기': '약 5,500 lm',
        '주요 특징': 'SBT90.2의 성능을 대중적인 7070 일반 패키징 직진성(Thrower) 칩'
    }
}

# 2. 딥러닝 모델 구조 정의 (깃허브 train 폴더 기준 자동 클래스 생성)
base_path = 'train'
if os.path.exists(base_path):
    classes = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
else:
    classes = ['519a', 'LHP531', 'LHP73B', 'SBT90.2', 'SFT42R', 'SFT70', 'SFT90']

num_classes = len(classes)

@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    weights_path = 'led_resnet18.pth'
    if not os.path.exists(weights_path):
        with st.spinner("📦 깃허브에서 모델 가중치 파일(44MB)을 다운로드하는 중입니다. 최초 1회만 진행됩니다..."):
            try:
                url = "https://github.com/guest260115/LED-Classification/releases/download/v1.0/led_resnet18.pth"
                urllib.request.urlretrieve(url, weights_path)
            except Exception as download_error:
                st.error(f"⚠️ 가중치 다운로드 실패: {download_error}")
                return None
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠️ 모델 로드 실패: {e}")
        return None

model = load_model()

# 3. 모델 입력용 전처리 파이프라인
def apply_clahe(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    res = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(res)

val_test_transforms = transforms.Compose([
    transforms.Lambda(apply_clahe),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. 웹사이트 이미지 업로드 및 판별 기능 구현
uploaded_file = st.file_uploader("LED 이미지를 여기에 업로드하세요...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='📤 업로드된 원본 이미지', use_container_width=True)
        
    with col2:
        clahe_image = apply_clahe(image)
        st.image(clahe_image, caption='✨ CLAHE 특징 강조 이미지', use_container_width=True)
        
    if model is not None:
        with st.spinner("🔄 인공지능 모델이 LED 패턴을 분석 중입니다..."):
            img_t = val_test_transforms(image).unsqueeze(0)
            with torch.no_grad():
                output = model(img_t)
                predicted = torch.max(output, 1)[1].item()
                prob = torch.nn.functional.softmax(output, dim=1)[0] * 100
                
        st.markdown("---")
        st.subheader("📊 인공지능 분석 결과")
        
        pred_class = classes[predicted]
        confidence = prob[predicted].item()
        
        # [수정 ③] 단순 안내문 대신 세부 하드웨어 스펙 대시보드 출력
        if pred_class == 'SBT90.2':
            st.success(f"🎉 판별 결과: **{pred_class}** (확신도: {confidence:.1f}%)")
            st.balloons()
        else:
            st.warning(f"⚠️ 판별 결과: **{pred_class}** (확신도: {confidence:.1f}%)")
            
        # 대소문자 방지용 소문자 매핑 검사
        spec_key = pred_class.lower()
        if spec_key in led_specs:
            specs = led_specs[spec_key]
            st.markdown(f"### 📋 {pred_class} 실제 하드웨어 사양")
            
            # 스펙 정보를 가로 2열 격자로 예쁘게 배치
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(f"**• 제조회사:** {specs['제조사']}")
                st.markdown(f"**• 칩셋 규격:** {specs['사이즈']}")
                st.markdown(f"**• 구동 전압:** {specs['구동 전압']}")
            with sc2:
                st.markdown(f"**• 최대 전류:** {specs['최대 전류']}")
                st.markdown(f"**• 최고 밝기:** {specs['최고 밝기']}")
                
            st.markdown(f"**💡 핵심 인포:** {specs['주요 특징']}")
            
            
        else:
            st.info("💡 자동으로 인식된 LED 종류이나 상세 스펙 데이터베이스가 구축되지 않은 항목입니다.")
