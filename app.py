import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# 1. 웹 페이지 기본 레이아웃 및 테마 설정
st.set_page_config(page_title="LED 스펙 및 정품 판별기", page_icon="💡", layout="centered")

st.title("💡 인공지능 기반 LED 스펙 및 정품 판별 서비스")
st.markdown("""
인터넷 쇼핑몰(알리익스프레스 등)의 **허위 스펙(가짜 칩)** 및 고가 제품(SBT90.2)을 저가 제품(SFT70 등)으로 속여 파는 행위를 방지하기 위한 소비자 보호용 딥러닝 서비스입니다.
""")
st.info("📸 LED 칩이 잘 보이도록 초근접(매크로 렌즈) 촬영한 사진을 업로드해 주세요.")

# 2. 딥러닝 모델 구조 정의 (ResNet18) 및 가중치 불러오기
# 유저님의 프로젝트에 맞게 클래스는 총 6개로 설정되어 있습니다.
classes = ['519a', 'LHP73B', 'SBT90.2', 'SFT42R', 'SFT70', 'SFT90']
num_classes = len(classes)

import os
import urllib.request

@st.cache_resource # 모델을 한 번만 읽어오도록 메모리에 캐싱합니다.
def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # [수정본] 파일이 없으면 깃허브 Release에서 자동으로 다운로드합니다.
    weights_path = 'led_resnet18.pth'
    if not os.path.exists(weights_path):
        with st.spinner("📦 깃허브에서 모델 가중치 파일(44MB)을 다운로드하는 중입니다. 최초 1회만 진행됩니다..."):
            url = "https://github.com/guest260115/LED-Classification/releases/download/v1.0/led_resnet18.pth"
            urllib.request.urlretrieve(url, weights_path)
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠️ 모델 로드 실패: {e}")
        return None

# 3. 모델 입력용 전처리 파이프라인 (유저님의 코랩 코드와 100% 일치)
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
    # 화면을 좌우 2개 칸으로 나누어 업로드 이미지와 전처리 이미지를 동시에 보여줍니다.
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='📤 업로드된 원본 이미지', use_container_width=True)
        
    with col2:
        # CLAHE 전처리가 적용된 모습 시각화 (교수님 어필 포인트)
        clahe_image = apply_clahe(image)
        st.image(clahe_image, caption='✨ CLAHE 특징 강조 이미지', use_container_width=True)
        
    if model is not None:
        with st.spinner("🔄 인공지능 모델이 LED 패턴을 분석 중입니다..."):
            # 전처리 및 모델 추론
            img_t = val_test_transforms(image).unsqueeze(0)
            with torch.no_grad():
                output = model(img_t)
                predicted = torch.max(output, 1)[1].item()
                prob = torch.nn.functional.softmax(output, dim=1)[0] * 100
                
        # 결과 대시보드 출력
        st.markdown("---")
        st.subheader("📊 인공지능 분석 결과")
        
        pred_class = classes[predicted]
        confidence = prob[predicted].item()
        
        # 고가/저가 제품에 따른 소비자 안내 메시지 커스텀 (스토리텔링)
        if pred_class == 'SBT90.2':
            st.success(f"🎉 판별 결과: **{pred_class}** (확신도: {confidence:.1f}%)")
            st.balloons() # 축하 효과 애니메이션
            st.markdown("💡 **소비자 안내:** 고가의 정품 SBT90.2 칩으로 추정됩니다. 판매 스펙과 일치합니다.")
        else:
            st.warning(f"⚠️ 판별 결과: **{pred_class}** (확신도: {confidence:.1f}%)")
            st.markdown(f"💡 **소비자 안내:** 모델 분석 결과 해당 칩은 고가의 SBT90.2가 아닌 **{pred_class}** 일 확률이 높습니다. 만약 SBT90.2 가격으로 구매하셨다면 가짜 스펙 사기를 의심해 볼 수 있습니다.")
