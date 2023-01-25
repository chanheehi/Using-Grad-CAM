import os
from torch.utils.data import DataLoader
import torch
import numpy as np

# 다른 .py 파일 import
from CustomDataset import CustomDataset_test
from Augmentation import Get_test_transforms

if __name__ == "__main__":
    test_dir = os.path.join("./dataset") # test 경로
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # GPU 사용 가능하면 GPU 사용
    
    ckpt_filename = "chekpoints.ckpt"
    net = torch.load("model.pt", map_location=device)   # 모델 불러오기
    checkpoint = torch.load(ckpt_filename)  # 체크포인트 불러오기
    net.load_state_dict(checkpoint['state_dict'])   # 모델에 체크포인트 가중치 넣기

    # 이미지 로더 및 전처리
    test_transform = Get_test_transforms()  
    test_set = CustomDataset_test(test_dir, test_transform)

    net.eval()  # 모델 평가 모드로 전환

    from tqdm import trange
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from PIL import Image
    import cv2
    from torch.utils.data.dataset import Dataset
    target_layers = [net.model.features[-1]]    # 마지막 레이어

    cam = GradCAM(model=net, target_layers=target_layers, use_cuda=torch.cuda.is_available())   # Gradcam 생성

    def draw_cam(test_set: Dataset, idx: int) -> np.ndarray:
        input_tensor = test_set[idx][0].unsqueeze(0).to(device)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)    # Gradcam에 이미지 넣고 결과물 텐서로 받아옴

        grayscale_cam = grayscale_cam[0, :] 
        img_path = './dataset/'+test_set.data[idx][1]+"/"+test_set.data[idx][0]
        rgb_img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0   # 이미지를 RGB형태의 numpy로 받아옴
        grayscale_cam = cv2.resize(grayscale_cam, dsize=(rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_LINEAR)   # Gradcam의 결과물을 원본 이미지 크기로 resize
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True) # Gradcam의 결과물
        return visualization
    
    for i in trange(len(test_set)):
        img = draw_cam(test_set, i) # 이미지를 cam에 넣은 결과물을 numpy로 받아옴/ DataSet의 i번째 이미지
        img = Image.fromarray(img)  # numpy를 PIL image로 변환
        os.makedirs('./results/'+test_set.data[i][1], exist_ok=True)    # 경로에 폴더 없으면 생성
        img.save('./results/'+test_set.data[i][1]+"/"+test_set.data[i][0])  # 이미지 저장