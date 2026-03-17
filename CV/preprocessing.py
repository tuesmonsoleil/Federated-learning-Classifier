import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

def remove_black_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
    return image

def remove_colored_marks(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 擴大紅色範圍
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 30, 30])
    upper_red2 = np.array([180, 255, 255])
    
    # 定義藍色範圍
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    # 創建紅色掩膜
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # 創建藍色掩膜
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 將紅色和藍色區域設為白色
    mask = cv2.bitwise_or(mask_red, mask_blue)
    image[mask > 0] = [255, 255, 255]
    
    return image

def enhance_contrast_and_sharpness(image):
    image_pil = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(image_pil)
    enhanced_image = enhancer.enhance(2)
    enhancer = ImageEnhance.Sharpness(enhanced_image)
    sharpened_image = enhancer.enhance(2)
    return np.array(sharpened_image)

def detect_and_crop_written_area(image):
    # 確保影像是灰度圖
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 偵測輪廓
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        for contour in contours:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            x = min(x, x1)
            y = min(y, y1)
            w = max(w, x1 + w1 - x)
            h = max(h, y1 + h1 - y)
        cropped = image[y:y+h, x:x+w]
    else:
        cropped = image
    return cropped

def remove_background_noise(image):
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised

def enhance_pencil_strokes(image):
    # 確保影像是灰度圖
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # 提高鉛筆筆跡對比度
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 使用形態學運算增強筆跡
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # 更小的內核
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)  # 減少迭代次數
    
    return morph

def clean_and_threshold(image):
    # 確保影像是灰度圖
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # 將影像進行二值化處理，使文字成為黑色，背景成為白色
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return binary

def process_image(input_path, output_path):
    image = cv2.imread(input_path)
    
    image = remove_black_lines(image)
    image = remove_colored_marks(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 偵測並裁剪書寫區域
    cropped_bgr = detect_and_crop_written_area(image)
    
    # 轉換為灰度圖以便進一步處理
    gray_cropped = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    
    # 去除背景雜訊，增強對比度和銳度
    denoised_cropped = remove_background_noise(gray_cropped)
    final_image = enhance_contrast_and_sharpness(denoised_cropped)
    
    # 進一步增強鉛筆筆跡
    final_image = enhance_pencil_strokes(final_image)
    
    # 清理並進行二值化處理
    final_image = clean_and_threshold(final_image)
    
    # 將圖像反轉為白底黑字
    final_image = cv2.bitwise_not(final_image)
    
    cv2.imwrite(output_path, final_image)

def process_folders(input_folders, output_folders):
    for input_folder, output_folder in zip(input_folders, output_folders):
        os.makedirs(output_folder, exist_ok=True)
        for filename in os.listdir(input_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                process_image(input_path, output_path)
                print(f'{filename} 處理完成並保存至 {output_path}')

# 設定輸入和輸出資料夾列表
input_folders = [
    "/local-pool-83/cheng-yao/mini/OpenCV/data_1/0",
    "/local-pool-83/cheng-yao/mini/OpenCV/data_1/1",
    "/local-pool-83/cheng-yao/mini/OpenCV/data_1.5/0",
    "/local-pool-83/cheng-yao/mini/OpenCV/data_1.5/1"
]
output_folders = [
    "/local-pool-83/cheng-yao/mini/OpenCV/processed_1/0",
    "/local-pool-83/cheng-yao/mini/OpenCV/processed_1/1",
    "/local-pool-83/cheng-yao/mini/OpenCV/processed_1.5/0",
    "/local-pool-83/cheng-yao/mini/OpenCV/processed_1.5/1"
]

# 處理資料夾
process_folders(input_folders, output_folders)

print("所有圖像已處理完成。")
