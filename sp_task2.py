import cv2
import numpy as np
import os
import imageio

# ========== 配置 ==========
dataset_root = r"D:\cxdownload\archive (1)\dataset"
category = "badWeather"
video = "wetSnow"

input_dir = os.path.join(dataset_root, category, video, "input")
output_dir = os.path.join(dataset_root, category, video, "result_MOG2")
roi_path = os.path.join(dataset_root, category, video, "ROI.bmp")
gif_save_path = os.path.join(output_dir, "result_show.gif")

os.makedirs(output_dir, exist_ok=True)

frames = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])

# ROI读取
roi = None
if os.path.exists(roi_path):
    roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
    if roi is not None:
        _, roi = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY)

# MOG2
backSub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=False
)

fps = 15
gif_frames = []
scale = 0.5

for fname in frames:
    frame = cv2.imread(os.path.join(input_dir, fname))
    if frame is None:
        continue

    fg = backSub.apply(frame)
    bg_img = backSub.getBackgroundImage()

    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

    if roi is not None:
        fg = cv2.bitwise_and(fg, roi)

    cv2.imwrite(os.path.join(output_dir, fname), fg)

    fg_bgr = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
    combine = np.vstack([frame, bg_img, fg_bgr])

    # 缩放画面
    show_img = cv2.resize(combine, None, fx=scale, fy=scale)
    combine_rgb = cv2.cvtColor(combine, cv2.COLOR_BGR2RGB)
    # GIF也同步缩小减少体积
    combine_rgb = cv2.resize(combine_rgb, None, fx=scale, fy=scale)
    gif_frames.append(combine_rgb)

    # 弹窗用缩小后的图
    cv2.imshow("Combine_All", show_img)
    key = cv2.waitKey(25) & 0xFF
    if key == 27:
        break

if len(gif_frames) > 0:
    imageio.mimsave(gif_save_path, gif_frames, duration=1/fps, loop=0)
    print(f"GIF动图保存成功：{gif_save_path}")

cv2.destroyAllWindows()
print(f"处理完毕，掩码保存在：{output_dir}")