import cv2
import os
import re


def images_to_video(image_folder, output_path, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png"))]

    # 按数字排序（如果你的文件名包含数字）
    images.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else x)

    # 获取第一张图片的尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 写入每一帧
    frames = len(images)
    print(f"开始合成视频，共有{frames}帧")
    count = 0
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
        count += 1
        print(f"已合成{count}/{frames}帧")

    video.release()
    print(f"视频已保存到: {output_path}")


# 使用示例
images_to_video("./game1", "output.mp4", fps=120)