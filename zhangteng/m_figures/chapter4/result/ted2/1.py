from PIL import Image

# def crop_image(input_image_path, output_image_path):
#     with Image.open(input_image_path) as img:
#         # 裁剪参数：从x=1024开始，宽度为1024，高度为1024
#         start_x = 0
#         start_y = 0  # 从图像顶部开始裁剪
#         width = 1024
#         height = 1024
#         box = (start_x, start_y, start_x + width, start_y + height)
        
#         # 裁剪图像
#         cropped_image = img.crop(box)
        
#         # 保存裁剪后的图像
#         cropped_image.save(output_image_path)
#         # cropped_image.show()

# # 使用示例
# input_path = 'ccbr_150_000000.png'  # 替换为你的图片路径
# output_path = 'orgin.jpg'  # 裁剪后图片的保存路径

# crop_image(input_path, output_path)

from PIL import Image
import os

def crop_images_in_folder(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)
        
        # 尝试打开图像文件
        try:
            with Image.open(file_path) as img:
                # 检查图像尺寸是否为 1024x1024
                if img.size == (1024, 1024):
                    # 计算裁剪的边界，以实现中心裁剪
                    left = (1024 - 512) / 2
                    top = (1024 - 900) / 2
                    right = (1024 + 512) / 2
                    bottom = (1024 + 900) / 2
                    
                    # 裁剪图像
                    img_cropped = img.crop((left, top, right, bottom))
                    
                    # 构建新的文件名
                    new_filename = filename.rsplit('.', 1)[0] + '_crop.' + filename.rsplit('.', 1)[1]
                    new_file_path = os.path.join(folder_path, new_filename)
                    
                    # 保存裁剪后的图像
                    img_cropped.save(new_file_path)
                    print(f"Cropped and saved {new_file_path}")
                    
        except IOError:
            # 如果打开图像失败，打印错误消息
            print(f"Error opening or processing image {file_path}")

# 设置你的图片文件夹路径
folder_path = r'D:\final\final160e'
crop_images_in_folder(folder_path)