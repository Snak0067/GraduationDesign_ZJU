from PIL import Image

def crop_image(input_image_path, output_image_path):
    with Image.open(input_image_path) as img:
        # 裁剪参数：从x=1024开始，宽度为1024，高度为1024
        start_x = 0
        start_y = 0  # 从图像顶部开始裁剪
        width = 1024
        height = 1024
        box = (start_x, start_y, start_x + width, start_y + height)
        
        # 裁剪图像
        cropped_image = img.crop(box)
        
        # 保存裁剪后的图像
        cropped_image.save(output_image_path)
        cropped_image.show()

# 使用示例
input_path = 'ccbr_110_000000.png'  # 替换为你的图片路径
output_path = 'origin.jpg'  # 裁剪后图片的保存路径

crop_image(input_path, output_path)