import cv2
import os
import numpy as np

def draw_str(image, target, s):
    # Draw string for visualisation.
    x, y = target
    cv2.putText(image, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(image, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

def make_pic_2(image_dir, img_name):
    ori_path = os.path.join(image_dir, img_name + ".png")
    tar_path = os.path.join(image_dir, img_name + "_1.png")
    save_path = os.path.join(image_dir, img_name + ".jpg")    
    ori_img = cv2.imread(ori_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    tar_img = cv2.imread(tar_path)
    tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
    ori_img = cv2.resize(ori_img, (240,240), cv2.INTER_LANCZOS4)
    tar_img = cv2.resize(tar_img, (240,240), cv2.INTER_LANCZOS4)
    # ori_img[:,53:173] = tar_img[:,60:180]
    ori_img[:,70:190] = tar_img[:,60:180]
    ori_img = cv2.resize(ori_img, (256,256), cv2.INTER_LANCZOS4)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, ori_img)

def make_pic(image_dir, img_name):
    ori_path = os.path.join(image_dir, img_name + ".png")
    tar_path = os.path.join(image_dir, img_name + "_1.png")
    save_path = os.path.join(image_dir, img_name + ".jpg")    
    ori_img = cv2.imread(ori_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    tar_img = cv2.imread(tar_path)
    tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
    ori_img = cv2.resize(ori_img, (720,720), cv2.INTER_LANCZOS4)
    tar_img = cv2.resize(tar_img, (720,720), cv2.INTER_LANCZOS4)
    ori_img[:,257:617] = tar_img[:,180:540]
    ori_img = cv2.resize(ori_img, (512,512), cv2.INTER_LANCZOS4)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, ori_img)

def get_pixel_distance(rgb_frame, fake_frame, total_distance, total_pixels, nmfc_frame=None):
    # If NMFC frame is given, use it as a mask.
    mask = None
    if nmfc_frame is not None:
        mask = np.sum(nmfc_frame, axis=2)
        mask = (mask > (np.ones_like(mask) * 0.01)).astype(np.int32)
    # Sum rgb distance across pixels.
    error = abs(rgb_frame.astype(np.int32) - fake_frame.astype(np.int32))
    if mask is not None:
        distance = np.multiply(np.linalg.norm(error, axis=2), mask)
        n_pixels = mask.sum()
    else:
        distance = np.linalg.norm(error, axis=2)
        n_pixels = distance.shape[0] * distance.shape[1]
    sum_distance = distance.sum()
    total_distance += sum_distance
    total_pixels += n_pixels
    # Heatmap
    maximum = 50.0
    minimum = 0.0
    maxim = maximum * np.ones_like(distance)
    distance_trunc = np.minimum(distance, maxim)
    zeros = np.zeros_like(distance)
    ratio = 2 * (distance_trunc-minimum) / (maximum - minimum)
    b = np.maximum(zeros, 255*(1 - ratio))
    r = np.maximum(zeros, 255*(ratio - 1))
    g = 255 - b - r
    heatmap = np.stack([r, g, b], axis=2).astype(np.uint8)
    if nmfc_frame is not None:
        heatmap = np.multiply(heatmap, np.expand_dims(mask, axis=2)).astype(np.uint8)
    draw_str(heatmap, (20, 20), "%0.1f" % (sum_distance/n_pixels))
    return total_distance, total_pixels, heatmap

def compute_img(fake_path, real_path):
    fake = cv2.imread(fake_path)
    fake = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)
    real = cv2.imread(real_path)
    real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
    height, width, channels = fake.shape # 512,512
    if real.shape[0] != fake.shape[0]:
        fake = cv2.resize(fake, (real.shape[0],real.shape[1]), cv2.INTER_LANCZOS4)
    # print("height:", height, "width:", width)
    _, _, heatmap = get_pixel_distance(real, fake, 0, 0)
    img_dir = os.path.dirname(fake_path)
    img_name = os.path.basename(fake_path).split('.')[0]
    heapmap_path = os.path.join(img_dir, img_name + "_heatmap.png")
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cv2.imwrite(heapmap_path, heatmap)

def generate_mask(img_foler, img_name):
    img_path = os.path.join(img_foler, img_name + ".png")
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # print(img.shape)
    mask = np.ones(img.shape) * 255
    height, width, _ = img.shape
    for i in range(height):
        for j in range(width):
            if img[i,j,3] == 0:
                mask[i,j,0] = 0
                mask[i,j,1] = 0
                mask[i,j,2] = 0
    mask_path = os.path.join("masks", img_name + ".png")
    cv2.imwrite(mask_path, mask)

def generate_mask_img(img_foler, img_name):
    mask_path = os.path.join("masks", img_name + "_mask_rgba.png")
    img_path = os.path.join(img_foler, img_name + ".png")
    # print(img_path)
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    h, w, _ = mask.shape
    if img.shape[0] != h:
        img = cv2.resize(img, (h,w), cv2.INTER_LANCZOS4)
    img = img * (mask[:,:,:]//255)
    mask_img_path = os.path.join(img_foler, img_name + "_mask.png")
    cv2.imwrite(mask_img_path, img)

def stich_image(img_foler, img_name):
    heatmap_path = os.path.join(img_foler, img_name + "_mask_heatmap.png")
    img_path = os.path.join(img_foler, img_name + ".png")
    out_path = os.path.join(img_foler, img_name + "_stich.png")
    print(out_path)
    img = cv2.imread(img_path)
    heatmap = cv2.imread(heatmap_path)
    h, w, _ = img.shape
    if heatmap.shape[0] != h:
        heatmap = cv2.resize(heatmap, (h,w), cv2.INTER_LANCZOS4)
    output = np.concatenate([img,heatmap],axis=1)
    cv2.imwrite(out_path,output)

if __name__ == "__main__":
    # a = [1052,1055,1060,1075,1090]
    # img_dir = "./tsnet/"
    # for i in a:
    #     make_pic(img_dir, str(i).zfill(6))

    b = [180]
    img_dir = "./tsnet/"
    for i in b:
        make_pic_2(img_dir, str(i).zfill(6))
    
    # dirs = ["styleunet_2", "styleunet_3", "flowen"]
    # # real_dir = "real"
    # # for img in a:
    # #     generate_mask("real", str(img).zfill(6)+"_mask_rgba")
    # #     generate_mask_img("real", str(img).zfill(6))
    # print("mask succeed!")
    # for dir in dirs:
    #     for img in a:
    #         stich_image(dir, str(img).zfill(6))
            # generate_mask_img(dir, str(img).zfill(6))
            # fake_path = os.path.join(dir, str(img).zfill(6) + "_mask.png")
            # real_path = os.path.join(real_dir, str(img).zfill(6) + "_mask.png")
            # compute_img(fake_path, real_path)
            # fake_path = os.path.join(dir, str(img).zfill(6) + ".png")
            # real_path = os.path.join(real_dir, str(img).zfill(6) + ".png")
            # compute_img(fake_path, real_path)