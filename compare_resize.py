"""
直观对比：Resize 拉伸会让图片变清晰吗？
答案：不会！拉伸只是插值，不会创造新细节。
纯 PIL 版本，无需安装 torchvision
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def create_test_image(width=500, height=464):
    """生成一张带丰富细节的测试图，方便观察拉伸后的模糊"""
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # 1. 细密的水平条纹（高频细节，最容易在拉伸后变糊）
    for y in range(0, height, 4):
        draw.line([(0, y), (width, y)], fill=(0, 0, 0), width=1)

    # 2. 细密的竖直条纹
    for x in range(0, width, 4):
        draw.line([(x, 0), (x, height)], fill=(200, 0, 0), width=1)

    # 3. 斜线网格
    for i in range(-height, width, 8):
        draw.line([(i, 0), (i + height, height)], fill=(0, 150, 0), width=1)

    # 4. 圆形（曲线边缘会变锯齿/模糊）
    draw.ellipse([50, 50, 200, 200], outline=(0, 0, 200), width=3)
    draw.ellipse([300, 250, 450, 400], outline=(200, 100, 0), width=3)

    # 5. 小文字（文字是最能体现清晰度的）
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font = ImageFont.load_default()
        font_small = font

    draw.text((20, 220), "ORIGINAL 500x464", fill=(0, 0, 0), font=font)
    draw.text((20, 250), "Text gets blurry when resized!", fill=(0, 0, 0), font=font_small)
    draw.text((20, 270), "Fine details = lost forever.", fill=(0, 0, 0), font=font_small)

    # 6. 渐变块（颜色过渡区域）
    for x in range(250, 450):
        r = int(255 * (x - 250) / 200)
        draw.line([(x, 320), (x, 420)], fill=(r, 100, 255 - r), width=1)

    # 7. 右下角小棋盘格（超精细细节）
    for y in range(400, 464, 2):
        for x in range(420, 500, 2):
            if ((x + y) // 2) % 2 == 0:
                draw.point((x, y), fill=(0, 0, 0))

    return img


def main():
    # ========== 1. 生成原始测试图 ==========
    img_original = create_test_image(500, 464)
    print(f"原始图片尺寸: {img_original.size}")  # (宽, 高)

    # ========== 2. 用 PIL 做 Resize ==========
    # PIL.Image.resize 用的也是插值算法，和 torchvision.transforms.Resize 效果一致
    # PIL 默认用的是 BICUBIC（双三次插值）

    img_512 = img_original.resize((512, 512), Image.BICUBIC)
    img_1024 = img_original.resize((1024, 1024), Image.BICUBIC)

    print(f"Resize 后尺寸: {img_512.size}, {img_1024.size}")

    # ========== 3. 直观对比展示 ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 子图1：原始图
    axes[0].imshow(img_original)
    axes[0].set_title(f"Original\n500 x 464 pixels\n(~232K pixels)", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 子图2：512x512 拉伸
    axes[1].imshow(img_512)
    axes[1].set_title(f"Resize to 512 x 512\n(~262K pixels)\n轻微变形 + 轻微模糊", fontsize=14, fontweight='bold', color='darkorange')
    axes[1].axis('off')

    # 子图3：1024x1024 拉伸
    axes[2].imshow(img_1024)
    axes[2].set_title(f"Resize to 1024 x 1024\n(~1M pixels)\n严重变形 + 明显模糊！", fontsize=14, fontweight='bold', color='crimson')
    axes[2].axis('off')

    plt.suptitle("Resize 拉伸对比：像素变多了，但清晰度下降！", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("resize_comparison.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()

    print("\n✅ 对比图已保存为 resize_comparison.png")
    print("💡 观察重点：")
    print("   - 文字边缘：越拉伸越模糊")
    print("   - 细密条纹：越拉伸越混成一片")
    print("   - 右下角棋盘格：大幅拉伸后几乎看不清")
    print("   - 结论：Resize 只是插值，不会创造真实细节！")

    # ========== 4. 额外：放大对比局部细节 ==========
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

    # 裁剪同一区域放大对比（左上角文字区域）
    crop_box = (20, 220, 200, 300)  # (left, upper, right, lower)

    axes2[0].imshow(img_original.crop(crop_box))
    axes2[0].set_title("Original Crop (180x80)", fontsize=12, fontweight='bold')
    axes2[0].axis('off')

    axes2[1].imshow(img_512.crop(crop_box))
    axes2[1].set_title("512x512 Crop (same area)", fontsize=12, fontweight='bold', color='darkorange')
    axes2[1].axis('off')

    axes2[2].imshow(img_1024.crop(crop_box))
    axes2[2].set_title("1024x1024 Crop (same area)", fontsize=12, fontweight='bold', color='crimson')
    axes2[2].axis('off')

    plt.suptitle("同一区域放大对比：拉伸后文字和边缘明显变糊", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("resize_comparison_zoom.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()

    print("\n✅ 放大对比图已保存为 resize_comparison_zoom.png")


if __name__ == "__main__":
    main()
