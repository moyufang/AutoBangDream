import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import sys
import os

# 添加路径
from song_recognition.predict_TitleNet import SongRecognition

def analyze_feature_similarity(recognizer, img_dir, num_pairs=2048):
    """
    分析特征向量相似度分布
    Args:
        recognizer: SongRecognizer实例
        img_dir: 图片目录
        num_pairs: 抽样对数
    """
    # 获取所有图片路径
    img_dir = Path(img_dir)
    img_paths = list(img_dir.glob('t-*.png'))
    
    if len(img_paths) < 2:
        print("错误: 需要至少2张图片进行分析")
        return
    
    print(f"找到 {len(img_paths)} 张图片，开始进行 {num_pairs} 对相似度分析...")
    
    similarities = []
    min_similarity = 1.0  # 初始化最小相似度
    
    for i in range(num_pairs):
        # 随机选择两张不同的图片
        img1_path, img2_path = random.sample(img_paths, 2)
        
        # 加载图片
        img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            print(f"警告: 无法加载图片 {img1_path} 或 {img2_path}")
            continue
        
        # 提取特征向量
        feature1 = recognizer.get_feature(img1)
        feature2 = recognizer.get_feature(img2)
        
        # 计算余弦相似度
        similarity = np.dot(feature1, feature2) / (
            np.linalg.norm(feature1) * np.linalg.norm(feature2)
        )
        
        similarities.append(similarity)
        
        # 更新最小相似度
        if similarity < min_similarity:
            min_similarity = similarity
        
        # 显示进度
        if (i + 1) % 100 == 0:
            print(f"已完成 {i + 1}/{num_pairs} 对分析...")
    
    return similarities, min_similarity

def plot_similarity_distribution(similarities, min_similarity):
    """
    绘制相似度分布图
    Args:
        similarities: 相似度列表
        min_similarity: 最小相似度
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 转换为numpy数组
    similarities = np.array(similarities)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制直方图
    plt.subplot(2, 1, 1)
    n, bins, patches = plt.hist(similarities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('余弦相似度')
    plt.ylabel('频数')
    plt.title(f'特征向量余弦相似度分布 (共{len(similarities)}对样本)')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息文本框
    stats_text = f'统计信息:\n样本数: {len(similarities)}\n最小值: {min_similarity:.4f}\n最大值: {similarities.max():.4f}\n平均值: {similarities.mean():.4f}\n标准差: {similarities.std():.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 绘制核密度估计
    plt.subplot(2, 1, 2)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(similarities)
    x_range = np.linspace(similarities.min(), similarities.max(), 200)
    plt.plot(x_range, kde(x_range), 'r-', linewidth=2)
    plt.fill_between(x_range, kde(x_range), alpha=0.3, color='red')
    plt.xlabel('余弦相似度')
    plt.ylabel('概率密度')
    plt.title('相似度分布核密度估计')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./song_recognition/similarity_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return similarities

def main():
    if False: ckpt_path = './song_recognition/ckpt_arcface.pth'
    else: ckpt_path = './song_recognition/ckpt_triplet.pth'
    
    """主函数"""
    # 初始化识别器
    recognizer = SongRecognition(
        ckpt_path=ckpt_path,
        img_dir='./song_recognition/title_imgs',
        feature_json_path='./song_recognition/feature_vectors.json',
        is_load_library=False
    )
    
    # 分析特征相似度
    similarities, min_similarity = analyze_feature_similarity(
        recognizer, 
        './song_recognition/title_imgs', 
        num_pairs=100
    )
    
    # 输出最小相似度
    print(f"\n{'='*50}")
    print(f"分析完成!")
    print(f"最小余弦相似度: {min_similarity:.6f}")
    print(f"平均余弦相似度: {np.mean(similarities):.6f}")
    print(f"相似度标准差: {np.std(similarities):.6f}")
    print(f"{'='*50}")
    
    # 绘制分布图
    plot_similarity_distribution(similarities, min_similarity)
    
    # 输出更多统计信息
    print(f"\n详细统计信息:")
    print(f"相似度范围: [{np.min(similarities):.6f}, {np.max(similarities):.6f}]")
    print(f"中位数: {np.median(similarities):.6f}")
    print(f"25%分位数: {np.percentile(similarities, 25):.6f}")
    print(f"75%分位数: {np.percentile(similarities, 75):.6f}")
    
    # 分析相似度分布的特征
    threshold_low = np.percentile(similarities, 5)  # 最低5%的阈值
    threshold_high = np.percentile(similarities, 95)  # 最高5%的阈值
    
    print(f"\n分布特征:")
    print(f"最低5%相似度阈值: {threshold_low:.6f}")
    print(f"最高5%相似度阈值: {threshold_high:.6f}")
    print(f"低于0.5相似度的样本比例: {np.mean(np.array(similarities) < 0.5):.4f}")
    print(f"高于0.8相似度的样本比例: {np.mean(np.array(similarities) > 0.8):.4f}")

if __name__ == '__main__':
    main()