import cv2
import torch
import numpy as np
import json
from pathlib import Path
import sys
import os

from song_recognition.TitleNet import load_TitleNet
from song_recognition.train_arcface import ArcFaceLoss  # 虽然推理时不需要，但加载模型需要
from song_recognition.train_triplet import TripletLoss
from utils.log import LogI, LogE, LogD, LogE

class SongRecognition:
  def __init__(self, ckpt_path, img_dir:str, feature_json_path:str, is_load_library:bool=True):
    """
    歌曲识别器
    Args:
      model_path: 模型权重路径
      img_dir: 图片目录路径
      feature_json_path: 特征向量JSON文件路径
    """
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.img_dir = Path(img_dir)
    self.feature_json_path = Path(feature_json_path)
    
    # 加载模型
    self.model = load_TitleNet(ckpt_path)
    self.model.eval()
    LogI(f"TitleNet loaded: {ckpt_path}")
    
    # 加载或创建特征检索库
    self.feature_library = self._load_or_create_feature_library(is_load_library)
  
  def _load_or_create_feature_library(self, is_load:bool = True):
    """加载或创建特征检索库"""
    if self.feature_json_path.exists() and is_load:
      # 从JSON文件加载特征库
      with open(self.feature_json_path, 'r', encoding='utf-8') as f:
        feature_library = json.load(f)
      LogI(f"features library loaded, including {len(feature_library)} songs")
    else:
      # 创建新的特征库
      feature_library = []
      img_paths = sorted(list(self.img_dir.glob('t-*.png')))
      
      LogI(f"build features library, got {len(img_paths)} imgs")
      for img_path in img_paths:
        # 从文件名提取歌曲ID
        song_id = int(img_path.stem.split('-')[1])
        
        # 加载图像并提取特征
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
          LogE(f"loading img failed: {img_path}")
          continue
        
        feature = self.get_feature(img)
        feature_library.append({
          'song_id': song_id,
          'feature_vector': feature.tolist(),
          'file_name': img_path.name
        })
      
      # 保存特征库到JSON文件
      self._save_feature_library(feature_library)
    
    return feature_library
  
  def _save_feature_library(self, feature_library=None):
    """保存特征检索库到JSON文件"""
    if feature_library is None:
      feature_library = self.feature_library
    
    with open(self.feature_json_path, 'w', encoding='utf-8') as f:
      json.dump(feature_library, f, ensure_ascii=False, indent=2)
    LogI(f"save features library to \"{self.feature_json_path}\"")
  
  def get_feature(self, img):
    """
    提取图像特征向量
    Args:
      img: 36x450 灰度图像 (numpy数组)
    Returns:
      feature: 128维特征向量 (numpy数组)
    """
    # 确保图像尺寸正确
    if img.shape != (36, 450):
      img = cv2.resize(img, (450, 36))
    
    # 应用阈值160的mask
    _, thresholded = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
    
    # 归一化到[0,1]
    normalized = thresholded.astype(np.float32) / 255.0
    
    # 添加批次和通道维度 [1, 1, H, W]
    input_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(self.device)
    
    # 提取特征
    with torch.no_grad():
      feature = self.model(input_tensor)
      feature = feature.cpu().numpy()[0]  # 转换为numpy数组
    
    return feature
  
  def add_song(self, img_path, new_song_id=None):
      """
      添加新歌曲到检索库
      Args:
        img_path: 新图片路径
        new_song_id: 新歌曲ID，如果为None则自动生成
      Returns:
        song_id: 新歌曲的ID
      """
      img_path = Path(img_path)
      
      # 加载图像
      img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
      if img is None:
        raise ValueError(f"无法加载图片: {img_path}")
      
      # 确定新歌曲ID
      if new_song_id is None:
        # 自动生成不冲突的ID
        existing_ids = [item['song_id'] for item in self.feature_library]
        new_song_id = max(existing_ids) + 1 if existing_ids else 1
      
      # 检查ID是否已存在
      existing_ids = [item['song_id'] for item in self.feature_library]
      if new_song_id in existing_ids:
        raise ValueError(f"歌曲ID {new_song_id} 已存在")
      
      # 生成新文件名
      new_file_name = f"t-{new_song_id:03d}.png"
      new_file_path = self.img_dir / new_file_name
      
      # 保存图片到图片目录
      cv2.imwrite(str(new_file_path), img)
      print(f"图片已保存: {new_file_path}")
      
      # 提取特征
      feature = self.get_feature(img)
      
      # 添加到特征库
      new_song = {
        'song_id': new_song_id,
        'feature_vector': feature.tolist(),
        'file_name': new_file_name
      }
      self.feature_library.append(new_song)
      
      # 保存更新后的特征库
      self._save_feature_library()
      
      print(f"新歌曲已添加到检索库，ID: {new_song_id}")
      return new_song_id
  
  def get_id(self, query_img):
      """
      识别查询图片对应的歌曲ID
      Args:
        query_img: 查询图像 (36x450 灰度图)
      Returns:
        song_id: 识别出的歌曲ID
        similarity: 最高相似度
      """
      # 提取查询图片特征
      query_feature = self.get_feature(query_img)
      
      # 计算与所有歌曲特征的余弦相似度
      max_similarity = -1
      best_song_id = -1
      
      for song in self.feature_library:
        library_feature = np.array(song['feature_vector'])
        
        # 计算余弦相似度
        similarity = np.dot(query_feature, library_feature) / (
          np.linalg.norm(query_feature) * np.linalg.norm(library_feature)
        )
        
        if similarity > max_similarity:
          max_similarity = similarity
          best_song_id = song['song_id']
      
      return best_song_id, max_similarity
  
  def get_similar_songs(self, query_img, top_k=5):
    """
    获取最相似的前k首歌曲
    Args:
      query_img: 查询图像
      top_k: 返回的最相似歌曲数量
    Returns:
      similar_songs: 相似歌曲列表，每个元素为(song_id, similarity)
    """
    # 提取查询图片特征
    query_feature = self.get_feature(query_img)
    
    # 计算与所有歌曲的相似度
    similarities = []
    for song in self.feature_library:
      library_feature = np.array(song['feature_vector'])
      
      similarity = np.dot(query_feature, library_feature) / (
        np.linalg.norm(query_feature) * np.linalg.norm(library_feature)
      )
      
      similarities.append((song['song_id'], similarity))
  
    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

if __name__ == '__main__':
  ckpt_path = './song_recognition/ckpt_arcface.ckpt'
  #'./song_recognition/ckpt_triplet.ckpt'
  
  # 初始化识别器
  recog = SongRecognition(
    model_path=ckpt_path,
    img_dir='./song_recognition/title_imgs',
    feature_json_path='./song_recognition/feature_vectors.json'
  )
  
  # 示例1: 添加新歌曲
  # new_id = recognizer.add_song('./new_song.png')
  # print(f"新歌曲ID: {new_id}")
  
  # 示例2: 识别歌曲
  # query_img = cv2.imread('./query.png', cv2.IMREAD_GRAYSCALE)
  # song_id, similarity = recognizer.get_id(query_img)
  # print(f"识别结果: 歌曲ID {song_id}, 相似度 {similarity:.4f}")
  
  # 示例3: 获取最相似的几首歌曲
  # similar_songs = recognizer.get_similar_songs(query_img, top_k=3)
  # for song_id, sim in similar_songs:
  #     print(f"歌曲ID: {song_id}, 相似度: {sim:.4f}")
