import cv2
import torch
import numpy as np
import json
from pathlib import Path
import sys
import os

from song_recognition.TitleNet import load_TitleNet, prepocess_img
from utils.log import LogI, LogE, LogD, LogE
from configuration import *

class SongRecognition:
  def __init__(self, ckpt_path, img_dir:str, feature_json_path:str, is_load_library:bool=True, user_config:UserConfig=None):
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
    
    with open(SHEETS_HEADER_PATH, "r", encoding='utf-8') as file:
      self.sheets_header = json.load(file)
    
    self.uc = user_config
    
  def get_id_by_full_img(self, full_img):
    # 选歌界面有选择难度的情况，即 FIX 的情况
    is_fix = self.uc.mode == Mode.Free or (self.uc.mode == Mode.Event and self.uc.event == Event.Challenge)
    x1,y1,x2,y2 = STD_LEVEL_FIX_TITLE_REGION if is_fix else STD_LEVEL_UNFIX_TITLE_REGION
    t_img = cv2.cvtColor(full_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    # LogD(f"f_img_shape:{full_img.shape} region:{STD_LEVEL_FIX_TITLE_REGION}")
    # cv2.imwrite("./song_recognition/title_imgs_temp/t_img.png", t_img)
    song_id, similarity = self.get_id(t_img)
    
    # BangDream 全游目前仅有三对同名不同谱的歌曲，但部分难度的 Level 不同，在此根据 level 特殊处理
    if song_id in [389, 462, 410, 467, 316, 676]:
      x1,y1,x2,y2 = STD_LEVEL_FIX_LEVEL_REGION if is_fix else STD_LEVEL_UNFIX_LEVEL_REGION
      l_img = cv2.cvtColor(full_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
      level = self.get_level(l_img)
      if song_id == 410: # 410 难度 26 21 14 8
        safe = self.uc.level in [Level.Expert, Level.Hard]
        if(self.uc.level == Level.Expert and level == 27) or \
          (self.uc.level == Level.Hard   and level == 22) or \
          (self.uc.level == Level.Normal and level == 14) or \
          (self.uc.level == Level.Easy   and level ==  8): song_id = 467
      if song_id == 462: # 462 难度 25 21 14 7
        safe = self.uc.level in [Level.Expert, Level.Normal, Level.Easy]
        if(self.uc.level == Level.Expert and level == 26) or \
          (self.uc.level == Level.Hard   and level == 21) or \
          (self.uc.level == Level.Normal and level == 13) or \
          (self.uc.level == Level.Easy   and level ==  8): song_id = 389
      if song_id == 316: # 316 难度 25 19 13 7
        safe = self.uc.level in [Level.Hard, Level.Normal]
        if(self.uc.level == Level.Expert and level == 25) or \
          (self.uc.level == Level.Hard   and level == 22) or \
          (self.uc.level == Level.Normal and level == 12) or \
          (self.uc.level == Level.Easy   and level ==  7): song_id = 676
    else: safe = True
    
    return song_id, self.sheets_header[str(song_id)][1], similarity, safe
  
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
      img_paths = sorted(list(self.img_dir.glob('?-*.png')))
      
      LogI(f"build features library, got {len(img_paths)} imgs")
      for img_path in img_paths:
        # 从文件名提取歌曲ID
        ty = img_path.stem.split('-')[0]
        if ty == 't':
          song_id = int(img_path.stem.split('-')[1])
        elif ty == 'l':
          song_id = -int(img_path.stem.split('-')[1])
        else:
          raise ValueError(f"Unknown ty:\"{ty}\"")
        
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
      
    input_tensor = prepocess_img(img).to(self.device).unsqueeze(0)
    
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
  
  def get_level(self, query_img):
    l_img = (np.ones_like(query_img)*255).astype(np.uint8)
    l_img[:, :64] = query_img[:, :64]
    level_id, similarity = self.get_id(l_img)
    if (not level_id < 0) or similarity < 0.95: raise ValueError("Unknown level")
    return -level_id
  
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
  ckpt_path = './song_recognition/ckpt_triplet.pth'
  
  # 初始化识别器
  recognizer = SongRecognition(
    ckpt_path=ckpt_path,
    img_dir='./song_recognition/title_imgs',
    feature_json_path='./song_recognition/feature_vectors.json',
    is_load_library=False
  )
  
  # 示例1: 添加新歌曲
  # new_id = recognizer.add_song('./new_song.png')
  # print(f"新歌曲ID: {new_id}")
  
  # 示例2: 识别歌曲
  # query_img = cv2.imread('./song_recognition/title_imgs/t-410.png', cv2.IMREAD_GRAYSCALE)
  # song_id, similarity = recognizer.get_id(query_img)
  # print(f"识别结果: 歌曲ID {song_id}, 相似度 {similarity:.4f}")
  
  # 示例3: 获取最相似的几首歌曲
  query_img = cv2.imread('./song_recognition/title_imgs/l-005.png', cv2.IMREAD_GRAYSCALE)
  similar_songs = recognizer.get_similar_songs(query_img, top_k=3)
  for song_id, sim in similar_songs:
      print(f"歌曲ID: {song_id}, 相似度: {sim:.4f}")
