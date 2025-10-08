# AutoBangDream(developing)

AutoBangDream 是一个基于 "mumu 模拟器 + 模拟触控 + 图像识别" 的 BangDream 游戏脚本。

# Functions
- 演奏歌曲：开环控制下演奏歌曲，即背谱演奏
- 全模式刷取脚本：舞台挑战、自由演出、协力演出、挑战演出、巡回演出、竞技演出、团竞演出
  - 注：协力演出等多人模式中，暂且仅支持在大厅公开匹配
- 多选歌模式：单曲循环、列表顺序、随机选取、不指定选曲
- 多表现模式：自定义 [Perfect, Great, Good, Bad, Miss] 比例；可选禁 FC
  - 注：开环控制+随机，不保证严格遵循比例）
- 歌曲识别：通过准备界面的歌曲名称识别歌曲
  - 注：BangDream 全游目前仅有两对同名不同谱的歌曲（シル・ヴ・プレジデント(389,462)，閃光(410,467)，但部分难度的 Level 不同，无法区分的情况仅在 Hard 及以下难度出现，一般不在使用脚本时出现，故暂时不支持区分

# Wickness
- [ ] 对主机性能要求高，掉帧严重影响歌曲演奏
- [ ] 图像识别基于神经网络，有极低概率出错（但随图片库增多，出错概率降低）
- [ ] 歌谱爬取依赖第三方网站

# Future
按 "重要性-难度" 排序
- [ ] 调试 & 发布 & 自动更新
- [ ] GUI
- [ ] 自动、稳定、无感同步 "主机-模拟器" 时差
- [ ] 多人模式中，自动抓取 Bangdori 车站车牌，自动进入房间
- [ ] 自动补火
- [ ] 歌曲演奏实现闭环控制
- [ ] 支持多分辨率，支持移动端

# Reference & Thanks

[Bangdream 判定规则](https://bbs.nga.cn/read.php?tid=37717081&rand=669)
[bestdori](https://bestdori.com/info/songs/)
[minitouch](https://github.com/openstf/minitouch)
[pytorch](https://github.com/pytorch/pytorch)
[deepseek](https://chat.deepseek.com/)