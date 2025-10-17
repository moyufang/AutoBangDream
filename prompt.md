我正在写一个音游辅助，其中一个需求是“将截取到的歌曲标题的图片，识别为对应歌曲”，这个需求有以下特点
（1）对于同一首歌，标题图片在每次截图中几乎没有改变，只可能是亮度改变、少量像素噪点导致截图有变。
（2）每首歌只能有一个张 36 * 450 的灰度图片截图作为训练材料，游戏有 1000 首左右的歌曲
（3）在未来游戏可能新增歌曲，我会截图到新的图片并加入检索库，我需要的模型应该不重新训练也能进行可靠推理
这样一个少数据、多种类、少噪声的图像分类任务，适合用什么图像分类模型？还是说传统的图像算法也能解决这个问题？

我过去的尝试：
用 ocr 识别歌曲标题，虽然部分文字有错，但通过最短编辑距离匹配能将大部分标题截图匹配到正确歌曲，对少数几个匹配仍然失败特例（不超过10个），使用附加的小型图像分类模型进行暴力分类。
实践效果非常好，能完美解决需求，但有以下缺点：
（1）ocr 用的是其它开源解决方案，每次启动程序加载ocr时，以及一次文字识别时，ocr耗时巨大，ocr成为程序的性能瓶颈。与 ocr 相比，图像分类模型性能开销小。
（2）完成需求的方式从逻辑上不够统一，像是打满补丁的丑陋怪物，我希望有逻辑上端到端的一次性解决方案，就像我上面提到的那个少数据、多种类、少噪声的图像分类模型
--------------------------------------------
对于我的任务，我打算采用 ArcFaceLoss 与 TripletLoss 分别训练两个模型，然后比较优劣，但我不知道backbone该选啥，这样一个少数据、多种类、少噪声的图像分类任务，适合用什么图像分类模型？
--------------------------------------------
我正在写一个音游辅助，其中一个需求是“将截取到的歌曲标题的图片，识别为对应歌曲”，这个需求有以下特点
（1）对于同一首歌，标题图片在每次截图中几乎没有改变，只可能是亮度改变、少量像素噪点导致截图有变。
（2）每首歌只能有一个张 36 * 450 的灰度图片截图作为训练材料，游戏有 1000 首左右的歌曲
（3）在未来游戏可能新增歌曲，我会截图到新的图片并加入检索库，我需要的模型应该不重新训练也能进行可靠推理

接下来你需要根据 “我的需求” 和 “你的任务” 逐步完成 4 个阶段的 pytorch 的代码编写，收到回答后你不需要一次性将4个阶段的代码全写完，我将会在后续对话中让你逐步完成每一阶段的代码。

现在请你读懂并理解我的需求和你的任务

“我的需求”：
少数据，少噪声，多类别的图片分类任务。
训练集是 1000 张左右的 36x450 大小的 灰度 图片，每张图片一个种类，路径格式为'./song_recognition/title_imgs/t-%3d.png'。
推理时实际在做图片检索的任务，推理输入是 1000 张图片中的某一张，但可能有轻微亮度变化和少量像素噪声。

“你的任务”：用pytorch逐步完成以下任务：
“1. 数据准备
   - 收集1000张歌曲标题图片
   - 定义增强策略：亮度变化（用不同threshold(150,155,160,165,170)的mask模拟，推理时固定用160的threshold）

2. 模型选择
   - Backbone: MobileNetV3-small
   - 特征维度: 128
   - 输出: L2归一化后的特征向量

3. 训练配置
   - 损失: ArcFace (margin=0.5)
   - 优化器: 我不确定，你推荐一个
   - Batch构造: 每batch 共b(b=32)个类别，每类别4个增强样本

4. 推理流程
   - 提取查询图片特征
   - 计算与所有歌曲特征的余弦相似度
   - 返回相似度最高的歌曲ID
”
--------------------------------------------
现在请你编写第2阶段的源码，包括 MobileNetV3-small 和  128 维转 L2 归一层的 pytorch 实现，我需要的是完整源码，不是从官网上下载下来的预训练模型，注意在你写的源码中，参数初始化应选择如kaiming法之类的高级初始化，模型命名为 TitleNet
--------------------------------------------
我将模型代码保存为 song_recognition/TitleNet.py 代码了，现在请你编写训练代码 train.py，同时完成阶段1（数据准备） 和 阶段3（训练配置） 的任务，具体如下（与对话开始的描述稍微不同,增强数据的threshold应为[150,155,160，165,170]，即与160对称分布）

用pathlib.Path从 './song_recognition/title_imgs/' 中读入图片地址构造出SongTitleDataset（对图片顺序打乱），然后自写BatchMaker，在每epoch中的每batch中即时构造训练数据，构造方式是以 mask_threshold \in [150,155,160,165,170] 的生成的图片为训练输入，标签为该类，这样一个类别构造出 5 个训练数据，再考虑到每batch 共b=32个类别，每batch共处理 5*32=160 个训练数据
--------------------------------------------
现在假设我的模型已经训练好了，保存为“./song_recognition/TitleNet.ckpt”，ckpt内容是：
“final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
    }
    torch.save(final_checkpoint, final_path)”
请你编写第4阶段的代码 "predict.py"，在其中图像处理用opencv，此外还需
（1）导入 TitleNet 和 model
（2）编写一个 get_feature 的函数，get_feature 输入是一张 36*450 的灰度图，你需要对 它取 threshold = 160 的 mask，除  255.0 后用 TitleNet 得出特征向量。
（3）鉴于还没有对 1000 张左右的图片建立检索库，你需要用 pathlib 抓取所有图片并建立检索库，格式应该是一个元素为[向量, 种类id]的列表，这个列表应该能直接保存为 json 文件"./song_recognition/feature_vectors.json"。这里我提示你，一个图片的名称格式是"t-%3d.png"，它的种类 id 就是 "%3d"。
（4）编写一个 add_song 函数，输入文件路径，保证图片名称仍然是 “t-%3d.png”，且种类 id 不与检索库中的冲突，该函数需要计算它的特征向量并加入到检索库中，另外也要把新增图片保存到 './song_recognition/title_imgs/'下，注意保存已更新的检索库
（5）编写一个 get_id 函数，输入灰度图，返回种类 id。
--------------------------------------------
我修改了一些训练参数，以下是训练过程的控制台输出。为什么Loss一开始居高不下，然后突然断崖式下降？还有0.5左右的Loss大概是什么水平？
"
D:\AutoGame\AutoBangDream>set PYTHONPATH=d:\AutoGame\AutoBangDream && python -u "d:\AutoGame\AutoBangDream\song_recognition\arcface.py"
使用设备: cpu
加载了 653 张图片
num_classes:653 batch_classes:64
开始训练...
Batch [18/72], Loss: 48.5695 batch_time:7.5065
Batch [36/72], Loss: 41.7642 batch_time:7.0992
Batch [54/72], Loss: 40.3499 batch_time:7.2553
Batch [72/72], Loss: 41.0730 batch_time:7.3661
Epoch [1/20], Average Loss: 44.0695, LR: 0.009938
模型已保存: ./song_recognition/ckpt.pth
Batch [18/72], Loss: 38.6978 batch_time:7.2915
Batch [36/72], Loss: 37.5903 batch_time:7.5431
Batch [54/72], Loss: 37.1432 batch_time:7.0163
Batch [72/72], Loss: 36.0041 batch_time:7.1862
Epoch [2/20], Average Loss: 37.9183, LR: 0.009755
模型已保存: ./song_recognition/ckpt.pth
Batch [18/72], Loss: 34.4465 batch_time:7.1870
Batch [36/72], Loss: 30.0598 batch_time:7.1412
Batch [54/72], Loss: 25.8519 batch_time:7.2189
Batch [72/72], Loss: 20.2025 batch_time:4.6480
Epoch [3/20], Average Loss: 29.5645, LR: 0.009455
模型已保存: ./song_recognition/ckpt.pth
Batch [18/72], Loss: 14.7035 batch_time:5.7456
Batch [36/72], Loss: 9.1538 batch_time:6.9517
Batch [54/72], Loss: 4.1129 batch_time:7.2396
Batch [72/72], Loss: 1.3352 batch_time:6.9514
Epoch [4/20], Average Loss: 9.4304, LR: 0.009045
模型已保存: ./song_recognition/ckpt.pth
Batch [18/72], Loss: 0.8203 batch_time:7.1230
Batch [36/72], Loss: 0.7358 batch_time:4.7422
Batch [54/72], Loss: 0.7522 batch_time:7.3027
Batch [72/72], Loss: 0.6815 batch_time:7.0124
Epoch [5/20], Average Loss: 0.8191, LR: 0.008536
模型已保存: ./song_recognition/ckpt.pth
Batch [18/72], Loss: 0.6942 batch_time:5.3783
Batch [36/72], Loss: 0.6397 batch_time:5.2356
Batch [54/72], Loss: 0.6280 batch_time:7.0931
Batch [72/72], Loss: 0.6506 batch_time:4.8320
Epoch [6/20], Average Loss: 0.6453, LR: 0.007939
模型已保存: ./song_recognition/ckpt.pth
Batch [18/72], Loss: 0.6022 batch_time:4.3856
Batch [36/72], Loss: 0.5898 batch_time:4.6368
Batch [54/72], Loss: 0.5498 batch_time:4.4155
Batch [72/72], Loss: 0.5631 batch_time:4.9778
Epoch [7/20], Average Loss: 0.5794, LR: 0.007270
"
--------------------------------------------
请写一个验证脚本 show_features_statistic.py，用 opencv 处理图像，要求
（1）从 predict 导入 SongRecognizer
（2）从"./song_recognition/title_imgs" 抽取 2 张不同的图片，然后用 SongRecognizer计算两者的特征向量和余项相似度，这样的统计抽样进行 m(m=2048)轮，然后用matplotlib展示余项距离分布。
（3）控制台输出这 m 对的最小余项相似度
--------------------------------------------
根据你的建议，在Triplet Loss 构造增强数据和三元组时，我修改我的策略

每批次中从图库中随机选取t张图片，对批次内每张灰度图片I(保证36*450)，用不同mask threshold阈值+随机噪声(<255*0.01)+微小位移（在x方向平移不超过正负3个像素，在y方向上平移不超过正负10个像素）构造 a 个正样本，从n=1000张左右的图片中随机选取 b 个不同的图片作为负样本。用待训练的模型跑出 a 个正样本中与 I 距离前 r*a 小的、b 个负样本中与 I 距离前 r*b 大的，然后构造出 (r*a)*(r*b) 个三元组，其中 r 是 0.1<r<=1，这样一个批次有 t*(r*a)*(r*b) 个三元组用于训练。

请再次评价这个策略，如果有，请提出改进建议
--------------------------------------------
我正在写一个音游辅助，其中一个需求是“将截取到的歌曲标题的图片，识别为对应歌曲”，这个需求有以下特点
（1）对于同一首歌，标题图片在每次截图中几乎没有改变，只可能是亮度改变、少量像素噪点导致截图有变。
（2）每首歌只能有一个张截图作为训练材料，游戏有 1000 首左右的歌曲
（3）在未来游戏可能新增歌曲，我会截图到新的图片并加入检索库，我需要的模型应该不重新训练也能进行可靠推理

接下来你需要根据 “我的需求” 和 “你的任务” 逐步完成 4 个阶段的 pytorch 的代码编写，收到回答后你不需要一次性将4个阶段的代码全写完，我将会在后续对话中让你逐步完成每一阶段的代码。

现在请你读懂并理解我的需求和你的任务

“我的需求”：
少数据，少噪声，多类别的图片分类任务。
训练集是 1000 张左右的 36x450 大小的灰度的白底黑字图片，每张图片一个种类，路径格式为'./song_recognition/title_imgs/t-%3d.png'。
推理时实际在做图片检索的任务，推理输入是 1000 张左右图片中的某一张，但可能有轻微亮度变化、少量像素噪声、少量像素平移。

“你的任务”：用pytorch逐步完成以下任务：
1. 数据准备
   - 从 './song_recognition/title_imgs/' 中抓取所有格式为 "t-%d.png" 的 1000 张左右的歌曲标题图片，注意图片id %d 不是 0-1000 依次排列的，可能有空缺

2. 模型选择
   - Backbone: MobileNetV3-small
   - 特征维度: 128
   - 输出: L2归一化后的特征向量

3. 训练配置
   - 损失: TripletLoss (margin=0.2)
   - 优化器: 我不确定，你自由决定
   - 批次训练：即时构造增强数据、三元组

4. 推理流程
   - 提取查询图片特征
   - 计算与所有歌曲特征的余弦相似度
   - 返回相似度最高的歌曲ID
”

在每批次中：
原始图片的定义是：从title_imgs中读入的图片，进行反转并以160为threshold二值化后的图片。这样定义是因为，推理时用 160 作为固定 threshold。

采用的图片增强策略：
对于每个原始图片，构造 a 个增强数据，方法如下
  微小平移：对36*450大小的原图，做y方向（高36像素）上不超过正负3像素、x方向（宽450像素）上不超过正负10像素的随机位移，得到图A
  随机噪声：绝对幅度不超过 255.0*0.02 的高斯噪声 + 椒盐噪声，两重噪声叠加到图A后还要限制像素值在 [0.0,255.0]，得到图B
  亮度变化：从[150,155,165,170]随机选取 threshold 作为mask对图B二值化来模拟亮度（推理时固定用160的threshold），得到图C作为增强图片

采用的三元组构造策略：

我们使用多个负样本，为每个anchor选择k个最难负样本（semi-hard 意义下的，具体见下方详细描述），然后构成k个三元组。

1. 批次采样：随机选择t个类别（每个类别一张原始图片）。

2. 数据增强：对每个原始图片，根据上方描述的3个流程（平移、噪声、mask）生成a个增强图片。

3. 特征提取：将t*(1+a)张图片通过模型，得到特征。

4. 对于每个增强图片（anchor）：

正样本：对应类别的原始图片（或者也可以选择同一类别的其他增强图片，但这里为了简单，我们使用原始图片作为正样本）。

负样本：采用 semi-hard 策略，从其他所有类别中（包括原始图片和增强图片）选择到anchor距离大于正样本到anchor距离的样本中，距离anchor前 k 小的那些样本作为负样本，如果不存在或者不够，则选择距离anchor前lackness小的样本作为负样本。

每批次中，一共有 t * a 张增强图片，可以构造出 t * a 个 (anchor,positive) 组，再与对应的前 k 难例做笛卡尔积，可以构造出 t * a * k 个三元组。

5. 计算三元组损失，更新模型参数。

注意：这里我们使用批次内所有样本作为负样本库，因此每个anchor都可以找到批次内前 k 难负样本（semi-hard 意义下的）。
---------------------------------------------------------
我的python项目后端部分已经有很高完成度了，现在缺一个UI。

我对 UI 的要求有
1. UI 里有多个模块，不同模块用界面内的 Tab 标签进行切换，每个模块对接后端的一个或多个任务，每个任务有若干配置以及启动/停止，配置的种类有：bool变量、整数、n选1模式、多元元组输入（比如("newbee",0.95,0.3,0.2,0.07,0.03)）。每次配置完，warpper会自动保存，使得下次加载时使用新配置。
2. UI 需要有一个用来显示后端控制台输出的区域，即需要前端提供一个打印函数 log 供后端调用，每个模块都有一个独立的显示后端控制台输出区域。
3. 同一模块中的所有任务都是互斥的，当某个任务运行时，除非停止后，不能运行其它任务。但是不同模块之间的任务是可以并行/并发的。
4. UI 开发中，用于配置变量的组件需要逻辑与样式分离(比如radio样式可以是经典的圆点，也可以是某些自定义icon，toggle也可以是自定义图标或者产生其它css美化效果)
5. UI 开发中，前端使用 mock 去模拟后端 API
6. 我的后端是基于 python 的，最后需要有一个 python 脚本 wrapper.py 包装并控制好所有后端功能，然后与前端 UI 进行通信。

UI具体上包括若干个界面（之后随着功能扩展，会增加界面），每界面一个模块，简要描述如下
（1）模块 Scriptor：
启动/停止（调用start_scriptor()或stop_scriptor()函数），若干可供用户配置的变量（详见下方）
（2）模块 Song Recognition：需要多个启停按钮控制多个任务的执行，不允许同时运行多个界面
启动/停止“歌曲添加”（调用add_song()或stop_adding()）
启动/停止“训练歌曲识别模型”（调用train_song_recognition(), stop_train_song_recognition()）
（3）模块 UI Recognition
启动/停止“图片添加”（调用add_img()或stop_adding()）
启动/停止"训练UI识别模型”（调用train_UI_recognition(), stop_train_UI_recognition()
（4）模块 Fetch：
启动/停止“抓取歌谱”（调用fetch()），若干可供用户配置的变量（详见下方）
（5）模块 WorkfLow：
启动/停止“工作流”（调用workflow()），若干可供用户配置的变量（详见下方）

出界面UI外，我还需要一个可展开与收起的左侧边栏，用于实时显示 wrapper.py 发送过来的后端运行状态，收起时侧边栏有若干图标，展开时在图标右侧显示文本。

# 模块 Scriptor 详细说明
配置变量：
"
dilation_time       =  1000000 # 范围 1000000 ~ 1005000
correction_time     = -  45000 # 范围 int32 的范围

mumu_port = 7555            # 范围1024~65535
server_port = 31415         # 范围1024~65535
bangcheater_port = 12345    # 范围1024~65535

is_no_action        = False
is_caliboration     = False

play_one_song_id    = 655     #调用check_song_id是否合法，如果不合法，提醒用户
is_play_one_song    = False
is_restart_play     = True

is_checking_3d      = True


is_repeat          = True
MAX_SAME_STATE     = 100     # 范围 10 ~ 1000
MAX_RE_READY       = 10      # 范围 0 ~ 30

is_allow_save       = True

protected_state    = ['join_wait', 'ready_done'] # 允许用户添加和删除合法state，调用 get_avail_state() 获取所有合法的 state

special_state_list = ['ready'] # 允许用户添加和删除合法state，调用 get_avail_state() 获取所有合法的 state

# 单选参数
mode = Mode.Event # 枚举类Mode有成员 Free, Collaborate, Stage, Event, Story
event = Event.Compete # 枚举类Event有成员 Compete, Team, Tour, Challenge, Trial, Mission
choose = Choose.Loop # 枚举类Choose有成员 Loop, Random, ListUp, ListDown, No
level = Level.Expert # 枚举类Level有成员 Special, Expert, Hard, Normal, Easy
performance = Performance # 枚举类Performance有成员AllPerfect, FullCombo, Custom, DropLastCustom
weight_title = 'skilled' # 运行用户选择合法字符串，调用 get_all_weight_title() 得到所有合法字符串

# 以下是条件参数，仅在特定条件下可设置，其它条件下失效

# 当 event in [Event.Tour, Event.Compete, Team] 时
lobby = True
"

# 模块 Song Recognition 详细说明

调用 add_song 时，后端用log打印出消息辅助用户进行交互，用户需要
点击按钮 Yes, No, Drop, Stop，后端收到响应后继续执行
如果后端收到 No，之后需要用户在编辑框输入 song_id 并交给后端处理

调用 train_song_recognition 前有以下配置变量
"
is_load_model = True
epoch = 20 # 范围1 ~ 100
num_batches = 32 # 范围 1 ~ 256
batch_size = 64 # 范围 32 ~ 128
learn_rate = 0.01 # 范围 0.0001 ~ 0.1
"
调用 train_song_recognition 后，后端会用 log 打印训练信息

# 模块 UI Recognition 详细说明

调用 add_img 时，后端用log打印出消息辅助用户进行交互，用户需要
点击按钮 Yes, No, Drop, Stop，后端收到响应后继续执行
如果后端收到 No，之后需要用户在编辑框输入 img_id 并交给后端处理
调用 train_UI_recognition 前有以下配置变量
"
is_load_mode = True
epoch = 20 # 范围1 ~ 100
batch_size = 128 # 范围 32 ~ 128
learn_rate = 0.01 # 范围 0.0001 ~ 0.1
"
调用 train_UI_recognition 后，后端会用 log 打印训练信息

# 模块 Fetch 详细说明

有以下配置
"
fetch_mode = FetchMode.FetchLack # 枚举类 FetchMode 有成员 FetchOne, FetchLack, FetchSongHeader, SpecialChar
"

# 模块 Workflow 详细说明

有一个区域用来显示最大 1280 * 720 的图片，后端会提供 hsv_img
鼠标在图片上移动时，显示鼠标所知的图片像素的 HSV 颜色(调用函数get_color(x,y))和像素位置


有以下配置
"
workflow_mode = WorkflowMode.Record # 枚举类 WorkflowMode 有成员 WalkThrough, WalkThroughSheet, Capture, TraceNote, TraceFristNote
"

我采用 vue+ts+scss+vite+pinia+vue_router+web_socket 的前端UI的开发方案，不使用第三方的基础组件库。

请你读懂我的需求，并评论，但不需要给出具体的代码

----------------------------------------------------------

我的后端 wrapper.py 管理 5 个子任务 scriptor,song_recognition,ui_recognition,fetch,workflow的启动停止，与此同时与前端通过websocket保持通信。
wrapper需要通过多进程去运行 5 个子任务，且启动前向个子任务传入一个字典"{module_name}_config"。
wrapper在于前端通信时，可能收到前端修改某个运行中的子任务的config的请求（如API"/api/module_name/update_config"，传输“{module_name}_config"中的部分