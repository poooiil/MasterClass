# MasterClass
                                                                   AI自动取景
背景：
电影摄像机的控制是视听艺术的基础，但这工作量很大，通常有专人负责。
近来，虽然deep learning直接从视频帧里学习镜头运动，但是，严重依赖大型带注释的语料库，并且对“没见过”的人物互动泛化较差。
所以，我希望有一个新的框架，可以直接从2人3d动作序列预测相机，这样可以不依赖预先存在的dataset。

目的：
自动化影视取景：学习两个角色怎么动 → 摄像机怎么摆这样的映射，让 AI 模型了解电影里常见的对话/追逐/冲突镜头该怎么选机位、选景别，替代人工来定中景、近景、肩上视角等。
降低人工成本：无需摄像师经验、storyboard 也能自动生成可剪辑的镜头建议和最终的画面构图。

输入输出：
在保证两位角色都完整入框的同时，选择符合影视习惯的机位（如中景、近景、肩上视角等），并输出平滑连贯的摄像机运动轨迹。

流程：

1. 数据收集
首先收集clip（in dataset folder）从全网筛选含有两人交互场景的视频片段，数据来源包括电影、电视剧、舞台剧等，涵盖不同风格的运镜方式。筛选的标准：
视频长度不少于 6 秒；
视频中包含明确的两人交互；
视频具有明显的相机运动风格；
数据收集过程中部分素材可通过如 Shotdeck 等平台的静态帧定位原始视频

2.IoU 追踪& 人物裁切
risk_movie_pos_generator_finally.py
get_frame_feature() 和 json_output()
逐帧检测并跟踪两个主要角色的边界框，利用得到的每帧人物边界框裁切出人像。通过计算交并比 (IoU)，边界框在连续帧之间传播，确保在整个序列中角色识别的一致性和跟踪的可靠性。

3. MeTRAbs 3D 姿态提取 & 输出 JSON
2D 关键点估计：MeTRAbs 用 heatmap 算出 2D 关键点。
Perspective 优化：从 2D heatmap 优化出相对 3D 骨架 + 绝对根节点平移。

4. 关节数据预处理 
process_intergen.py
把上一步输出的序列转换成神经网络输入的 6D-旋转特征。

5.初始 Canonical D 计算
process_canon.py

6.模型训练与推理

7.可视化


google drive: dataset link password

comment




reference：
https://github.com/jianghd1996/Camera-control/blob/master/%5B2020%5D%5BSIG%5D_Example_driven/data_generation/data_processing.py



