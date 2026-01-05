# Go2机器人自主站立训练说明

## 概述

本训练脚本用于训练Go2机器人保持稳定站立的能力。训练目标是让机器人在不移动的情况下保持平衡，抵抗各种干扰。

## 文件说明

- `Scripts/Go2StandAgent.cs`: 训练脚本，继承自ML-Agents的Agent类
- `config_go2_stand.yaml`: ML-Agents训练配置文件

## 训练目标

1. **保持身体平衡**：维持身体姿态稳定（pitch和roll角度接近0）
2. **保持静止**：不产生不必要的移动（线速度和角速度接近0）
3. **抵抗干扰**：能够从随机初始姿态恢复稳定站立

## 奖励函数设计

### 正向奖励
- **存活奖励**：每步给予0.1的基础奖励
- **高度奖励**：保持在合适高度时给予奖励

### 负向奖励（惩罚）
- **姿态惩罚**：根据pitch和roll角度进行惩罚
- **角速度惩罚**：惩罚身体旋转
- **线速度惩罚**：惩罚不必要的移动
- **高度惩罚**：高度偏离正常范围时惩罚

## 使用方法

### 1. 在Unity中设置

1. 打开包含Go2机器人的场景
2. 选择Go2机器人GameObject
3. 添加`Go2StandAgent`组件
4. 在Inspector中设置参数：
   - `train`: 设置为`true`以启用训练模式
   - `fixbody`: 设置为`false`（允许身体移动）
   - `maxTiltAngle`: 最大倾斜角度（默认20度）
   - `maxEpisodeSteps`: 最大episode步数（默认2000）
   - `randomizeStartPose`: 是否随机化初始姿态（建议开启）
   - `randomPoseRange`: 随机姿态范围（默认5度）

5. 添加Behavior Parameters组件：
   - Behavior Name: `Go2Stand`
   - Vector Observation Space Size: 根据实际关节数计算（17 + 2 * ActionNum，其中ActionNum通常是12，所以总观察空间为41）
     - 包含：3(重力方向) + 3(角速度) + 3(线速度) + 24(12个关节的位置和速度) + 3(目标速度) + 5(预留控制参数)
   - Actions: Continuous Actions = 12（12个关节）

### 2. 开始训练

1. 将`config_go2_stand.yaml`文件放在项目根目录或ML-Agents配置目录
2. 在命令行中运行：
```bash
mlagents-learn config_go2_stand.yaml --run-id=go2_stand_001
```

3. 在Unity中点击Play开始训练

### 3. 训练监控

训练过程中可以通过TensorBoard查看训练进度：
```bash
tensorboard --logdir=results
```

## 训练参数说明

### 超参数（config_go2_stand.yaml）

- `learning_rate`: 0.0003 - 学习率
- `batch_size`: 2048 - 批次大小
- `buffer_size`: 20480 - 经验回放缓冲区大小
- `num_epoch`: 3 - 每次更新的epoch数
- `gamma`: 0.99 - 折扣因子
- `max_steps`: 10000000 - 最大训练步数

### 脚本参数（Go2StandAgent.cs）

- `maxTiltAngle`: 超过此角度会结束episode（默认20度）
- `maxEpisodeSteps`: 每个episode的最大步数（默认2000）
- `randomizeStartPose`: 是否随机化初始姿态以增加训练难度
- `randomPoseRange`: 初始姿态随机化范围（度）

## 训练技巧

1. **初始训练**：可以先设置较小的`randomPoseRange`（如2-3度），让机器人学习基本的平衡
2. **逐步增加难度**：随着训练进行，可以增加`randomPoseRange`和`maxTiltAngle`
3. **调整奖励权重**：如果机器人过于保守或过于激进，可以调整`ko`和`kv`参数
4. **检查训练曲线**：通过TensorBoard观察奖励曲线，确保训练稳定上升

## 注意事项

1. 确保Unity的Fixed Timestep设置为0.01（在Project Settings > Time中）
2. 训练时会自动创建24个机器人实例以加速训练
3. 如果训练不稳定，可以尝试：
   - 降低学习率
   - 增加batch_size
   - 调整奖励权重
   - 减少随机化范围

## 测试训练好的模型

训练完成后，将训练好的模型（.onnx文件）导入Unity：
1. 将模型文件放在`Assets/ONNX/`目录
2. 在Behavior Parameters组件中选择该模型
3. 将`train`设置为`false`
4. 点击Play测试模型效果

## 参考

- ML-Agents文档: https://github.com/Unity-Technologies/ml-agents
- 参考脚本: `Assets/Playground/Go2omniAgent.cs`

