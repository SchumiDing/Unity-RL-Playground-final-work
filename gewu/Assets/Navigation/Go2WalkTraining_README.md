# Go2机器人前进训练说明

## 概述

本训练脚本用于训练Go2机器人学习前进移动。脚本复用了站立训练的模型结构（相同的观察空间和动作空间），但调整了奖励函数以鼓励前进而不是保持静止。

## 文件说明

- `Scripts/Go2WalkAgent.cs`: 前进训练脚本，继承自ML-Agents的Agent类
- `config_go2_walk.yaml`: ML-Agents训练配置文件

## 训练目标

1. **保持身体平衡**：维持身体姿态稳定（pitch和roll角度接近0）
2. **前进移动**：达到并维持目标前进速度
3. **稳定行走**：减少不必要的侧向移动和旋转

## 与站立训练的区别

### 观察空间
- **相同**：观察空间维度完全一致（41维：17 + 2*12）
- **区别**：目标速度 `v1` 设置为正值（前进速度），而不是0

### 动作空间
- **相同**：12个连续动作（对应12个关节）

### 奖励函数
- **前进速度奖励**：鼓励达到目标前进速度（`v1 * vel[2]`）
- **侧向速度惩罚**：惩罚不必要的侧向移动
- **垂直速度惩罚**：惩罚上下移动
- **姿态奖励**：保持与站立训练相同的姿态稳定性奖励

### 步态控制
- **启用步态**：使用对角步态模式（trot gait）
- **步态参数**：`dh = 20`, `d0 = 30`, `T1 = 30`

## 使用方法

### 1. 在Unity中设置

1. 打开包含Go2机器人的场景
2. 选择Go2机器人GameObject
3. 添加`Go2WalkAgent`组件（替换或与`Go2StandAgent`并存）
4. 在Inspector中设置参数：
   - `train`: 设置为`true`以启用训练模式
   - `fixbody`: 设置为`false`（允许身体移动）
   - `maxTiltAngle`: 最大倾斜角度（默认20度）
   - `maxEpisodeSteps`: 最大episode步数（默认2000）
   - `randomizeStartPose`: 是否随机化初始姿态（建议开启）
   - `randomPoseRange`: 随机姿态范围（默认5度）
   - `targetForwardSpeed`: 目标前进速度（默认1.5 m/s）
   - `speedRandomRange`: 速度随机范围（默认0.5 m/s）

5. 添加Behavior Parameters组件：
   - Behavior Name: `gewu`（与站立训练使用相同的名称，可以复用模型）
   - Vector Observation Space Size: 41（17 + 2 * 12）
   - Actions: Continuous Actions = 12（12个关节）

### 2. 复用站立训练的模型（可选）

如果你想从站立训练的模型开始训练前进：

1. 将站立训练好的模型文件（.onnx）导入Unity
2. 在Behavior Parameters组件中选择该模型
3. 将`train`设置为`true`开始继续训练

这样可以实现**迁移学习**，从站立技能迁移到行走技能。

### 3. 开始训练

1. 将`config_go2_walk.yaml`文件放在项目根目录
2. 在命令行中运行：
```bash
mlagents-learn config_go2_walk.yaml --run-id=go2_walk_001
```

3. 在Unity中点击Play开始训练

### 4. 训练监控

训练过程中可以通过TensorBoard查看训练进度：
```bash
tensorboard --logdir=results
```

## 训练参数说明

### 超参数（config_go2_walk.yaml）

- `learning_rate`: 0.0003 - 学习率
- `batch_size`: 2048 - 批次大小
- `buffer_size`: 20480 - 经验回放缓冲区大小
- `num_epoch`: 3 - 每次更新的epoch数
- `gamma`: 0.99 - 折扣因子
- `max_steps`: 10000000 - 最大训练步数

### 脚本参数（Go2WalkAgent.cs）

- `maxTiltAngle`: 超过此角度会结束episode（默认20度）
- `maxEpisodeSteps`: 每个episode的最大步数（默认2000）
- `randomizeStartPose`: 是否随机化初始姿态以增加训练难度
- `randomPoseRange`: 初始姿态随机化范围（度）
- `targetForwardSpeed`: 目标前进速度（默认1.5 m/s）
- `speedRandomRange`: 目标速度的随机范围（默认0.5 m/s）

## 奖励函数详解

### 正向奖励
- **存活奖励**：每步给予0.1的基础奖励
- **前进速度奖励**：`2 * v1 * vel[2]` - 鼓励达到目标速度
- **高度奖励**：保持在合适高度时给予奖励

### 负向奖励（惩罚）
- **姿态惩罚**：根据pitch和roll角度进行惩罚
- **角速度惩罚**：惩罚不必要的旋转
- **侧向速度惩罚**：惩罚侧向移动
- **垂直速度惩罚**：惩罚上下移动
- **高度惩罚**：高度偏离正常范围时惩罚

## 训练技巧

1. **从站立模型开始**：如果已经训练好站立模型，可以加载该模型作为初始权重，加速训练
2. **逐步增加速度**：初始训练时可以设置较小的`targetForwardSpeed`（如0.5-1.0 m/s），然后逐步增加
3. **调整奖励权重**：如果机器人过于激进或过于保守，可以调整`ko`和`kv`参数
4. **检查训练曲线**：通过TensorBoard观察奖励曲线，确保训练稳定上升

## 注意事项

1. 确保Unity的Fixed Timestep设置为0.01（在Project Settings > Time中）
2. 训练时会自动创建24个机器人实例以加速训练
3. 如果训练不稳定，可以尝试：
   - 降低学习率
   - 增加batch_size
   - 调整奖励权重
   - 减少目标速度

## 测试训练好的模型

训练完成后，将训练好的模型（.onnx文件）导入Unity：
1. 将模型文件放在`Assets/ONNX/`目录
2. 在Behavior Parameters组件中选择该模型
3. 将`train`设置为`false`
4. 点击Play测试模型效果

## 参考

- ML-Agents文档: https://github.com/Unity-Technologies/ml-agents
- 站立训练脚本: `Assets/Navigation/Scripts/Go2StandAgent.cs`
- 参考脚本: `Assets/Playground/Go2Agent.cs`, `Assets/Playground/Go2omniAgent.cs`

