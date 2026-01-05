# Go2机器人多动作控制训练说明

## 概述

本训练脚本用于训练Go2机器人学习6种基本动作：前进、左平移、右平移、原地左转、原地右转、后退。脚本复用了前进训练的模型结构，通过5个控制参数来指定不同的动作模式。

## 文件说明

- `Scripts/Go2ControlAgent.cs`: 多动作控制训练脚本
- `config_go2_control.yaml`: ML-Agents训练配置文件

## 训练目标

训练机器人根据5个控制参数执行6种动作：
1. **前进** [0,0,0,0,0] - 向前移动
2. **左平移** [1,0,0,0,0] - 向左平移
3. **右平移** [0,1,0,0,0] - 向右平移
4. **原地左转** [0,0,1,0,0] - 原地逆时针旋转
5. **原地右转** [0,0,0,1,0] - 原地顺时针旋转
6. **后退** [0,0,0,0,1] - 向后移动

## 控制参数映射

| 控制参数 | 值 | 动作 | 目标速度 |
|---------|-----|------|---------|
| [1,0,0,0,0] | 左平移 | 侧向速度 v2 = -strafeSpeed | 平移速度 |
| [0,1,0,0,0] | 右平移 | 侧向速度 v2 = +strafeSpeed | 平移速度 |
| [0,0,1,0,0] | 原地左转 | 角速度 wr = -turnSpeed | 转向速度 |
| [0,0,0,1,0] | 原地右转 | 角速度 wr = +turnSpeed | 转向速度 |
| [0,0,0,0,1] | 后退 | 前进速度 v1 = -backwardSpeed | 后退速度 |
| [0,0,0,0,0] | 前进 | 前进速度 v1 = +forwardSpeed | 前进速度 |

## 训练架构

### Agent分组
- **总Agent数**: 48个（包括原始agent）
- **分组方式**: 每8个agent为一组，共6组
- **每组对应**: 一种动作模式
  - Group 0 (Agent 0-7): 前进
  - Group 1 (Agent 8-15): 左平移
  - Group 2 (Agent 16-23): 右平移
  - Group 3 (Agent 24-31): 原地左转
  - Group 4 (Agent 32-39): 原地右转
  - Group 5 (Agent 40-47): 后退

### 观察空间
- **维度**: 41维（与前进训练相同）
  - 3维：重力方向
  - 3维：角速度
  - 3维：线速度
  - 24维：12个关节的位置和速度
  - 3维：目标速度（v1, v2, wr）
  - **5维：控制参数**（这是关键输入）

### 动作空间
- **维度**: 12维（12个关节的连续动作）

## 速度区间奖励机制

所有动作都使用速度区间奖励，鼓励匀速移动：

### 前进/后退动作
- **最优速度区间**: `optimalSpeedMin` ~ `optimalSpeedMax` (默认0.8-1.8 m/s)
- **最优区间内**: 高奖励倍数 (`highRewardMultiplier = 2.0`)
- **最优区间外**: 低奖励倍数 (`lowRewardMultiplier = 0.5`)

### 平移动作
- **最优速度区间**: `optimalSpeedMin` ~ `optimalSpeedMax` (默认0.8-1.8 m/s)
- 奖励机制与前进/后退相同

### 转向动作
- **最优速度区间**: `optimalTurnSpeedMin` ~ `optimalTurnSpeedMax` (默认0.5-1.5 rad/s)
- **最优区间内**: 高奖励倍数
- **最优区间外**: 低奖励倍数

## 使用方法

### 1. 在Unity中设置

1. 打开包含Go2机器人的场景
2. 选择Go2机器人GameObject
3. 添加`Go2ControlAgent`组件
4. 在Inspector中设置参数：
   - `train`: 设置为`true`以启用训练模式
   - `fixbody`: 设置为`false`
   - `maxTiltAngle`: 最大倾斜角度（默认20度）
   - `maxEpisodeSteps`: 最大episode步数（默认2000）
   - `randomizeStartPose`: 是否随机化初始姿态
   - `randomPoseRange`: 随机姿态范围（默认5度）
   - **速度参数**:
     - `forwardSpeed`: 前进速度（默认1.5 m/s）
     - `strafeSpeed`: 平移速度（默认1.0 m/s）
     - `backwardSpeed`: 后退速度（默认1.0 m/s）
     - `turnSpeed`: 转向角速度（默认1.0 rad/s）
   - **速度区间奖励参数**:
     - `optimalSpeedMin`: 最优速度区间最小值（默认0.8 m/s）
     - `optimalSpeedMax`: 最优速度区间最大值（默认1.8 m/s）
     - `optimalTurnSpeedMin`: 最优转向速度区间最小值（默认0.5 rad/s）
     - `optimalTurnSpeedMax`: 最优转向速度区间最大值（默认1.5 rad/s）
     - `highRewardMultiplier`: 最优速度区间奖励倍数（默认2.0）
     - `lowRewardMultiplier`: 非最优速度区间奖励倍数（默认0.5）

5. 添加Behavior Parameters组件：
   - Behavior Name: `gewu`（与前进训练相同，可复用模型）
   - Vector Observation Space Size: 41
   - Actions: Continuous Actions = 12

### 2. 复用前进训练的模型（推荐）

1. 将前进训练好的模型文件（.onnx）导入Unity
2. 在Behavior Parameters组件中选择该模型
3. 将`train`设置为`true`开始训练

这样可以实现**迁移学习**，从前进技能迁移到多动作控制。

### 3. 开始训练

1. 将`config_go2_control.yaml`文件放在项目根目录
2. 在命令行中运行：
```bash
mlagents-learn config_go2_control.yaml --run-id=go2_control_001
```

3. 在Unity中点击Play开始训练

### 4. 训练监控

训练过程中可以通过TensorBoard查看训练进度：
```bash
tensorboard --logdir=results
```

## 奖励函数设计

### 基础奖励
- **存活奖励**: 每步0.1
- **姿态奖励**: 惩罚倾斜（pitch和roll）
- **高度奖励**: 鼓励保持合适高度

### 速度奖励（根据动作模式）

#### 前进/后退
- 速度在最优区间内：`highRewardMultiplier * v1 * vel[2]`
- 速度在最优区间外：`lowRewardMultiplier * v1 * vel[2]`
- 惩罚侧向移动和旋转

#### 平移
- 速度在最优区间内：`highRewardMultiplier * v2 * vel[0]`
- 速度在最优区间外：`lowRewardMultiplier * v2 * vel[0]`
- 惩罚前进和旋转

#### 转向
- 角速度在最优区间内：`highRewardMultiplier * wr * wel[1]`
- 角速度在最优区间外：`lowRewardMultiplier * wr * wel[1]`
- 惩罚所有移动

## 训练技巧

1. **从前进模型开始**: 强烈建议从前进训练的模型开始，可以显著加速训练
2. **调整速度区间**: 根据实际训练效果调整`optimalSpeedMin/Max`参数
3. **平衡各组训练**: 确保6组agent都能得到充分训练
4. **监控各动作**: 通过TensorBoard分别监控不同动作的奖励曲线

## 注意事项

1. 确保Unity的Fixed Timestep设置为0.01
2. 训练时会自动创建48个机器人实例（包括原始agent）
3. Agent分组是根据名称自动解析的，确保命名格式正确
4. 如果某个动作训练效果不好，可以：
   - 调整该动作的速度参数
   - 调整速度区间范围
   - 增加该组agent的数量（需要修改代码）

## 测试训练好的模型

训练完成后，将训练好的模型（.onnx文件）导入Unity：
1. 将模型文件放在`Assets/ONNX/`目录
2. 在Behavior Parameters组件中选择该模型
3. 将`train`设置为`false`
4. 在代码中动态设置控制参数来测试不同动作：
   ```csharp
   agent.controlParam1 = 1f;  // 左平移
   agent.controlParam2 = 1f;  // 右平移
   // ... 等等
   ```

## 参考

- ML-Agents文档: https://github.com/Unity-Technologies/ml-agents
- 前进训练脚本: `Assets/Navigation/Scripts/Go2WalkAgent.cs`
- 站立训练脚本: `Assets/Navigation/Scripts/Go2StandAgent.cs`

