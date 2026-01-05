using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;
using System.Collections.Generic;

/// <summary>
/// Go2机器人多动作控制训练脚本
/// 训练目标：根据5个控制参数学习6种动作（前进、左平移、右平移、左转、右转、后退）
/// 复用前进训练的模型结构
/// </summary>
public class Go2ControlAgent : Agent
{
    int tp = 0;
    int tt = 0;
    public bool fixbody = false;
    public bool train = true;
    
    float uf1 = 0;
    float uf2 = 0;
    float[] u = new float[12] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float[] utotal = new float[12] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int T1 = 50;
    int tp0 = 0;
    
    Transform body;
    public int ObservationNum;
    public int ActionNum;

    List<float> P0 = new List<float>();
    List<float> W0 = new List<float>();
    Vector3 pos0;
    Quaternion rot0;
    ArticulationBody[] arts = new ArticulationBody[40];
    ArticulationBody[] acts = new ArticulationBody[12];

    // 关节控制参数
    float[] kb = new float[12] { 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30 };
    float dh = 25;
    float d0 = 15;
    
    // 控制参数（5维输入）
    float controlParam1 = 0f;  // 左平移
    float controlParam2 = 0f;  // 右平移
    float controlParam3 = 0f;  // 原地左转
    float controlParam4 = 0f;  // 原地右转
    float controlParam5 = 0f;  // 后退
    
    // 目标速度（根据控制参数计算）
    float v1 = 0;  // 前进速度（正值为前进，负值为后退）
    float v2 = 0;  // 侧向速度（正值为右，负值为左）
    float wr = 0;  // 角速度（正值为右转，负值为左转）
    
    // 动作模式
    public enum ActionMode
    {
        Forward = 0,      // 前进 [0,0,0,0,0]
        LeftStrafe = 1,   // 左平移 [1,0,0,0,0]
        RightStrafe = 2,  // 右平移 [0,1,0,0,0]
        LeftTurn = 3,     // 原地左转 [0,0,1,0,0]
        RightTurn = 4,    // 原地右转 [0,0,0,1,0]
        Backward = 5      // 后退 [0,0,0,0,1]
    }
    
    ActionMode currentActionMode = ActionMode.Forward;
    int agentGroupIndex = 0;  // Agent在组内的索引（0-7）
    
    // 奖励权重
    float ko = 2f;  // 姿态奖励权重
    float kv = 2f;  // 速度奖励权重
    
    // 训练参数
    [Header("训练参数")]
    public float maxTiltAngle = 20f;  // 最大倾斜角度，超过此角度结束episode
    public int maxEpisodeSteps = 2000;  // 最大episode步数
    public bool randomizeStartPose = true;  // 是否随机化初始姿态
    public float randomPoseRange = 5f;  // 随机姿态范围（度）
    [Header("速度参数")]
    public float forwardSpeed = 1.5f;  // 前进速度
    public float strafeSpeed = 1.0f;  // 平移速度
    public float backwardSpeed = 1.0f;  // 后退速度
    public float turnSpeed = 1.0f;  // 转向角速度（rad/s）
    [Header("速度区间奖励")]
    public float optimalSpeedMin = 0.8f;  // 最优速度区间最小值
    public float optimalSpeedMax = 1.8f;  // 最优速度区间最大值
    public float optimalTurnSpeedMin = 0.5f;  // 最优转向速度区间最小值
    public float optimalTurnSpeedMax = 1.5f;  // 最优转向速度区间最大值
    public float highRewardMultiplier = 2.0f;  // 最优速度区间奖励倍数
    public float lowRewardMultiplier = 0.5f;  // 非最优速度区间奖励倍数

    public override void Initialize()
    {
        arts = this.GetComponentsInChildren<ArticulationBody>();
        ActionNum = 0;
        for (int k = 0; k < arts.Length; k++)
        {
            if(arts[k].jointType.ToString() == "RevoluteJoint")
            {
                acts[ActionNum] = arts[k];
                print(acts[ActionNum]);
                ActionNum++;
            }
        }
        body = arts[0].GetComponent<Transform>();
        pos0 = body.position;
        rot0 = body.rotation;
        arts[0].GetJointPositions(P0);
        arts[0].GetJointVelocities(W0);
        
        // 根据agent名称确定组索引和动作模式
        ParseAgentGroupAndMode();
    }

    void ParseAgentGroupAndMode()
    {
        // 解析agent名称，确定组索引和动作模式
        // 命名格式：Go2ControlAgent 或 Go2ControlAgent_Clone_X，其中X是索引（1-47）
        string agentName = this.name;
        
        // 提取克隆索引（原始agent索引为0，Clone_1索引为1，以此类推）
        int cloneIndex = 0;
        if (agentName.Contains("Clone_"))
        {
            string indexStr = agentName.Substring(agentName.LastIndexOf("_") + 1);
            if (int.TryParse(indexStr, out int parsedIndex))
            {
                cloneIndex = parsedIndex;  // Clone_1 -> 1, Clone_2 -> 2, ...
            }
        }
        // 原始agent的cloneIndex保持为0
        
        // 48个agent（索引0-47），每8个一组，共6组对应6种动作
        // Group 0: 索引 0-7   -> 前进
        // Group 1: 索引 8-15  -> 左平移
        // Group 2: 索引 16-23 -> 右平移
        // Group 3: 索引 24-31 -> 原地左转
        // Group 4: 索引 32-39 -> 原地右转
        // Group 5: 索引 40-47 -> 后退
        int groupIndex = cloneIndex / 8;  // 0-5
        agentGroupIndex = cloneIndex % 8;  // 0-7
        
        // 确保groupIndex在有效范围内
        if (groupIndex < 0) groupIndex = 0;
        if (groupIndex > 5) groupIndex = 5;
        
        currentActionMode = (ActionMode)groupIndex;
        
        // 设置控制参数
        SetControlParamsFromMode(currentActionMode);
    }

    void SetControlParamsFromMode(ActionMode mode)
    {
        // 重置所有控制参数
        controlParam1 = 0f;
        controlParam2 = 0f;
        controlParam3 = 0f;
        controlParam4 = 0f;
        controlParam5 = 0f;
        
        // 根据模式设置对应的控制参数
        switch (mode)
        {
            case ActionMode.Forward:
                // [0,0,0,0,0] 前进
                v1 = forwardSpeed;
                v2 = 0f;
                wr = 0f;
                break;
            case ActionMode.LeftStrafe:
                // [1,0,0,0,0] 左平移
                controlParam1 = 1f;
                v1 = 0f;
                v2 = -strafeSpeed;
                wr = 0f;
                break;
            case ActionMode.RightStrafe:
                // [0,1,0,0,0] 右平移
                controlParam2 = 1f;
                v1 = 0f;
                v2 = strafeSpeed;
                wr = 0f;
                break;
            case ActionMode.LeftTurn:
                // [0,0,1,0,0] 原地左转
                controlParam3 = 1f;
                v1 = 0f;
                v2 = 0f;
                wr = -turnSpeed;
                break;
            case ActionMode.RightTurn:
                // [0,0,0,1,0] 原地右转
                controlParam4 = 1f;
                v1 = 0f;
                v2 = 0f;
                wr = turnSpeed;
                break;
            case ActionMode.Backward:
                // [0,0,0,0,1] 后退
                controlParam5 = 1f;
                v1 = -backwardSpeed;
                v2 = 0f;
                wr = 0f;
                break;
        }
    }

    private bool _isClone = false; 
    void Start()
    {
        Time.fixedDeltaTime = 0.01f;

        if (train && !_isClone) 
        {
            // 创建48个训练实例，每8个一组，共6组对应6种动作
            for (int i = 1; i < 48; i++)
            {
                GameObject clone = Instantiate(gameObject); 
                clone.name = $"{name}_Clone_{i}"; 
                clone.GetComponent<Go2ControlAgent>()._isClone = true; 
            }
        }
    }

    public override void OnEpisodeBegin()
    {
        tp = 0;
        tt = 0;
        for (int i = 0; i < 12; i++) u[i] = 0;

        ObservationNum = 17 + 2 * ActionNum;  // 3(重力) + 3(角速度) + 3(线速度) + 2*ActionNum(关节) + 3(目标速度) + 5(预留控制参数)
        
        // 重新解析组索引和动作模式（每次episode开始时）
        if (_isClone)
        {
            ParseAgentGroupAndMode();
        }
        else
        {
            currentActionMode = ActionMode.Forward;
            SetControlParamsFromMode(currentActionMode);
        }
        
        if (fixbody) 
        {
            arts[0].immovable = true;
        }
        else
        {
            // 重置机器人位置和姿态
            Vector3 resetPos = pos0;
            Quaternion resetRot = rot0;
            
            if (train && randomizeStartPose)
            {
                // 随机化初始位置（小范围）
                resetPos = pos0 + new Vector3(
                    Random.Range(-0.5f, 0.5f),
                    0,
                    Random.Range(-0.5f, 0.5f)
                );
                
                // 随机化初始姿态（小角度）
                resetRot = rot0 * Quaternion.Euler(
                    Random.Range(-randomPoseRange, randomPoseRange),
                    Random.Range(-10f, 10f),
                    Random.Range(-randomPoseRange, randomPoseRange)
                );
            }
            
            arts[0].TeleportRoot(resetPos, resetRot);
            arts[0].velocity = Vector3.zero;
            arts[0].angularVelocity = Vector3.zero;
            arts[0].SetJointPositions(P0);
            arts[0].SetJointVelocities(W0);
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 观察：重力方向（相对于机器人身体坐标系）
        sensor.AddObservation(body.InverseTransformDirection(Vector3.down));
        
        // 观察：角速度（相对于机器人身体坐标系）
        sensor.AddObservation(body.InverseTransformDirection(arts[0].angularVelocity));
        
        // 观察：线速度（相对于机器人身体坐标系）
        sensor.AddObservation(body.InverseTransformDirection(arts[0].velocity));
        
        // 观察：所有关节的位置和速度
        for (int i = 0; i < ActionNum; i++)
        {
            sensor.AddObservation(acts[i].jointPosition[0]);
            sensor.AddObservation(acts[i].jointVelocity[0]);
        }
        
        // 目标速度（根据动作模式设置）
        sensor.AddObservation(v1);  // v1 (前进速度)
        sensor.AddObservation(v2);  // v2 (侧向速度)
        sensor.AddObservation(wr);  // wr (角速度)
        
        // 控制参数（5维输入）
        sensor.AddObservation(controlParam1);  // 控制参数1 - 左平移
        sensor.AddObservation(controlParam2);  // 控制参数2 - 右平移
        sensor.AddObservation(controlParam3);  // 控制参数3 - 原地左转
        sensor.AddObservation(controlParam4);  // 控制参数4 - 原地右转
        sensor.AddObservation(controlParam5);  // 控制参数5 - 后退
    }
    
    float EulerTrans(float eulerAngle)
    {
        if (eulerAngle <= 180)
            return eulerAngle;
        else
            return eulerAngle - 360f;
    }
    
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        for (int i = 0; i < 12; i++) utotal[i] = 0;
        var continuousActions = actionBuffers.ContinuousActions;
        var kk = 0.9f;  // 动作平滑系数
        
        kb = new float[12] { 30, 30, 50, 30, 30, 50, 30, 30, 50, 30, 30, 50 };
        
        for (int i = 0; i < ActionNum; i++)
        {
            u[i] = u[i] * kk + (1 - kk) * continuousActions[i];
            utotal[i] = kb[i] * u[i];
            if (fixbody) utotal[i] = 0;
        }

        // 根据动作模式调整步态控制
        d0 = 30;
        dh = 20;
        T1 = 30;
        
        // 应用步态模式（对角步态）
        utotal[1] += dh * uf1 + d0;
        utotal[2] += (dh * uf1 + d0) * -2;
        utotal[4] += dh * uf2 + d0;
        utotal[5] += (dh * uf2 + d0) * -2;
        utotal[7] += dh * uf2 + d0;
        utotal[8] += (dh * uf2 + d0) * -2;
        utotal[10] += dh * uf1 + d0;
        utotal[11] += (dh * uf1 + d0) * -2;

        for (int i = 0; i < ActionNum; i++) SetJointTargetDeg(acts[i], utotal[i]);
    }
    
    void SetJointTargetDeg(ArticulationBody joint, float x)
    {
        var drive = joint.xDrive;
        drive.stiffness = 180f;
        drive.damping = 8f;
        drive.target = x;
        joint.xDrive = drive;
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // 手动控制模式（可选）
        var continuousActionsOut = actionsOut.ContinuousActions;
        for (int i = 0; i < ActionNum; i++)
        {
            continuousActionsOut[i] = 0f;
        }
    }

    void FixedUpdate()
    {
        tp++;
        tt++;
        
        // 步态周期（对角步态）
        if (tp > 0 && tp <= T1)
        {
            tp0 = tp;
            uf1 = (-Mathf.Cos(3.14f * 2 * tp0 / T1) + 1f) / 2f;
            uf2 = 0;
        }
        if (tp > T1 && tp <= 2 * T1)
        {
            tp0 = tp - T1;
            uf1 = 0;
            uf2 = (-Mathf.Cos(3.14f * 2 * tp0 / T1) + 1f) / 2f;
        }
        if (tp >= 2 * T1) tp = 0;
        
        // 计算奖励
        ko = 2f;
        kv = 2f;
        
        var vel = body.InverseTransformDirection(arts[0].velocity);
        var wel = body.InverseTransformDirection(arts[0].angularVelocity);
        
        // 基础存活奖励
        var live_reward = 0.1f;
        
        // 姿态奖励：惩罚倾斜（pitch和roll应该接近0）
        var pitch = EulerTrans(body.eulerAngles[0]);
        var roll = EulerTrans(body.eulerAngles[2]);
        var ori_reward1 = -0.1f * Mathf.Abs(pitch);
        var ori_reward2 = -0.1f * Mathf.Min(Mathf.Abs(roll), Mathf.Abs(roll - 360f));
        
        // 根据动作模式计算速度奖励
        float vel_reward = 0f;
        
        switch (currentActionMode)
        {
            case ActionMode.Forward:
            case ActionMode.Backward:
                // 前进/后退：使用前进速度区间奖励
                float forwardSpeed = Mathf.Abs(vel[2]);
                float forwardMatch = v1 * vel[2];  // 方向匹配
                
                if (forwardSpeed >= optimalSpeedMin && forwardSpeed <= optimalSpeedMax)
                {
                    vel_reward = highRewardMultiplier * forwardMatch;
                }
                else
                {
                    vel_reward = lowRewardMultiplier * forwardMatch;
                }
                
                // 惩罚侧向和旋转
                vel_reward += -1f * Mathf.Abs(vel[0] - v2);
                vel_reward += -0.3f * Mathf.Abs(wel[1] - wr);
                break;
                
            case ActionMode.LeftStrafe:
            case ActionMode.RightStrafe:
                // 平移：使用平移速度区间奖励
                float strafeSpeed = Mathf.Abs(vel[0]);
                float strafeMatch = v2 * vel[0];  // 方向匹配
                
                if (strafeSpeed >= optimalSpeedMin && strafeSpeed <= optimalSpeedMax)
                {
                    vel_reward = highRewardMultiplier * strafeMatch;
                }
                else
                {
                    vel_reward = lowRewardMultiplier * strafeMatch;
                }
                
                // 惩罚前进和旋转
                vel_reward += -1f * Mathf.Abs(vel[2] - v1);
                vel_reward += -0.3f * Mathf.Abs(wel[1] - wr);
                break;
                
            case ActionMode.LeftTurn:
            case ActionMode.RightTurn:
                // 转向：使用转向速度区间奖励
                float turnSpeed = Mathf.Abs(wel[1]);
                float turnMatch = wr * wel[1];  // 方向匹配
                
                if (turnSpeed >= optimalTurnSpeedMin && turnSpeed <= optimalTurnSpeedMax)
                {
                    vel_reward = highRewardMultiplier * turnMatch;
                }
                else
                {
                    vel_reward = lowRewardMultiplier * turnMatch;
                }
                
                // 惩罚移动
                vel_reward += -1f * Mathf.Abs(vel[2] - v1);
                vel_reward += -1f * Mathf.Abs(vel[0] - v2);
                break;
        }
        
        // 垂直速度惩罚：惩罚上下移动
        vel_reward += -0.5f * Mathf.Abs(vel[1]);
        
        // 高度奖励：鼓励保持合适的站立高度
        var height_reward = 0f;
        if (body.position.y > pos0.y - 0.1f && body.position.y < pos0.y + 0.2f)
        {
            height_reward = 0.05f;
        }
        else
        {
            height_reward = -0.1f * Mathf.Abs(body.position.y - pos0.y);
        }
        
        // 总奖励
        var reward = live_reward + 
                     (ori_reward1 + ori_reward2) * ko + 
                     vel_reward * kv +
                     height_reward;
        
        AddReward(reward);
        
        // 检查是否应该结束episode
        if (Mathf.Abs(pitch) > maxTiltAngle || 
            Mathf.Abs(roll) > maxTiltAngle || 
            tt >= maxEpisodeSteps ||
            body.position.y < pos0.y - 0.3f)  // 如果摔倒（高度过低）
        {
            if(train) EndEpisode();
        }
    }
}

