using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;
using System.Collections.Generic;

/// <summary>
/// Go2机器人自主站立训练脚本
/// 训练目标：让机器人保持稳定站立，不移动，保持平衡
/// </summary>
public class Go2StandAgent : Agent
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
    float d0 = 15;
    
    // 奖励权重
    float ko = 2f;  // 姿态奖励权重
    float kv = 2f;  // 速度奖励权重
    
    // 训练参数
    [Header("训练参数")]
    public float maxTiltAngle = 20f;  // 最大倾斜角度，超过此角度结束episode
    public int maxEpisodeSteps = 2000;  // 最大episode步数
    public bool randomizeStartPose = true;  // 是否随机化初始姿态
    public float randomPoseRange = 5f;  // 随机姿态范围（度）

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
    }

    private bool _isClone = false; 
    void Start()
    {
        Time.fixedDeltaTime = 0.01f;

        if (train && !_isClone) 
        {
            // 创建多个训练实例以加速训练
            for (int i = 1; i < 24; i++)
            {
                GameObject clone = Instantiate(gameObject); 
                clone.name = $"{name}_Clone_{i}"; 
                clone.GetComponent<Go2StandAgent>()._isClone = true; 
            }
        }
    }

    public override void OnEpisodeBegin()
    {
        tp = 0;
        tt = 0;
        for (int i = 0; i < 12; i++) u[i] = 0;

        ObservationNum = 17 + 2 * ActionNum;  // 3(重力) + 3(角速度) + 3(线速度) + 2*ActionNum(关节) + 3(目标速度) + 5(预留控制参数)
        
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
        
        // 目标速度（站立时应该为0）
        sensor.AddObservation(0f);  // v1 = 0 (前进速度)
        sensor.AddObservation(0f);  // v2 = 0 (侧向速度)
        sensor.AddObservation(0f);  // wr = 0 (角速度)
        
        // 预留控制参数位置（5维，用于后续训练，站立训练时置为0）
        sensor.AddObservation(0f);  // 控制参数1
        sensor.AddObservation(0f);  // 控制参数2
        sensor.AddObservation(0f);  // 控制参数3
        sensor.AddObservation(0f);  // 控制参数4
        sensor.AddObservation(0f);  // 控制参数5
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

        // 站立时的基础姿态控制（保持腿部轻微弯曲以维持平衡）
        d0 = 30;
        T1 = 30;
        
        // 应用基础站立姿态（所有腿保持相同姿态）
        utotal[1] += d0;
        utotal[2] += d0 * -2;
        utotal[4] += d0;
        utotal[5] += d0 * -2;
        utotal[7] += d0;
        utotal[8] += d0 * -2;
        utotal[10] += d0;
        utotal[11] += d0 * -2;

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
        // 站立时所有动作应该接近0
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
        
        // 步态周期（站立时不需要步态，但保留以保持兼容性）
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
        var ori_reward1 = -0.2f * Mathf.Abs(pitch);
        var ori_reward2 = -0.2f * Mathf.Abs(roll);
        
        // 角速度奖励：惩罚旋转（应该保持稳定）
        var wel_reward = -0.5f * (Mathf.Abs(wel[0]) + Mathf.Abs(wel[1]) + Mathf.Abs(wel[2]));
        
        // 线速度奖励：惩罚移动（站立时应该保持静止）
        var vel_reward = -0.3f * (Mathf.Abs(vel[0]) + Mathf.Abs(vel[2])) - 0.5f * Mathf.Abs(vel[1]);
        
        // 高度奖励：鼓励保持站立高度
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
                     wel_reward * kv + 
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

