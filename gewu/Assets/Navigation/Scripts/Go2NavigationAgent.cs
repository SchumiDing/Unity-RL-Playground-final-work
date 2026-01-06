using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;
using System.Collections.Generic;

/// <summary>
/// Go2机器人自动导航Agent
/// 高层决策网络：根据位置历史和周围环境，决定执行哪个控制动作
/// 底层控制网络：使用Go2ControlAgent训练好的模型执行具体动作
/// </summary>
public class Go2NavigationAgent : Agent
{
    [Header("导航参数")]
    public Transform targetPoint;  // 目标点
    public Vector3 startPoint;  // 起始点
    public float targetReachDistance = 1.0f;  // 到达目标的距离阈值
    public float maxEpisodeTime = 60f;  // 最大episode时间（秒）
    
    [Header("感知参数")]
    public int historyLength = 10;  // 位置历史长度
    public int nearbyObjectsCount = 10;  // 检测周围物体数量
    public float detectionRadius = 10f;  // 检测半径
    public LayerMask obstacleLayer = -1;  // 障碍物图层
    
    [Header("决策参数")]
    public float decisionInterval = 0.5f;  // 决策间隔（秒）
    private float decisionTimer = 0f;
    private int currentActionChoice = 0;  // 当前选择的动作（0-5）
    
    [Header("底层控制")]
    public Go2ControlAgent controlAgent;  // 底层控制Agent组件
    private bool useControlAgent = false;
    
    [Header("训练参数")]
    public bool train = true;  // 是否训练模式
    
    // 位置历史
    private Queue<Vector3> positionHistory = new Queue<Vector3>();
    private Queue<float> timeHistory = new Queue<float>();
    
    // 状态信息
    private Vector3 currentPosition;
    private Vector3 lastPosition;
    private float episodeStartTime;
    private float episodeTime;
    private bool hasReachedTarget = false;
    
    // 动作模式映射（对应Go2ControlAgent的6种动作）
    private int[] actionToControlParams = new int[6] { 0, 1, 2, 3, 4, 5 };
    
    Transform body;
    ArticulationBody[] arts;
    
    public override void Initialize()
    {
        arts = this.GetComponentsInChildren<ArticulationBody>();
        body = arts[0].GetComponent<Transform>();
        
        // 检查是否有底层控制Agent
        if (controlAgent == null)
        {
            controlAgent = GetComponent<Go2ControlAgent>();
        }
        useControlAgent = (controlAgent != null);
        
        // 初始化位置历史
        for (int i = 0; i < historyLength; i++)
        {
            positionHistory.Enqueue(body.position);
            timeHistory.Enqueue(0f);
        }
    }
    
    public override void OnEpisodeBegin()
    {
        // 重置状态
        episodeStartTime = Time.time;
        episodeTime = 0f;
        hasReachedTarget = false;
        decisionTimer = 0f;
        currentActionChoice = 0;
        
        // 重置位置历史
        positionHistory.Clear();
        timeHistory.Clear();
        for (int i = 0; i < historyLength; i++)
        {
            positionHistory.Enqueue(body.position);
            timeHistory.Enqueue(0f);
        }
        
        // 设置起始位置和目标位置
        if (train)
        {
            // 随机起始位置
            startPoint = body.position + new Vector3(
                Random.Range(-5f, 5f),
                0,
                Random.Range(-5f, 5f)
            );
            
            // 随机目标位置（距离起始点一定范围）
            float distance = Random.Range(5f, 15f);
            float angle = Random.Range(0f, 360f) * Mathf.Deg2Rad;
            targetPoint.position = startPoint + new Vector3(
                Mathf.Cos(angle) * distance,
                0,
                Mathf.Sin(angle) * distance
            );
        }
        
        // 重置机器人位置
        body.position = startPoint;
        body.rotation = Quaternion.identity;
        if (arts[0] != null)
        {
            arts[0].velocity = Vector3.zero;
            arts[0].angularVelocity = Vector3.zero;
        }
        
        lastPosition = startPoint;
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        currentPosition = body.position;
        
        // 1. 当前位置相对于目标的位置（4维：方向3维+距离1维）
        Vector3 toTarget = targetPoint.position - currentPosition;
        float distanceToTarget = toTarget.magnitude;
        if (distanceToTarget > 0.001f)
        {
            sensor.AddObservation(toTarget.normalized);  // 方向（归一化，3维）
        }
        else
        {
            sensor.AddObservation(Vector3.zero);  // 已到达目标
        }
        sensor.AddObservation(distanceToTarget / 20f);   // 距离（归一化，假设最大距离20米）
        
        // 2. 位置历史（10个时间点，每个3维，共30维）
        foreach (Vector3 pos in positionHistory)
        {
            Vector3 relativePos = pos - currentPosition;
            sensor.AddObservation(relativePos);
        }
        
        // 3. 周围物体距离（10个，共10维）
        List<float> obstacleDistances = GetNearbyObstacleDistances();
        for (int i = 0; i < nearbyObjectsCount; i++)
        {
            if (i < obstacleDistances.Count)
            {
                sensor.AddObservation(obstacleDistances[i]);
            }
            else
            {
                sensor.AddObservation(detectionRadius);  // 没有检测到物体，距离为最大检测半径
            }
        }
        
        // 4. 当前速度（3维）
        Vector3 velocity = (currentPosition - lastPosition) / Time.fixedDeltaTime;
        sensor.AddObservation(velocity);
        
        // 5. 当前姿态（2维：pitch和roll）
        float pitch = EulerTrans(body.eulerAngles.x);
        float roll = EulerTrans(body.eulerAngles.z);
        sensor.AddObservation(pitch / 90f);  // 归一化到[-1, 1]
        sensor.AddObservation(roll / 90f);
        
        // 6. 时间信息（1维：episode剩余时间比例）
        float timeRatio = 1f - (episodeTime / maxEpisodeTime);
        sensor.AddObservation(timeRatio);
        
        // 总观察空间：3 + 30 + 10 + 3 + 2 + 1 = 49维
    }
    
    float EulerTrans(float eulerAngle)
    {
        if (eulerAngle <= 180)
            return eulerAngle;
        else
            return eulerAngle - 360f;
    }
    
    List<float> GetNearbyObstacleDistances()
    {
        List<float> distances = new List<float>();
        
        // 使用Raycast检测周围物体
        int raysPerLayer = nearbyObjectsCount;
        float angleStep = 360f / raysPerLayer;
        
        for (int i = 0; i < raysPerLayer; i++)
        {
            float angle = i * angleStep * Mathf.Deg2Rad;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            direction = body.TransformDirection(direction);
            
            RaycastHit hit;
            if (Physics.Raycast(body.position, direction, out hit, detectionRadius, obstacleLayer))
            {
                distances.Add(hit.distance);
            }
            else
            {
                distances.Add(detectionRadius);
            }
        }
        
        // 按距离排序，取最近的10个
        distances.Sort();
        if (distances.Count > nearbyObjectsCount)
        {
            distances = distances.GetRange(0, nearbyObjectsCount);
        }
        
        return distances;
    }
    
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // 决策网络输出6个动作的选择（离散动作）
        var discreteActions = actionBuffers.DiscreteActions;
        currentActionChoice = discreteActions[0];  // 动作选择（0-5）
        
        // 确保动作在有效范围内
        if (currentActionChoice < 0) currentActionChoice = 0;
        if (currentActionChoice > 5) currentActionChoice = 5;
        
        // 每隔一定时间更新动作（避免动作切换过于频繁）
        decisionTimer += Time.fixedDeltaTime;
        if (decisionTimer >= decisionInterval)
        {
            decisionTimer = 0f;
            // 将动作选择传递给底层控制Agent
            if (useControlAgent)
            {
                ApplyControlAction(currentActionChoice);
            }
        }
    }
    
    void ApplyControlAction(int actionIndex)
    {
        if (!useControlAgent || controlAgent == null) return;
        
        // 根据动作索引设置控制参数并更新底层控制Agent
        // 0: 前进 [0,0,0,0,0]
        // 1: 左平移 [1,0,0,0,0]
        // 2: 右平移 [0,1,0,0,0]
        // 3: 原地左转 [0,0,1,0,0]
        // 4: 原地右转 [0,0,0,1,0]
        // 5: 后退 [0,0,0,0,1]
        
        Go2ControlAgent.ActionMode mode = (Go2ControlAgent.ActionMode)actionIndex;
        controlAgent.SetControlParamsFromMode(mode);
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // 手动控制：使用键盘选择动作
        var discreteActionsOut = actionsOut.DiscreteActions;
        
        if (Input.GetKey(KeyCode.W)) discreteActionsOut[0] = 0;  // 前进
        else if (Input.GetKey(KeyCode.A)) discreteActionsOut[0] = 1;  // 左平移
        else if (Input.GetKey(KeyCode.D)) discreteActionsOut[0] = 2;  // 右平移
        else if (Input.GetKey(KeyCode.Q)) discreteActionsOut[0] = 3;  // 左转
        else if (Input.GetKey(KeyCode.E)) discreteActionsOut[0] = 4;  // 右转
        else if (Input.GetKey(KeyCode.S)) discreteActionsOut[0] = 5;  // 后退
        else discreteActionsOut[0] = 0;  // 默认前进
    }
    
    void FixedUpdate()
    {
        episodeTime = Time.time - episodeStartTime;
        
        // 更新位置历史
        positionHistory.Dequeue();
        positionHistory.Enqueue(body.position);
        timeHistory.Dequeue();
        timeHistory.Enqueue(episodeTime);
        
        // 计算奖励
        CalculateRewards();
        
        // 检查episode结束条件
        CheckEpisodeEnd();
        
        lastPosition = currentPosition;
    }
    
    void CalculateRewards()
    {
        float reward = 0f;
        
        // 1. 到达目标奖励（大奖励）
        float distanceToTarget = Vector3.Distance(currentPosition, targetPoint.position);
        if (distanceToTarget < targetReachDistance && !hasReachedTarget)
        {
            reward += 10f;  // 到达目标
            hasReachedTarget = true;
        }
        else
        {
            // 接近目标奖励（鼓励向目标移动）
            float lastDistance = Vector3.Distance(lastPosition, targetPoint.position);
            float distanceImprovement = lastDistance - distanceToTarget;
            reward += distanceImprovement * 0.5f;  // 每接近1米奖励0.5
        }
        
        // 2. 碰撞惩罚
        if (HasCollision())
        {
            reward -= 2f;  // 碰撞惩罚
        }
        
        // 3. 速度奖励（鼓励移动）
        Vector3 velocity = (currentPosition - lastPosition) / Time.fixedDeltaTime;
        float speed = velocity.magnitude;
        if (speed > 0.1f && !hasReachedTarget)
        {
            reward += 0.01f * speed;  // 速度奖励
        }
        
        // 4. 时间惩罚（鼓励快速到达）
        if (!hasReachedTarget)
        {
            reward -= 0.01f;  // 每步小惩罚，鼓励快速完成
        }
        
        // 5. 姿态奖励（保持平衡）
        float pitch = EulerTrans(body.eulerAngles.x);
        float roll = EulerTrans(body.eulerAngles.z);
        if (Mathf.Abs(pitch) < 10f && Mathf.Abs(roll) < 10f)
        {
            reward += 0.05f;  // 保持平衡奖励
        }
        else
        {
            reward -= 0.1f * (Mathf.Abs(pitch) + Mathf.Abs(roll)) / 90f;  // 倾斜惩罚
        }
        
        // 6. 摔倒惩罚
        if (IsFallen())
        {
            reward -= 5f;  // 摔倒大惩罚
        }
        
        AddReward(reward);
    }
    
    bool HasCollision()
    {
        // 检查是否与障碍物碰撞
        Collider[] colliders = Physics.OverlapSphere(body.position, 0.5f, obstacleLayer);
        return colliders.Length > 0;
    }
    
    bool IsFallen()
    {
        // 检查是否摔倒：高度过低或姿态过度倾斜
        bool heightTooLow = body.position.y < startPoint.y - 0.3f;
        float pitch = EulerTrans(body.eulerAngles.x);
        float roll = EulerTrans(body.eulerAngles.z);
        bool tiltTooMuch = Mathf.Abs(pitch) > 45f || Mathf.Abs(roll) > 45f;
        
        return heightTooLow || tiltTooMuch;
    }
    
    void CheckEpisodeEnd()
    {
        bool shouldEnd = false;
        
        // 1. 到达目标
        if (hasReachedTarget)
        {
            shouldEnd = true;
        }
        
        // 2. 超时
        if (episodeTime >= maxEpisodeTime)
        {
            shouldEnd = true;
        }
        
        // 3. 摔倒（高度过低）
        if (body.position.y < startPoint.y - 0.5f)
        {
            shouldEnd = true;
        }
        
        // 4. 姿态过度倾斜
        float pitch = EulerTrans(body.eulerAngles.x);
        float roll = EulerTrans(body.eulerAngles.z);
        if (Mathf.Abs(pitch) > 45f || Mathf.Abs(roll) > 45f)
        {
            shouldEnd = true;
        }
        
        if (shouldEnd && train)
        {
            EndEpisode();
        }
    }
    
    // 公共方法：供外部设置目标点
    public void SetTarget(Vector3 target)
    {
        if (targetPoint != null)
        {
            targetPoint.position = target;
        }
    }
    
    // 公共方法：供外部设置起始点
    public void SetStartPoint(Vector3 start)
    {
        startPoint = start;
    }
}

