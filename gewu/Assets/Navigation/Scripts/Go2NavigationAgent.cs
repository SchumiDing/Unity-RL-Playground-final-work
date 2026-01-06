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
    
    // 全局时间窗口（用于同步所有agent的episode结束）
    private static float globalEpisodeStartTime = 0f;
    private static float globalEpisodeDuration = 60f;  // 全局episode持续时间（秒）
    private static bool globalEpisodeEnded = false;
    
    // 位置历史
    private Queue<Vector3> positionHistory = new Queue<Vector3>();
    private Queue<float> timeHistory = new Queue<float>();
    
    // 状态信息
    private Vector3 currentPosition;
    private Vector3 lastPosition;
    private float episodeStartTime;
    private float episodeTime;
    private bool hasReachedTarget = false;
    private int tt = 0;  // 步数计数器
    
    // 倒地检测参数
    private float fallenTime = 0f;  // 倒地持续时间
    private float fallenThreshold = 0.3f;  // 倒地判定时间阈值（秒）
    private float minMovementThreshold = 0.1f;  // 最小移动幅度阈值（米/秒）
    private float stillTime = 0f;  // 静止持续时间
    private float stillThreshold = 3f;  // 静止判定时间阈值（秒）
    
    // 动作模式映射（对应Go2ControlAgent的6种动作）
    private int[] actionToControlParams = new int[6] { 0, 1, 2, 3, 4, 5 };
    
    // 并行训练
    private bool _isClone = false;
    private Transform targetPointInstance;  // 克隆agent的目标点实例
    
    Transform body;
    ArticulationBody[] arts;
    Vector3 pos0;
    Quaternion rot0;
    
    public override void Initialize()
    {
        arts = this.GetComponentsInChildren<ArticulationBody>();
        body = arts[0].GetComponent<Transform>();
        pos0 = body.position;
        rot0 = body.rotation;
        
        // 检查是否有底层控制Agent
        if (controlAgent == null)
        {
            controlAgent = GetComponent<Go2ControlAgent>();
        }
        useControlAgent = (controlAgent != null);
        
        // 如果是克隆agent，创建独立的目标点
        if (_isClone && targetPoint != null)
        {
            GameObject targetObj = new GameObject($"TargetPoint_{name}");
            targetPointInstance = targetObj.transform;
            targetPointInstance.position = targetPoint.position;
        }
        else
        {
            targetPointInstance = targetPoint;
        }
        
        // 初始化位置历史
        for (int i = 0; i < historyLength; i++)
        {
            positionHistory.Enqueue(body.position);
            timeHistory.Enqueue(0f);
        }
        
        // 禁用ML-Agents观察空间大小不匹配的警告
        // 这是因为导航Agent和底层控制Agent有不同的观察空间大小
        // 使用日志回调来过滤这个特定的警告
        if (!_isClone)  // 只在原始agent上设置一次
        {
            Application.logMessageReceived += FilterObservationWarning;
        }
    }
    
    // 过滤观察空间大小警告
    void FilterObservationWarning(string logString, string stackTrace, LogType type)
    {
        // 如果是不匹配观察空间大小的警告，完全忽略
        if (type == LogType.Warning && 
            (logString.Contains("More observations") || logString.Contains("observations will be truncated")))
        {
            return;  // 不输出这个警告
        }
        
        // 其他日志正常输出（使用默认行为）
        // 注意：这里不能直接调用Debug.Log，否则会递归
        // Unity会自动处理其他日志
    }
    
    void OnDestroy()
    {
        // 清理日志回调（只在原始agent上清理）
        if (!_isClone)
        {
            Application.logMessageReceived -= FilterObservationWarning;
        }
    }
    
    void Start()
    {
        if (train && !_isClone)
        {
            // 初始化全局时间窗口
            globalEpisodeStartTime = Time.time;
            globalEpisodeDuration = maxEpisodeTime;
            globalEpisodeEnded = false;
            
            // 创建63个训练实例（总共64个agent）
            for (int i = 1; i < 64; i++)
            {
                GameObject clone = Instantiate(gameObject);
                clone.name = $"{name}_Clone_{i}";
                Go2NavigationAgent navAgent = clone.GetComponent<Go2NavigationAgent>();
                navAgent._isClone = true;
                
                // 为克隆agent创建独立的目标点
                if (targetPoint != null)
                {
                    GameObject targetObj = new GameObject($"TargetPoint_{clone.name}");
                    navAgent.targetPoint = targetObj.transform;
                    navAgent.targetPoint.position = targetPoint.position;
                }
            }
        }
    }
    
    public override void OnEpisodeBegin()
    {
        // 如果是原始agent且训练模式，重置全局时间窗口
        if (train && !_isClone)
        {
            globalEpisodeStartTime = Time.time;
            globalEpisodeEnded = false;
        }
        
        // 重置状态
        episodeStartTime = Time.time;
        episodeTime = 0f;
        hasReachedTarget = false;
        decisionTimer = 0f;
        currentActionChoice = 0;
        tt = 0;  // 重置步数计数器
        fallenTime = 0f;  // 重置倒地时间
        stillTime = 0f;  // 重置静止时间
        
        // 调试：确认episode开始
        if (train && !_isClone && tt == 0)
        {
            Debug.Log($"[导航训练] Episode开始 | maxEpisodeTime={maxEpisodeTime}s | 全局时间窗口={globalEpisodeDuration}s");
        }
        
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
            // 为每个agent设置不同的随机起始位置（避免重叠）
            int agentIndex = _isClone ? int.Parse(name.Substring(name.LastIndexOf("_") + 1)) : 0;
            float spacing = 3f;  // agent之间的间距
            float gridSize = Mathf.Ceil(Mathf.Sqrt(64));  // 8x8网格
            int row = agentIndex / (int)gridSize;
            int col = agentIndex % (int)gridSize;
            
            // 基础位置 + 网格偏移 + 随机扰动
            Vector3 baseOffset = new Vector3(
                (col - gridSize / 2) * spacing,
                0,
                (row - gridSize / 2) * spacing
            );
            
            startPoint = body.position + baseOffset + new Vector3(
                Random.Range(-1f, 1f),
                0,
                Random.Range(-1f, 1f)
            );
            
            // 重置机器人位置和姿态
            arts[0].TeleportRoot(startPoint, rot0);
            arts[0].velocity = Vector3.zero;
            arts[0].angularVelocity = Vector3.zero;
            
            // 更新初始位置记录
            pos0 = startPoint;
            
            // 随机目标位置（距离起始点一定范围）
            float distance = Random.Range(5f, 15f);
            float angle = Random.Range(0f, 360f) * Mathf.Deg2Rad;
            Vector3 targetPos = startPoint + new Vector3(
                Mathf.Cos(angle) * distance,
                0,
                Mathf.Sin(angle) * distance
            );
            
            // 使用目标点实例（克隆agent使用独立的目标点）
            if (targetPointInstance != null)
            {
                targetPointInstance.position = targetPos;
            }
            else if (targetPoint != null)
            {
                targetPoint.position = targetPos;
            }
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
        
        // 获取目标点位置（使用目标点实例或原始目标点）
        Vector3 targetPos = (targetPointInstance != null) ? targetPointInstance.position : 
                           (targetPoint != null ? targetPoint.position : Vector3.zero);
        
        // 1. 当前位置相对于目标的位置（4维：方向3维+距离1维）
        Vector3 toTarget = targetPos - currentPosition;
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
        
        // 总观察空间：4 + 30 + 10 + 3 + 2 + 1 = 50维
        // 注意：如果Behavior Parameters中设置的Space Size小于实际观察数，ML-Agents会截断并警告
        // 这是正常的，因为底层控制Agent（Go2ControlAgent）有41维观察，但导航Agent只需要自己的观察
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
        if (!useControlAgent || controlAgent == null)
        {
            Debug.LogWarning("[Go2NavigationAgent] controlAgent未设置或为空！");
            return;
        }
        
        // 根据动作索引设置控制参数并更新底层控制Agent
        // 0: 前进 [0,0,0,0,0]
        // 1: 左平移 [1,0,0,0,0]
        // 2: 右平移 [0,1,0,0,0]
        // 3: 原地左转 [0,0,1,0,0]
        // 4: 原地右转 [0,0,0,1,0]
        // 5: 后退 [0,0,0,0,1]
        
        Go2ControlAgent.ActionMode mode = (Go2ControlAgent.ActionMode)actionIndex;
        controlAgent.SetControlParamsFromMode(mode);
        
        // 调试信息
        if (train && tt % 200 == 0)  // 每200步打印一次
        {
            Debug.Log($"[导航训练] 选择动作: {actionIndex} ({mode}) | " +
                     $"控制参数: [{controlAgent.controlParam1}, {controlAgent.controlParam2}, " +
                     $"{controlAgent.controlParam3}, {controlAgent.controlParam4}, {controlAgent.controlParam5}]");
        }
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
        // 更新当前位置
        currentPosition = body.position;
        
        tt++;  // 增加步数
        episodeTime = Time.time - episodeStartTime;
        
        // 更新位置历史
        positionHistory.Dequeue();
        positionHistory.Enqueue(body.position);
        timeHistory.Dequeue();
        timeHistory.Enqueue(episodeTime);
        
        // 计算奖励（必须在检查结束条件之前调用，与其他训练脚本保持一致）
        CalculateRewards();
        
        // 检查episode结束条件（必须在AddReward之后调用，与其他训练脚本保持一致）
        // 参考Go2StandAgent和Go2WalkAgent的模式：AddReward -> 检查条件 -> EndEpisode
        bool shouldEnd = CheckEpisodeEnd();
        
        if (shouldEnd && train)
        {
            EndEpisode();
        }
        
        lastPosition = currentPosition;
    }
    
    void CalculateRewards()
    {
        float reward = 0f;
        
        // 获取目标点位置（使用目标点实例或原始目标点）
        Vector3 targetPos = (targetPointInstance != null) ? targetPointInstance.position : targetPoint.position;
        
        // 1. 到达目标奖励（大奖励）
        float distanceToTarget = Vector3.Distance(currentPosition, targetPos);
        if (distanceToTarget < targetReachDistance && !hasReachedTarget)
        {
            reward += 10f;  // 到达目标
            hasReachedTarget = true;
        }
        else
        {
            // 接近目标奖励（鼓励向目标移动）
            float lastDistance = Vector3.Distance(lastPosition, targetPos);
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
    
    bool CheckEpisodeEnd()
    {
        bool shouldEnd = false;
        string endReason = "";
        
        // 0. 检查全局时间窗口（优先检查，确保所有agent同时结束）
        if (train)
        {
            float globalElapsedTime = Time.time - globalEpisodeStartTime;
            if (globalElapsedTime >= globalEpisodeDuration && !globalEpisodeEnded)
            {
                shouldEnd = true;
                endReason = $"全局时间窗口超时（{globalEpisodeDuration}秒）";
                
                // 标记全局episode已结束（只标记一次）
                if (!_isClone)
                {
                    globalEpisodeEnded = true;
                    Debug.Log($"[全局时间窗口] 触发所有agent同时结束 | 全局时间={globalElapsedTime:F2}s");
                }
            }
        }
        
        // 1. 到达目标
        if (hasReachedTarget)
        {
            shouldEnd = true;
            endReason = "到达目标";
        }
        
        // 2. 超时（导航时间过长）- 作为备用检查
        // 使用>=确保超时一定会触发
        if (episodeTime >= maxEpisodeTime)
        {
            shouldEnd = true;
            endReason = $"超时（{maxEpisodeTime}秒）";
            
            // 调试：确认超时检测被触发
            if (train && !_isClone)
            {
                Debug.Log($"[超时检测] 触发超时 | episodeTime={episodeTime:F2}s | maxEpisodeTime={maxEpisodeTime}s");
            }
        }
        
        // 调试：每5秒输出一次episode状态（仅原始agent，训练模式）
        if (train && !_isClone && tt > 0 && tt % 500 == 0)  // FixedUpdate是0.01秒，500步=5秒
        {
            Vector3 targetPos = (targetPointInstance != null) ? targetPointInstance.position : targetPoint.position;
            float distanceToTarget = Vector3.Distance(currentPosition, targetPos);
            Debug.Log($"[调试] Episode状态 | 时间: {episodeTime:F1}s/{maxEpisodeTime}s | " +
                     $"步数: {tt} | 是否到达: {hasReachedTarget} | " +
                     $"距离目标: {distanceToTarget:F2}m | " +
                     $"位置: ({currentPosition.x:F1}, {currentPosition.y:F1}, {currentPosition.z:F1})");
        }
        
        // 3. 检查是否摔倒
        bool isFallen = IsFallen();
        if (isFallen)
        {
            fallenTime += Time.fixedDeltaTime;
            
            // 计算移动速度
            Vector3 velocity = (currentPosition - lastPosition) / Time.fixedDeltaTime;
            float speed = velocity.magnitude;
            
            // 如果倒地且移动幅度很小
            if (speed < minMovementThreshold)
            {
                stillTime += Time.fixedDeltaTime;
                
                // 如果倒地且长时间静止
                if (fallenTime >= fallenThreshold && stillTime >= stillThreshold)
                {
                    shouldEnd = true;
                    endReason = $"倒地且长时间静止（{fallenTime:F1}秒倒地，{stillTime:F1}秒静止）";
                }
            }
            else
            {
                // 如果还在移动，重置静止时间
                stillTime = 0f;
            }
        }
        else
        {
            // 如果没有摔倒，重置倒地时间和静止时间
            fallenTime = 0f;
            stillTime = 0f;
        }
        
        // 4. 姿态过度倾斜（立即结束）
        float pitch = EulerTrans(body.eulerAngles.x);
        float roll = EulerTrans(body.eulerAngles.z);
        if (Mathf.Abs(pitch) > 60f || Mathf.Abs(roll) > 60f)
        {
            shouldEnd = true;
            endReason = $"姿态过度倾斜（Pitch={pitch:F1}°, Roll={roll:F1}°）";
        }
        
        // 如果应该结束，输出日志（但不在这里调用EndEpisode，由FixedUpdate调用）
        if (shouldEnd && train && !string.IsNullOrEmpty(endReason) && !_isClone)
        {
            Vector3 targetPos = (targetPointInstance != null) ? targetPointInstance.position : targetPoint.position;
            Debug.Log($"[导航训练] Episode结束 | 原因: {endReason} | " +
                     $"时间: {episodeTime:F1}s | " +
                     $"步数: {tt} | " +
                     $"距离目标: {Vector3.Distance(currentPosition, targetPos):F2}m");
        }
        
        return shouldEnd;
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

