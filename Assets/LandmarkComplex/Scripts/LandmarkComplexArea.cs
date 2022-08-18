using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgentsExamples;
using Unity.MLAgents;

using System;
using UnityEngine.Networking;
using System.Linq;

using Random = UnityEngine.Random;

public class LandmarkComplexArea : Area
{
    public HashSet<int> seenLandmarks = new HashSet<int>();
    public int nodeCount = 0;

    [Header("Max Environment Steps")] public int MaxEnvironmentSteps = 50000;
    private int m_ResetTimer;

    public bool demonstrateDestroyedLandmark;
    public Material demonstrateDestroyedLandmarkMaterial;

    public GameObject landmark;
    public GameObject empty2Simplex;
    public LayerMask landmarkLayerMask;


    public bool spawnRandomObstacles;
    public Material obstacleMaterial;
    public int numberOfRandomObstacles = 10;
    public LayerMask obstacleLayerMask;

    public bool randomlyDestroyLandmarks;
    public float landmarkDestroyPossibility = 0.3f;
    
    public int minObstacleLength = 50;
    public int minObstacleWidth = 20; 

    public int maxObstacleLength = 100;
    public int maxObstacleWidth = 50; 

    public float horizontalRange = 239;
    public float verticalRange = 140.5f;

    // In radius
    public List<float> sensorFootprints;

    [HideInInspector]
    public List<GameObject> landmarks;

    public List<GameObject> generatedObstacles;

    public HashSet<Vector3> triangles = new HashSet<Vector3>();
    public HashSet<Vector2> edges = new HashSet<Vector2>();
    public HashSet<int> nodes = new HashSet<int>();

    public int totalNumberOfNodes;



    private SimpleMultiAgentGroup m_AgentGroup;

    public List<GameObject> agents;

    private List<LandmarkComplexAgent> agentsList = new List<LandmarkComplexAgent>();

    private int currentObstacleNumberRound = 0;

    private int currentLandmarkDestroyRound = 0;

    private bool allLandmarkCreated = false;

    private bool randomDestroyedRoundCountFixer = true;


    //public float previous_cumulative_reward = 10000f;

    void Start()
    {
        m_AgentGroup = new SimpleMultiAgentGroup();

        foreach (var agent in agents)
        {
            LandmarkComplexAgent currentAgent = agent.GetComponent<LandmarkComplexAgent>();
            agentsList.Add(currentAgent);
            m_AgentGroup.RegisterAgent(currentAgent);
        }
    }

    void FixedUpdate()
    {
        
        m_AgentGroup.AddGroupReward(-0.2f);
        if (nodes.Count == 0.93 * totalNumberOfNodes)
        {
            m_AgentGroup.AddGroupReward(5000);
            m_AgentGroup.EndGroupEpisode();
            ResetLandmarkComplexArea();
        }

        m_ResetTimer += 1;

        if ((randomDestroyedRoundCountFixer) && (randomlyDestroyLandmarks))
        {
            if(m_ResetTimer >= 5)
            {
                /*
                float current_cum_reward = GetCumulativeReward();
                if ((current_cum_reward/previous_cumulative_reward > 1.5f) || (previous_cumulative_reward/current_cum_reward > 1.5f))
                {
                    float new_current_cum_reward = previous_cumulative_reward + (current_cum_reward - previous_cumulative_reward) * 0.3f;
                    SetReward(new_current_cum_reward);
                    previous_cumulative_reward = new_current_cum_reward;
                }*/


                m_AgentGroup.GroupEpisodeInterrupted();
                ResetLandmarkComplexArea();
            }
        }
        else
        {
            if (m_ResetTimer >= MaxEnvironmentSteps && MaxEnvironmentSteps > 0)
            {
                /*
                float current_cum_reward = GetCumulativeReward();
                if ((current_cum_reward/previous_cumulative_reward > 1.5f) || (previous_cumulative_reward/current_cum_reward > 1.5f))
                {
                    float new_current_cum_reward = previous_cumulative_reward + (current_cum_reward - previous_cumulative_reward) * 0.3f;
                    SetReward(new_current_cum_reward);
                    previous_cumulative_reward = new_current_cum_reward;
                }*/

                m_AgentGroup.GroupEpisodeInterrupted();
                ResetLandmarkComplexArea();
            }
        }

        
    }


    void ClearObjects(List<GameObject> objects)
    {
        foreach (var landmark in objects)
        {
            Destroy(landmark);
        }
    }

    void ClearObjectsArray(GameObject[] objects)
    {
        foreach (var landmark in objects)
        {
            Destroy(landmark);
        }
    }


    void CreateRandomObstacles(int num)
    {
        for (int o = 0; o < num; o++)
        {
            Vector3 obstaclePosition = new Vector3();
            Vector3 obstacleScale = new Vector3(); 
            
            bool overlappedWithOtherGameObject = true;
            while (overlappedWithOtherGameObject)
            {
                obstaclePosition = new Vector3(Random.Range(-horizontalRange, horizontalRange), 2, Random.Range(-verticalRange, verticalRange)) + transform.position;
                obstacleScale = new Vector3(Random.Range(minObstacleLength, maxObstacleLength), 3, Random.Range(minObstacleWidth, maxObstacleWidth));

                Collider[] hitColliders = Physics.OverlapBox(obstaclePosition, obstacleScale / 2, Quaternion.identity, obstacleLayerMask);

                if (hitColliders.Length == 0)
                {
                    overlappedWithOtherGameObject = false;
                }
            }

            GameObject obstacle = GameObject.CreatePrimitive(PrimitiveType.Cube);
            generatedObstacles.Add(obstacle);
            obstacle.transform.position = obstaclePosition;
            obstacle.transform.localScale = obstacleScale;
            obstacle.GetComponent<Renderer>().material = obstacleMaterial;
            obstacle.tag = "wall";
            obstacle.layer = LayerMask.NameToLayer("obstacle");
        }
    }

    // LPA
    void CreateLandmark(GameObject type)
    {
        int currentLandmarkID = 0;
        for(int i = 0; i < sensorFootprints.Count; i++)
        {
            
            for (float j = -horizontalRange; j < horizontalRange; )
            {
                for (float k = -verticalRange; k < verticalRange; )
                {
                    // First landmark spawn at center
                    
                    if ((i == 0) && (j == -horizontalRange) && (k == -verticalRange))
                    {
                        Vector3 firstSpawnPoint = new Vector3(0, 1f, 0) + transform.position;
                        bool shouldSpawn = CheckIfShouldSpawnLandmark(firstSpawnPoint, sensorFootprints[i]); 
                        if (shouldSpawn)
                        {
                            GameObject l = Instantiate(type, firstSpawnPoint, Quaternion.identity);
                            landmarks.Add(l);
                            l.GetComponent<LandmarkLogic>().myArea = this;
                            l.GetComponent<LandmarkLogic>().landmarkID = currentLandmarkID;
                            currentLandmarkID += 1;
                        }
                    }

                    Vector3 regularSpawnPoint = new Vector3(j, 1f, k) + transform.position;

                    bool regularShouldSpawn = CheckIfShouldSpawnLandmark(regularSpawnPoint, sensorFootprints[i]); 
                    if (regularShouldSpawn)
                    {
                        //GameObject lr = Instantiate(type, regularSpawnPoint, Quaternion.identity);

                        if (randomlyDestroyLandmarks)
                        {
                            if(Random.value > landmarkDestroyPossibility)
                            {
                                GameObject lr = Instantiate(type, regularSpawnPoint, Quaternion.identity);
                                landmarks.Add(lr);
                                lr.GetComponent<LandmarkLogic>().myArea = this;
                                lr.GetComponent<LandmarkLogic>().landmarkID = currentLandmarkID;
                                currentLandmarkID += 1;
                            }
                            else
                            {
                                if(demonstrateDestroyedLandmark)
                                {
                                    GameObject lr = Instantiate(type, regularSpawnPoint, Quaternion.identity);
                                    landmarks.Add(lr);
                                    lr.GetComponent<Renderer>().material = demonstrateDestroyedLandmarkMaterial;
                                    lr.GetComponent<LandmarkLogic>().myArea = this;
                                    lr.GetComponent<LandmarkLogic>().landmarkID = currentLandmarkID;
                                    currentLandmarkID += 1;
                                }
                            }
                        }
                        else
                        {
                            GameObject lr = Instantiate(type, regularSpawnPoint, Quaternion.identity);
                            landmarks.Add(lr);
                            lr.GetComponent<LandmarkLogic>().myArea = this;
                            lr.GetComponent<LandmarkLogic>().landmarkID = currentLandmarkID;
                            currentLandmarkID += 1;
                        }

                        
                        //lr.GetComponent<LandmarkLogic>().myArea = this;
                        //lr.GetComponent<LandmarkLogic>().landmarkID = currentLandmarkID;
                        //currentLandmarkID += 1;
                    }

                    k += sensorFootprints[i];
                }
                j += sensorFootprints[i];
            }
            
        }
        foreach(GameObject currentLandmark in landmarks)
        {
            foreach(GameObject testLandmark in landmarks)
            {
                if ((currentLandmark != testLandmark) && (currentLandmark.transform.position == testLandmark.transform.position))
                {
                    Destroy(currentLandmark);
                }
            }
        }

        allLandmarkCreated = true;
    }

    private bool CheckIfShouldSpawnLandmark(Vector3 spawnPoint, float sensorFootprint)
    {
        // y axis of spawn point should be 1
        bool ifShouldSpawnLandmark = false;

        spawnPoint = spawnPoint - new Vector3(0, 0.5f, 0);

        // Check if the point is occupied first
        Collider[] occupied = Physics.OverlapSphere(spawnPoint, 0.5f, landmarkLayerMask);

        if (occupied.Length > 0)
        {
            ifShouldSpawnLandmark = false;
        }
        else
        {
            Collider[] hitColliders = Physics.OverlapSphere(spawnPoint, sensorFootprint, landmarkLayerMask);
        
            if (hitColliders.Length > 0)
            {
                ifShouldSpawnLandmark = true;
                foreach(Collider hc in hitColliders)
                {
                    if (hc.tag == "undetectedLandmark")
                    {
                        RaycastHit hit;
                        if (Physics.Linecast(spawnPoint, hc.transform.position, out hit))
                        {
                            if (hit.transform.tag == "wall")
                            {
                                ifShouldSpawnLandmark = true;
                            }
                        }
                        else
                        {
                            ifShouldSpawnLandmark = false;
                        }
                    }
                }
            }
            else
            {
                ifShouldSpawnLandmark = true;
            }
        }
        

        return ifShouldSpawnLandmark;
    }

    public void ResetLandmarkComplexArea()
    {
        m_ResetTimer = 0;

        agents[0].transform.GetChild(3).gameObject.GetComponent<Detector>().detectionFreeze = true;
        agents[1].transform.GetChild(3).gameObject.GetComponent<Detector>().detectionFreeze = true;
        agents[2].transform.GetChild(3).gameObject.GetComponent<Detector>().detectionFreeze = true;
        agents[3].transform.GetChild(3).gameObject.GetComponent<Detector>().detectionFreeze = true;

        agents[0].transform.GetChild(3).gameObject.GetComponent<Detector>().currentObservedLandmarkIDs.Clear();
        agents[1].transform.GetChild(3).gameObject.GetComponent<Detector>().currentObservedLandmarkIDs.Clear();
        agents[2].transform.GetChild(3).gameObject.GetComponent<Detector>().currentObservedLandmarkIDs.Clear();
        agents[3].transform.GetChild(3).gameObject.GetComponent<Detector>().currentObservedLandmarkIDs.Clear();

        ClearObjectsArray(GameObject.FindGameObjectsWithTag("2Simplex"));
        nodes.Clear();
        edges.Clear();
        triangles.Clear();

        ClearObjects(landmarks);
        landmarks.Clear();

        if (spawnRandomObstacles)
        {
            ClearObjects(generatedObstacles);
            generatedObstacles.Clear();
        }

        agents[0].transform.position = new Vector3(-horizontalRange - 6, 2f, -verticalRange) + transform.position;
        agents[0].transform.rotation = Quaternion.Euler(new Vector3(0f, 45f, 0f));
        

        agents[1].transform.position = new Vector3(horizontalRange - 6, 2f, -verticalRange) + transform.position;
        agents[1].transform.rotation = Quaternion.Euler(new Vector3(0f, -45f, 0f));

        agents[2].transform.position = new Vector3(horizontalRange - 6, 2f, verticalRange) + transform.position;
        agents[2].transform.rotation = Quaternion.Euler(new Vector3(0f, 255f, 0f));

        agents[3].transform.position = new Vector3(-horizontalRange - 6, 2f, verticalRange) + transform.position;
        agents[3].transform.rotation = Quaternion.Euler(new Vector3(0f, 120f, 0f));

        foreach (var agent in agentsList)
        {
            m_AgentGroup.RegisterAgent(agent);
        }

        if (spawnRandomObstacles)
        {
            CreateRandomObstacles(numberOfRandomObstacles);

            if (!randomDestroyedRoundCountFixer)
            {
                currentObstacleNumberRound += 1;

                if (randomlyDestroyLandmarks)
                {

                    currentLandmarkDestroyRound += 1;
                    if (currentLandmarkDestroyRound == 5)
                    {
                        landmarkDestroyPossibility += 0.05f;
                        currentLandmarkDestroyRound = 0;

                    }
                }


                if (currentObstacleNumberRound == 25)
                {
                    numberOfRandomObstacles += 1;
                    currentObstacleNumberRound = 0;

                    landmarkDestroyPossibility = 0.05f;

                }
            }

            

            // numberOfRandomObstacles += 1;
        }

        
        
        allLandmarkCreated = false;

        

        StartCoroutine(LandmarkGenerationCoroutine());

        totalNumberOfNodes = landmarks.Count;

        //Debug.Log(totalNumberOfNodes);

        if (totalNumberOfNodes > 500)
        {
            randomDestroyedRoundCountFixer = false;
        }
        else
        {
            randomDestroyedRoundCountFixer = true;
        }
    }

    public override void ResetArea()
    {

    }

    public (int, int, int) UpdateSimplices(List<int> currentNodes, List<List<int>> currentEdges, List<List<int>> currentTriangles)
    {
        int numberOfNewNodes = 0;
        int numberOfNewEdges = 0;
        int numberOfNewTriangles = 0;

        // Add nodes
        foreach(int node in currentNodes)
        {
            bool addStatus = nodes.Add(node);
            if (addStatus)
            {
                numberOfNewNodes += 1;
            }
        }

        //Debug.Log("Edge length" + currentEdges.Count);

        // Add edges
        foreach(List<int> edge in currentEdges)
        {
            Vector2 edgeVec = new Vector2(edge[0], edge[1]);

            bool existed = ((edges.Contains(edgeVec)) || 
                            (edges.Contains(new Vector2(edge[1], edge[0]))));
            //Debug.Log("new edge: " + (!existed));
            if (!existed)
            {
                edges.Add(edgeVec);
                numberOfNewEdges += 1;
            }

        }

        // Add triangles
        foreach(List<int> triangle in currentTriangles)
        {
            Vector3 triangleVec = new Vector3(triangle[0], triangle[1], triangle[2]);

            bool existed = ((triangles.Contains(triangleVec)) || 
                            (triangles.Contains(new Vector3(triangle[0], triangle[2], triangle[1]))) ||
                            (triangles.Contains(new Vector3(triangle[1], triangle[0], triangle[2]))) ||
                            (triangles.Contains(new Vector3(triangle[1], triangle[2], triangle[0]))) ||
                            (triangles.Contains(new Vector3(triangle[2], triangle[1], triangle[0]))) ||
                            (triangles.Contains(new Vector3(triangle[2], triangle[0], triangle[1])))
                            );
            //Debug.Log("new triangle: " + (!existed));
            if (!existed)
            {
                triangles.Add(triangleVec);
                GameObject currentTriangle = (GameObject)Instantiate(empty2Simplex, new Vector3(0, 0, 0), Quaternion.identity);
                
                Vector3 vertex1 = landmarks[triangle[0]].transform.position + new Vector3(0, 18, 0);
                Vector3 vertex2 = landmarks[triangle[1]].transform.position + new Vector3(0, 18, 0);
                Vector3 vertex3 = landmarks[triangle[2]].transform.position + new Vector3(0, 18, 0);

                currentTriangle.GetComponent<DrawTriangle>().drawTriangle(vertex1, vertex2, vertex3);

                numberOfNewTriangles += 1;
            }

        }
        
        //Debug.Log("Number of Nodes in HashSet: " + nodes.Count);
        //Debug.Log("Number of Edges in HashSet: " + edges.Count);
        //Debug.Log("Number of Triangles in HashSet: " + triangles.Count);
        

        return (numberOfNewNodes, numberOfNewEdges, numberOfNewTriangles);
    }



    private void OnApplicationQuit()
    {
        string terminateServerAPI = "localhost:80/terminate";
        StartCoroutine(GetRequest(terminateServerAPI));
    }

    IEnumerator GetRequest(string uri)
    {
        UnityWebRequest uwr = UnityWebRequest.Get(uri);
        yield return uwr.SendWebRequest();

        if (uwr.result == UnityWebRequest.Result.ConnectionError)
        {
            Debug.Log("Error While Sending: " + uwr.error);
        }
        else
        {
            Debug.Log("Received: " + uwr.downloadHandler.text);
        }
    }

    IEnumerator LandmarkGenerationCoroutine()
    {
        CreateLandmark(landmark);
        yield return new WaitUntil(() => allLandmarkCreated);

        agents[0].transform.GetChild(3).gameObject.GetComponent<Detector>().detectionFreeze = false;
        agents[1].transform.GetChild(3).gameObject.GetComponent<Detector>().detectionFreeze = false;
        agents[2].transform.GetChild(3).gameObject.GetComponent<Detector>().detectionFreeze = false;
        agents[3].transform.GetChild(3).gameObject.GetComponent<Detector>().detectionFreeze = false;
    }
}
