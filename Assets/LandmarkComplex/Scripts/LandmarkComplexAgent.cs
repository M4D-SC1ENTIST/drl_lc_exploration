using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using static Combinations<int>;

using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;

using UnityEngine.Networking;


// Helper Classes
[Serializable]
public class ObservationStruct
{
    public int agent_id;
    public List<int> left_observed_landmark_id;
    public List<int> right_observed_landmark_id;
    public List<int> straight_observed_landmark_id;
    public List<float> agent_location;
    public List<float> agent_rotation;
}

[Serializable]
public class ObservationResponseStruct
{
    public bool not_in_complex;
    public int num_triangles_added;
    public List<int> triangles_added;
}


public class LandmarkComplexAgent : Agent
{
    public string observationReceiverAPI = "localhost:80/observation_receiver";
    public bool experimentalServerConnection = false;

    LandmarkComplexSettings m_LandmarkComplexSettings;
    public GameObject area;
    LandmarkComplexArea m_MyArea;

    bool m_SendObservation;

    public bool visualizeDetectionRangeWhenSendingObservation;

    Rigidbody m_AgentRb;
    // float m_LaserLength;
    // Speed of agent rotation.
    public float turnSpeed = 300;

    // Speed of agent movement.
    public float moveSpeed = 2;


    // public GameObject myLaser;

    public GameObject myDetector;

    public bool contribute;
    public bool useVectorObs;
    /*
    [Tooltip("Use only the frozen flag in vector observations. If \"Use Vector Obs\" " +
             "is checked, this option has no effect. This option is necessary for the " +
             "VisualFoodCollector scene.")]
    public bool useVectorFrozenFlag;
    */

    EnvironmentParameters m_ResetParams;

    Detector m_Detector;

    public override void Initialize()
    {
        m_AgentRb = GetComponent<Rigidbody>();
        m_MyArea = area.GetComponent<LandmarkComplexArea>();
        m_LandmarkComplexSettings = FindObjectOfType<LandmarkComplexSettings>();
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        m_Detector = myDetector.GetComponent<Detector>();
        SetResetParameters();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (useVectorObs)
        {
            var localVelocity = transform.InverseTransformDirection(m_AgentRb.velocity);
            sensor.AddObservation(localVelocity.x);
            sensor.AddObservation(localVelocity.z);

            sensor.AddObservation(m_SendObservation);
        }
        // else if (useVectorFrozenFlag)
        // {
        //     sensor.AddObservation(m_Frozen);
        // }
    }

    public Color32 ToColor(int hexVal)
    {
        var r = (byte)((hexVal >> 16) & 0xFF);
        var g = (byte)((hexVal >> 8) & 0xFF);
        var b = (byte)(hexVal & 0xFF);
        return new Color32(r, g, b, 255);
    }

    public void MoveAgent(ActionBuffers actionBuffers)
    {
        m_SendObservation = false;

        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var continuousActions = actionBuffers.ContinuousActions;
        var discreteActions = actionBuffers.DiscreteActions;


        var forward = Mathf.Clamp(continuousActions[0], -1f, 1f);
        var right = Mathf.Clamp(continuousActions[1], -1f, 1f);
        var rotate = Mathf.Clamp(continuousActions[2], -1f, 1f);

        dirToGo = transform.forward * forward;
        dirToGo += transform.right * right;
        rotateDir = -transform.up * rotate;

        var sendObservationCommand = discreteActions[0] > 0;
        if (sendObservationCommand)
        {
            m_SendObservation = true;
            // dirToGo *= 0.5f;
            // m_AgentRb.velocity *= 0.75f;
        }
        m_AgentRb.AddForce(dirToGo * moveSpeed, ForceMode.VelocityChange);
        transform.Rotate(rotateDir, Time.fixedDeltaTime * turnSpeed);


        if (m_AgentRb.velocity.sqrMagnitude > 25f) // slow it down
        {
            m_AgentRb.velocity *= 0.95f;
        }

        // Action for send observation
        if (m_SendObservation)
        {
            try
            {
                myDetector.GetComponent<Renderer>().enabled = true;
            
                List<int> currentNodes = m_Detector.currentObservedLandmarkIDs;

                if (experimentalServerConnection)
                {
                    ObservationStruct currentObservationInstance = new ObservationStruct();
                    currentObservationInstance.left_observed_landmark_id = currentNodes;

                    Vector3 currentAgentLocation = gameObject.transform.parent.position;
                    Quaternion currentAgentRotation = gameObject.transform.parent.rotation;

                    currentObservationInstance.agent_location = new List<float>() { currentAgentLocation.x, currentAgentLocation.y, currentAgentLocation.z };
                    currentObservationInstance.agent_rotation = new List<float>() { currentAgentRotation.w, currentAgentRotation.x, currentAgentRotation.y, currentAgentRotation.z };

                    string currentObservationJSON = JsonUtility.ToJson(currentObservationInstance);

                    StartCoroutine(ObservationPostRequest(observationReceiverAPI, currentObservationJSON));
                }

                foreach (int id in currentNodes)
                {
                    m_MyArea.landmarks[id].tag = "detectedLandmark";
                }

                List<List<int>> currentEdges = Combinations<int>.GetCombinations(currentNodes, 2);

                List<List<int>> currentTriangles = Combinations<int>.GetCombinations(currentNodes, 3);

                (int numNewNodes, int numNewEdges, int numNewTriangles) = m_MyArea.UpdateSimplices(currentNodes, currentEdges, currentTriangles);

                //Debug.Log("Number of new nodes added: " + numNewNodes);
                //Debug.Log("Number of new edges added: " + numNewEdges);
                //Debug.Log("Number of new triangles added: " + numNewTriangles);

                AddReward(numNewNodes);
                if (contribute)
                {
                    m_LandmarkComplexSettings.totalScore += numNewNodes;
                }

                AddReward(numNewEdges * 1.5f);
                if (contribute)
                {
                    m_LandmarkComplexSettings.totalScore += numNewEdges;
                }

                AddReward(numNewTriangles * 2);
                if (contribute)
                {
                    m_LandmarkComplexSettings.totalScore += 2 * numNewTriangles;
                }

                // Deduction for making observation
                AddReward(-2f);
                if (contribute)
                {
                    m_LandmarkComplexSettings.totalScore -= 2;
                }
            }
            catch
            {
                Debug.Log("Index out of range. Error catched. Probably because of resetting the environment");
            }
            
        }
        else
        {
            myDetector.GetComponent<Renderer>().enabled = false;
        }
    }

    

    public override void OnActionReceived(ActionBuffers actionBuffers)

    {
        MoveAgent(actionBuffers);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        if (Input.GetKey(KeyCode.D))
        {
            continuousActionsOut[2] = 1;
        }
        if (Input.GetKey(KeyCode.W))
        {
            continuousActionsOut[0] = 1;
        }
        if (Input.GetKey(KeyCode.A))
        {
            continuousActionsOut[2] = -1;
        }
        if (Input.GetKey(KeyCode.S))
        {
            continuousActionsOut[0] = -1;
        }
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = Input.GetKey(KeyCode.Space) ? 1 : 0;
    }

    public override void OnEpisodeBegin()
    {
        m_Detector.currentObservedLandmarkIDs.Clear();
        m_SendObservation = false;
        m_AgentRb.velocity = Vector3.zero;
        myDetector.GetComponent<Renderer>().enabled = false;

        if (gameObject == m_MyArea.agents[0])
        {
            transform.position = new Vector3(-m_MyArea.horizontalRange, 2f, -m_MyArea.verticalRange) + area.transform.position;;
            transform.rotation = Quaternion.Euler(new Vector3(0f, 45f, 0f));
        }
        else if (gameObject == m_MyArea.agents[1])
        {
            transform.position = new Vector3(m_MyArea.horizontalRange, 2f, -m_MyArea.verticalRange) + area.transform.position;;
            transform.rotation = Quaternion.Euler(new Vector3(0f, -45f, 0f));
        }
        else if (gameObject == m_MyArea.agents[2])
        {
            transform.position = new Vector3(m_MyArea.horizontalRange, 2f, m_MyArea.verticalRange) + area.transform.position;;
            transform.rotation = Quaternion.Euler(new Vector3(0f, 255f, 0f));
        }
        else if (gameObject == m_MyArea.agents[3])
        {
            transform.position = new Vector3(-m_MyArea.horizontalRange, 2f, m_MyArea.verticalRange) + area.transform.position;;
            transform.rotation = Quaternion.Euler(new Vector3(0f, 120f, 0f));
        }

        SetResetParameters();
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("wall") || collision.gameObject.CompareTag("agent"))
        {
            AddReward(-5.0f);
            if (contribute)
            {
                m_LandmarkComplexSettings.totalScore -= 5;
            }
        }
    }

/*
    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("undetectedLandmark"))
        {
            //Satiate();
            collision.gameObject.GetComponent<LandmarkLogic>().OnDetected();
            AddReward(1f);
            if (contribute)
            {
                m_LandmarkComplexSettings.totalScore += 1;
            }
        }
        if (collision.gameObject.CompareTag("detectedLandmark"))
        {
            //Poison();
            collision.gameObject.GetComponent<LandmarkLogic>().OnDetected();

            AddReward(-1f);
            if (contribute)
            {
                m_LandmarkComplexSettings.totalScore -= 1;
            }
        }
    }
*/
    /*
    public void SetLaserLengths()
    {
        m_LaserLength = m_ResetParams.GetWithDefault("laser_length", 1.0f);
    }
    */

    public void SetAgentScale()
    {
        float agentScale = m_ResetParams.GetWithDefault("agent_scale", 1.0f);
        gameObject.transform.localScale = new Vector3(agentScale, agentScale, agentScale);
    }

    public void SetResetParameters()
    {
        SetAgentScale();
    }





    IEnumerator ObservationPostRequest(string url, string json)
    {
        var uwr = new UnityWebRequest(url, "POST");
        byte[] jsonToSend = new System.Text.UTF8Encoding().GetBytes(json);
        uwr.uploadHandler = (UploadHandler)new UploadHandlerRaw(jsonToSend);
        uwr.downloadHandler = (DownloadHandler)new DownloadHandlerBuffer();
        uwr.SetRequestHeader("Content-Type", "application/json");

        //Send the request then wait here until it returns
        yield return uwr.SendWebRequest();

        if (uwr.result == UnityWebRequest.Result.ConnectionError)
        {
            Debug.Log("Error While Sending: " + uwr.error);
        }
        else
        {
            Debug.Log("Server received");
            
        }
    }
}
