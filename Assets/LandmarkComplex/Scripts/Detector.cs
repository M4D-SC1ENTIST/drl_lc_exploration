using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Detector : MonoBehaviour
{
    public bool detectionFreeze = false;
    public List<int> currentObservedLandmarkIDs = new List<int>();

    private void OnTriggerStay(Collider other)
    {
        if (!detectionFreeze)
        {
            // Debug.Log("Perform collision checking");
            // Check if it is a landmark
            if ((other.tag == "undetectedLandmark") || (other.tag == "detectedLandmark"))
            {
                // Debug.Log("Candidate is landmark");
                // Check if it is behind a wall
                RaycastHit hit;
                if (Physics.Linecast(transform.position, other.transform.position, out hit))
                {
                    if ((hit.transform.tag == "undetectedLandmark") || (hit.transform.tag == "detectedLandmark"))
                    {
                        int otherLandmarkID = other.gameObject.GetComponent<LandmarkLogic>().landmarkID;
                        if(!currentObservedLandmarkIDs.Contains(otherLandmarkID))
                        {
                            currentObservedLandmarkIDs.Add(otherLandmarkID);
                        }  
                    }
                    else
                    {
                        int otherLandmarkID = other.gameObject.GetComponent<LandmarkLogic>().landmarkID;
                        if(currentObservedLandmarkIDs.Contains(otherLandmarkID))
                        {
                            currentObservedLandmarkIDs.Remove(otherLandmarkID);
                        } 
                    }
                }
            }
        }
        
    }

    

    private void OnTriggerExit(Collider other)
    {
        if (!detectionFreeze)
        {
            if ((other.tag == "undetectedLandmark") || (other.tag == "detectedLandmark"))
            {
                int otherLandmarkID = other.gameObject.GetComponent<LandmarkLogic>().landmarkID;
                if(currentObservedLandmarkIDs.Contains(otherLandmarkID))
                {
                    currentObservedLandmarkIDs.Remove(otherLandmarkID);
                }
            }
        }

        
    }
}
