using UnityEngine;

public class LandmarkLogic : MonoBehaviour
{
    // public bool respawn;
    public LandmarkComplexArea myArea;

    public int landmarkID;

    public void OnDetected()
    {
        gameObject.tag = "detectedLandmark";
    }
    
    /*
    void OnCollisionEnter(Collision c)
    {
        if(c.gameObject.tag == "wall")
        {
            // myArea.landmarks.Remove(this.gameObject);
            // myArea.totalNumberOfNodes = myArea.landmarks.Count;
            // Destroy(this);
            myArea.totalNumberOfNodes -= 1;
        }
    }*/
}


