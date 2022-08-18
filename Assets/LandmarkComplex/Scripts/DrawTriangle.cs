using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DrawTriangle : MonoBehaviour
{
    public List<Material> triangleMaterials = new List<Material>();

    Mesh m;
    MeshFilter mf;

    // Use this for initialization
    void Start()
    {
        
    }

    //This draws a triangle
    public void drawTriangle(Vector3 vertex1, Vector3 vertex2, Vector3 vertex3)
    {
        mf = GetComponent<MeshFilter>();
        m = new Mesh();
        mf.mesh = m;

        MeshRenderer meshRend = GetComponent<MeshRenderer>();

        //We need two arrays one to hold the vertices and one to hold the triangles
        Vector3[] VerteicesArray = new Vector3[3];
        int[] trianglesArray = new int[3];

        //lets add 3 vertices in the 3d space
        VerteicesArray[0] = vertex1;
        VerteicesArray[1] = vertex2;
        VerteicesArray[2] = vertex3;

        //define the order in which the vertices in the VerteicesArray shoudl be used to draw the triangle
        trianglesArray[0] = 0;
        trianglesArray[1] = 1;
        trianglesArray[2] = 2;

        //add these two triangles to the mesh
        m.vertices = VerteicesArray;
        m.triangles = trianglesArray;

        meshRend.material = triangleMaterials[Random.Range(0, triangleMaterials.Count)];
    }
}
