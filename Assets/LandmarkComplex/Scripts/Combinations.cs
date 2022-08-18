using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

public class Combinations<T>
{
    private List<T> m_Items;
    private List<List<T>> m_Result;
    private T[] current;
    private int m_Length;
    private Combinations(List<T> aItems, int aLength)
    {
        m_Items = aItems;
        m_Length = aLength;
        m_Result = new List<List<T>>();
        current = new T[aLength];
    }
    public static List<List<T>> GetCombinations(List<T> aItems, int aLength)
    {
        if (aItems == null || aItems.Count < aLength)
            return new List<List<T>>();
        var context = new Combinations<T>(aItems, aLength);
        
        context.GetCombinations(0, 0);
        return context.m_Result;
    }
    private void GetCombinations(int aStart, int aDepth)
    {
        if (aDepth >= m_Length)
            return;
        int c = m_Items.Count + aDepth - m_Length + 1;
        for (int i = aStart; i < c; i++)
        {
            current[aDepth] = m_Items[i];
            if (aDepth == m_Length-1)
                m_Result.Add(current.ToList());
            else
                GetCombinations(i+1, aDepth + 1);
        }
    }
}