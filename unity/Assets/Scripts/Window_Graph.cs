using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Window_Graph : MonoBehaviour
{

    [SerializeField] private Sprite circleSprite;
    private RectTransform graphContainer;
    private RectTransform labelTemplateX;
    private RectTransform labelTemplateY;

    private void Awake()
    {
        graphContainer = transform.Find("graphContainer").GetComponent<RectTransform>();
        labelTemplateX = graphContainer.Find("labelTemplateX").GetComponent<RectTransform>();
        labelTemplateY = graphContainer.Find("labelTemplateY").GetComponent<RectTransform>();

        //Show one Circle
        //CreateCircle(new Vector2(10, 10));

        //Show the entire Circle
        List<Vector2> valueList = new List<Vector2>() { new Vector2(10,20), new Vector2(5, 50) };
        ShowGraph(valueList);
    }

    private GameObject CreateCircle(Vector2 anchoredPosition)
    {
        GameObject gameObject = new GameObject("circle", typeof(Image));
        gameObject.transform.SetParent(graphContainer, false);
        gameObject.GetComponent<Image>().sprite = circleSprite;
        RectTransform rectTransform = gameObject.GetComponent<RectTransform>();
        rectTransform.anchoredPosition = anchoredPosition;
        rectTransform.sizeDelta = new Vector2(11, 11);
        rectTransform.anchorMin = new Vector2(0, 0);
        rectTransform.anchorMax = new Vector2(0, 0);
        return gameObject;
    }

    private void ShowGraph(List<Vector2> valueList)
    {
        float graphHeight = graphContainer.sizeDelta.y;
        float graphwidth = graphContainer.sizeDelta.x;
        float yMaximum = 100f;
        float xMaximum = 100f;
        //float xSize = 50f;

        for (int i = 0; i < valueList.Count; i++)
        {
            //float xPosition = xSize + i * xSize;
            float xPosition = (valueList[i].x / xMaximum) * graphHeight;
            float yPosition = (valueList[i].y / yMaximum) * graphHeight;
            GameObject circleGameObject = CreateCircle(new Vector2(xPosition, yPosition));

        }

        int separatorCount = 10;
        for (int i = 0; i <= separatorCount; i++)
        {
            RectTransform labelX = Instantiate(labelTemplateX);
            labelX.SetParent(graphContainer, false);
            labelX.gameObject.SetActive(true);
            float normalizedValueX = i * 1f / separatorCount;
            labelX.anchoredPosition = new Vector2(normalizedValueX * graphwidth , - 7f);
            labelX.GetComponent<Text>().text = ((normalizedValueX * xMaximum) / 100).ToString();

            RectTransform labelY = Instantiate(labelTemplateY);
            labelY.SetParent(graphContainer, false);
            labelY.gameObject.SetActive(true);
            float normalizedValue = i * 1f / separatorCount;
            labelY.anchoredPosition = new Vector2(-7f, normalizedValue * graphHeight);
            labelY.GetComponent<Text>().text = ((normalizedValue * yMaximum) / 100).ToString();
            
        }

    }

}
