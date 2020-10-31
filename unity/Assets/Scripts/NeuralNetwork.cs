using TestCSharpLibrary;
using UnityEngine;
using System;
using System.Runtime.InteropServices;
using System.Linq;

public class NeuralNetwork : MonoBehaviour
{
    public const int trainingSet_Size = 3;

    public double[,] inputs = new double[3, 2] {
        {1, 1},
        {2, 3},
        {3, 3}
    };

    public double[] output = new double[3] {
        1,
        -1,
        -1,
    };

    
    private double[] linearModel_Managed;

    public int epochs = 1;

    private const int numInputs = 2;
    //private const int numHiddenNodes = 2;
    private const int numOutputs = 1;

    //double hiddenLayer[numHiddenNodes];
    //double outputLayer[numOutputs];
    //double hiddenLayerBias[numHiddenNodes];
    //double outputLayerBias[numOutputs];
    //double hiddenWeights[numInputs][numHiddenNodes];
    //double outputWeights[numHiddenNodes][numOutputs];

    private void Start()
    {
        Debug.Log(trainingSet_Size);
        var linearModel = LoadLibrary.create_linear_model(trainingSet_Size);
        linearModel_Managed = new double[trainingSet_Size];

        Marshal.Copy(linearModel, linearModel_Managed, 0, trainingSet_Size);

        foreach (var item in linearModel_Managed)
        {
            Debug.Log(item);
        }

        LoadLibrary.delete_linear_model(linearModel);

        // Training
        Debug.Log("........................Start training ......................");
        for (int i = 0; i < epochs; i++)
        {
            int[] trainingSetOrder = new int[trainingSet_Size];
            for (int j = 0; j < trainingSet_Size; j++)
            {
                trainingSetOrder[j] = j;
            }

            shuffle(ref trainingSetOrder);

            for (int j = 0; j < trainingSet_Size; j++)
            {
                Debug.Log(trainingSetOrder[j]);
            }
            Debug.Log("............................................");

            // Cycle through each of the training set elements
            for (int x = 0; x < trainingSet_Size; x++)
            {
                int setID = trainingSetOrder[x];

                // Compute output layer activation
                //for (int j = 0; j < nb_Output; j++)
                //{
                //    double activation = outputLayerBias[j];
                //    for (int k = 0; k < numHiddenNodes; k++)
                //    {
                //        activation += hiddenLayer[k] * outputWeights[k][j];
                //    }
                //    outputLayer[j] = sigmoid(activation);
                //}
            }
        }
    }

    private void Update()
    {

    }

    private void shuffle(ref int[] array)
    {
        System.Random rnd = new System.Random();
        array = array.OrderBy(x => rnd.Next()).ToArray();
    }
}
