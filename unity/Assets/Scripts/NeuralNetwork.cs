using TestCSharpLibrary;
using UnityEngine;
using System;
using System.Runtime.InteropServices;
using System.Linq;
using System.Drawing;

public class NeuralNetwork : MonoBehaviour
{
    public const int trainingSet_Size = 3;

    public double[,] inputs = new double[3, 2] {
        {1, 1},
        {2, 3},
        {3, 3}
    };

    public double[,] outputs = new double[3, 1] {
        { 1 },
        { -1 },
        { -1 },
    };

    
    private double[] linearModel_Managed; // weights array -> size of inputs (2)

    public int epochs = 1;
    private int count = 0;

    private const int numInputs = 2;
    //private const int numHiddenNodes = 2;
    private const int numOutputs = 1;

    //double hiddenLayer[numHiddenNodes];
    //double outputLayer[numOutputs];
    //double hiddenLayerBias[numHiddenNodes];
    //double[] outputLayerBias;
    //double hiddenWeights[numInputs][numHiddenNodes];
    //double outputWeights[numHiddenNodes][numOutputs];

    private void Start()
    {

        // Initialise
        initializeWeights();


        // Training
        Debug.Log("........................Start training ......................");

        for (int i = 0; i < epochs; i++)
        {
            Debug.Log("........................Iteration : " + count + "............");
            int[] trainingSetOrder = new int[trainingSet_Size];

            shuffleTrainingSetArray(ref trainingSetOrder);
           

            // Cycle through each of the training set elements
            for (int x = 0; x < trainingSet_Size; x++)
            {
                int setID = trainingSetOrder[x];

                Debug.Log("set: " + setID);

                double[] setInputs = new double[numInputs];
                for (int j = 0; j < setInputs.Length; j++)
                {
                    setInputs[j] = inputs[setID,j];
                }
                Debug.Log("inputs: [ " + setInputs[0] + ", " + setInputs[1] + " ]");

                double[] setOutputs = new double[numOutputs];
                for (int j = 0; j < setOutputs.Length; j++)
                {
                    setOutputs[j] = outputs[setID, j];
                }
                Debug.Log("outputs: [ " + setOutputs[0] + " ]");
                // Compute output layer activation

                double totalError = 0;
                double activation = 0;
                for (int j = 0; j < numOutputs; j++) // One unique output
                {
                    for (int k = 0; k < setInputs.Length; k++)
                    {
                        activation += setInputs[k] * linearModel_Managed[k];
                    }
                    activation = sigmoid(activation);
                    Debug.Log("Activation = " + activation);
                    double squaredError = 0.5 * Mathf.Pow((float)(activation - setOutputs[j]), 2);
                    Debug.Log("Cost = " + squaredError);

                    totalError += squaredError;
                }
                //Debug.Log("Total cost = " + totalError);

                Debug.Log("before backpropagation .................");
                foreach (var item in linearModel_Managed)
                {
                    Debug.Log(item);
                }
                // backpropagation of error on weights
                for (int j = 0; j < numOutputs; j++) // One unique output
                {
                    for (int k = 0; k < setInputs.Length; k++)
                    {
                        linearModel_Managed[k] = updateWeight(
                            linearModel_Managed[k],
                            0.5f,
                            setOutputs[0], // one output
                            activation,
                            setInputs[k]
                            );
                    }
                }
                Debug.Log("after backpropagation .................");
                foreach (var item in linearModel_Managed)
                {
                    Debug.Log(item);
                }

            }

            count++;
        }
    }

    private void Update()
    {

    }

    private void initializeWeights()
    {
        Debug.Log("initialize weights for " + numInputs + " inputs") ; // initialize weights for each inputs
        var linearModel = LoadLibrary.create_linear_model(numInputs);
        linearModel_Managed = new double[numInputs];

        Marshal.Copy(linearModel, linearModel_Managed, 0, numInputs);

        foreach (var item in linearModel_Managed)
        {
            Debug.Log(item);
        }

        LoadLibrary.delete_linear_model(linearModel);
    }

    private void shuffleTrainingSetArray(ref int[] array)
    {
        Debug.Log("........................ Training set ......................");

        for (int j = 0; j < trainingSet_Size; j++)
        {
            array[j] = j;
        }

        shuffle(ref array);

        //for (int j = 0; j < trainingSet_Size; j++)
        //{
        //    Debug.Log(array[j]);
        //}
    }

    private void shuffle(ref int[] array)
    {
        System.Random rnd = new System.Random();
        array = array.OrderBy(x => rnd.Next()).ToArray();
    }

    // Activation function and its derivative
    private double sigmoid(double x) { return 1 / (1 + Mathf.Exp((float)-x)); }
    private double dSigmoid(double x) { return x * (1 - x); }

    private double updateWeight(double oldWeight, float learningRate, double targetValue, double actualValue, double entryValue)
    {
        return oldWeight - (learningRate * (actualValue - targetValue) * dSigmoid(actualValue) * entryValue);
    }
}
