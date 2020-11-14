using TestCSharpLibrary;
using UnityEngine;
using System;
using System.Runtime.InteropServices;
using System.Linq;
using System.Drawing;

public class NeuralNetwork : MonoBehaviour
{
    public const int trainingSet_Size = 3;

    //public double[,] inputs = new double[3, 2] {
    //    {1, 1},
    //    {2, 3},
    //    {3, 3}
    //};
    //
    //public double[,] outputs = new double[3, 1] {
    //    { 1 },
    //    { -1 }, // -1
    //    { -1 }, // -1
    //};

    private const int numInputs = 2;
    public double[] inputs = new double[3 * 2] {
        1, 1,
        2, 3,
        3, 3
    };

    private const int numOutputs = 1;
    public double[] outputs = new double[3 * 1] {
        1,
        0, // -1
        0, // -1
    };


    private double[] linearModel_Managed; // weights array -> size of inputs (2)

    public int epochs = 1;
    public float learningRate = 0.5f;

    //private int count = 0;


    //private const int numHiddenNodes = 2
    //double hiddenLayer[numHiddenNodes];
    //double outputLayer[numOutputs];
    //double hiddenLayerBias[numHiddenNodes];
    //double[] outputLayerBias;
    //double hiddenWeights[numInputs][numHiddenNodes];
    //double outputWeights[numHiddenNodes][numOutputs];

    private void Start()
    {

        // Initialise
        //initializeWeights();

        // Training
        //trainWeights(linearModel_Managed);
        trainWeightsFromNothing();


        /*
        Debug.Log("........................Start training ......................");

        for (int i = 0; i < epochs; i++)
        {
            Debug.Log("........................Iteration : " + count + "............");


            count++;
        }
        */
    }

    private void initializeWeights()
    {
        Debug.Log("initialize weights for " + numInputs + " inputs between 0 and 1") ; // initialize weights for each inputs
        var linearModel = LoadLibrary.create_linear_model(numInputs);
        linearModel_Managed = new double[numInputs];

        Marshal.Copy(linearModel, linearModel_Managed, 0, numInputs);

        foreach (var item in linearModel_Managed)
        {
            Debug.Log(item);
        }

        LoadLibrary.delete_linear_model(linearModel);
    }

    /*
    private void trainWeights(double[] weights)
    {
        // create the ptr for the DLL
        IntPtr ptr = Marshal.AllocCoTaskMem(sizeof(double) * weights.Length);

        // copy the weight array into it
        Marshal.Copy(weights, 0, ptr, weights.Length);

        // passing to the DLL
        LoadLibrary.train_linear_model_rosenblatt_test(ptr, weights.Length, epochs);

        Marshal.Copy(ptr, weights, 0, weights.Length);

        Marshal.FreeCoTaskMem(ptr);

        foreach(var item in weights)
        {
            Debug.Log(item);
        }
    }
    */

    private void trainWeightsFromNothing()
    {
        // init model
        var linearModel = LoadLibrary.create_linear_model(numInputs);

        // create the input_ptr for the DLL
        IntPtr inputs_ptr = Marshal.AllocCoTaskMem(sizeof(double) * inputs.Length);
        // copy the input array into it
        Marshal.Copy(inputs, 0, inputs_ptr, inputs.Length);


        // create the output_ptr for the DLL
        IntPtr outputs_ptr = Marshal.AllocCoTaskMem(sizeof(double) * outputs.Length);
        // copy the input array into it
        Marshal.Copy(outputs, 0, outputs_ptr, outputs.Length);

        // passing to the DLL and training
        //LoadLibrary.train_linear_model_rosenblatt_test(linearModel, numInputs, epochs);
        LoadLibrary.train_linear_model_rosenblatt_test(
            linearModel, // weights
            numInputs, // numWeight but here it's 1 layer
            trainingSet_Size, // number of training sets
            inputs_ptr, // all_inputs array
            numInputs, // number of inputs for 1 set
            outputs_ptr, // all_inputs array
            numOutputs, // number of inputs for 1 set
            epochs, // number of epoch
            learningRate // learning rate
        );

        linearModel_Managed = new double[numInputs];
        Marshal.Copy(linearModel, linearModel_Managed, 0, numInputs);

        // display results
        Debug.Log("... End Results ...");
        forwardPropagation(linearModel_Managed);

        // display weights
        Debug.Log("... End Weights ...");
        foreach (var item in linearModel_Managed)
        {
            Debug.Log(item);
        }

        // free the model
        LoadLibrary.delete_linear_model(linearModel);
        Marshal.FreeCoTaskMem(inputs_ptr);
        Marshal.FreeCoTaskMem(outputs_ptr);
    }

    private void forwardPropagation(double[] model)
    {
        int[] trainingSetOrder = new int[trainingSet_Size];

        // Shuffle the training set
        shuffleTrainingSetArray(ref trainingSetOrder);


        // Cycle through each of the training set elements
        for (int x = 0; x < trainingSet_Size; x++)
        {
            int setID = trainingSetOrder[x];

            //Debug.Log("set: " + setID);

            double[] setInputs = new double[numInputs];
            for (int j = 0; j < setInputs.Length; j++)
            {
                setInputs[j] = inputs[(setID * numInputs) + j];
            }
            Debug.Log("inputs: [ " + setInputs[0] + ", " + setInputs[1] + " ]");

            double[] setOutputs = new double[numOutputs];
            for (int j = 0; j < setOutputs.Length; j++)
            {
                setOutputs[j] = outputs[(setID * numOutputs) + j];
            }
            Debug.Log("outputs: [ " + setOutputs[0] + " ]");
            // Compute output layer activation

            double totalError = 0;
            double activation = 0;
            for (int j = 0; j < numOutputs; j++) // One unique output
            {
                for (int k = 0; k < setInputs.Length; k++)
                {
                    activation += setInputs[k] * model[k];
                }
                activation = sigmoid(activation);
                Debug.Log("Activation = " + activation);
                double squaredError = 0.5 * Mathf.Pow((float)(activation - setOutputs[j]), 2);

                totalError += squaredError;
            }

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
        }
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
