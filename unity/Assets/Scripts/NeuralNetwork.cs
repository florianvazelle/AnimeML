using UnityEngine;
using System;
using System.Runtime.InteropServices;
using System.Linq;
using System.Drawing;
using System.Collections.Generic;

public class NeuralNetwork : MonoBehaviour
{
    public const int trainingSet_Size = 3;
    
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

    public int epochs = 10;
    public float learningRate = 0.5f;

    private void Start() {
        // Init model
        IntPtr model = LoadLibrary.CreateModel((int)Interface.ModelType.Linear, numInputs, true);

        IntPtr inputs_ptr = ConvertManagedToUnmanaged(inputs);
        IntPtr outputs_ptr = ConvertManagedToUnmanaged(outputs);

        // Passing to the DLL and training
        LoadLibrary.Train(
            model,                  // weights
            trainingSet_Size,       // number of training sets
            inputs_ptr,             // all_inputs array
            numInputs,              // number of inputs for 1 set
            outputs_ptr,            // all_inputs array
            numOutputs,             // number of inputs for 1 set
            epochs,                 // number of epoch
            learningRate            // learning rate
        );

        // Display results
        Debug.Log("... End Results ...");
        IntPtr Result = Marshal.AllocCoTaskMem(sizeof(double) * 3);
        // On predit sur les cas de test pour debuger
        LoadLibrary.Predict(model, trainingSet_Size, inputs_ptr, numInputs, Result, numOutputs); 
        DebugUnmanagedList(Result, 3);


        // Display weights
        Debug.Log("... End Weights ...");
        IntPtr linearModel = LoadLibrary.GetWeigths(model);
        DebugUnmanagedList(linearModel, numInputs);

        // Free the model
        LoadLibrary.DeleteModel(model);

        Marshal.FreeCoTaskMem(inputs_ptr);
        Marshal.FreeCoTaskMem(outputs_ptr);
        Marshal.FreeCoTaskMem(Result);
    }

    /**
     * Helpers Methods
     */

    public static IntPtr ConvertManagedToUnmanaged(double[] data) {
        // create the ptr for the DLL
        IntPtr ptr = Marshal.AllocCoTaskMem(sizeof(double) * data.Length);

        // copy the data array into it
        Marshal.Copy(data, 0, ptr, data.Length);

        return ptr;
    }

    public static void DebugUnmanagedList(IntPtr unmanagedList, int size) {
        double[] managedList = new double[size];
        Marshal.Copy(unmanagedList, managedList, 0, size);
        DebugList<double>(managedList);
    }

    public static void DebugList<T>(IList<T> list) {
        foreach (T item in list) {
            Debug.Log(item);
        }
    }
}