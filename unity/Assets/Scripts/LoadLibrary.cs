using System;
using System.Runtime.InteropServices;

public static class LoadLibrary
{
    /* For detail on all fields, check documentation on Library.cpp */

    [DllImport("example")]
    public static extern IntPtr CreateModel(int flag, int weights_count, bool is_classification);

    [DllImport("example")]
    public static extern void Train(
        IntPtr model, int sample_count,
        IntPtr train_inputs, int inputs_size,
        IntPtr train_outputs, int outputs_size,
        int epochs, double learning_rate
    );

    [DllImport("example")]
    public static extern double Predict(
        IntPtr model, int sample_count,
        IntPtr inputs, int inputs_size,
        IntPtr outputs, int outputs_size
    );

    [DllImport("example")]
    public static extern IntPtr GetWeigths(IntPtr model);

    [DllImport("example")]
    public static extern void SaveModel(IntPtr model, string path);

    [DllImport("example")]
    public static extern void LoadModel(IntPtr model, string path);

    [DllImport("example")]
    public static extern void DeleteModel(IntPtr model);
}