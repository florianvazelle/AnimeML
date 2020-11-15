using System;
using System.Runtime.InteropServices;

public static class LoadLibrary
{
    /* For detail on all fields, check documentation on Example.cpp */

    [DllImport("example")]
    public static extern IntPtr CreateModel(int flag, int weights_count);

    [DllImport("example")]
    public static extern void Train(
        IntPtr model, int sample_count,
        IntPtr train_inputs, int inputs_size,
        IntPtr train_outputs, int outputs_size,
        int epochs, float learning_rate
    );

    [DllImport("example")]
    public static extern double Predict(
        IntPtr model,
        IntPtr inputs, int inputs_size,
        IntPtr outputs, int outputs_size
    );

    [DllImport("example")]
    public static extern IntPtr GetWeigths(IntPtr model);

    [DllImport("example")]
    public static extern void DeleteModel(IntPtr model);
}