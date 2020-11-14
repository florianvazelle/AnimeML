using System;
using System.Runtime.InteropServices;

namespace TestCSharpLibrary
{

    public static class LoadLibrary
    {
        [DllImport("example")]
        public static extern IntPtr create_linear_model(int inputs_count);

        [DllImport("example")]
        public static extern void my_free(IntPtr pArrayOfDouble);

        [DllImport("example")]
        public static extern void write();

        [DllImport("example")]
        public static extern IntPtr create_linear_model(int weights_count);

        //        [DllImport("example")]
        //        public static extern void train_linear_model_rosenblatt_test(IntPtr weights, int weights_count, int epochs);

        //[DllImport("example")]
        //public static extern void train_linear_model_rosenblatt_test(
        //    IntPtr weights,
        //    int weights_count,
        //    int sample_count_size,
        //    IntPtr all_inputs,
        //    int inputs_size,
        //    int epochs
        //);

        [DllImport("example")]
        public static extern void train_linear_model_rosenblatt_test(
            IntPtr weights,
            int weights_count,
            int sample_count_size,
            IntPtr all_inputs,
            int inputs_size,
            IntPtr all_outputs,
            int outputs_size,
            int epochs,
            float learningRate
        );

        [DllImport("example")]
        public static extern double predict_linear_model_rosenblatt_test(
            IntPtr weights,
            IntPtr setInputs,
            IntPtr setOutputs,
            int inputs_size = 2,
            int outputs_size = 1
        );

        [DllImport("example")]
        public static extern void delete_linear_model(IntPtr model);
    }
}