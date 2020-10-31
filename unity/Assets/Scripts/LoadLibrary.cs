using System;
using System.Runtime.InteropServices;

namespace TestCSharpLibrary
{

    public static class LoadLibrary
    {

        [DllImport("example")]
        public static extern int GetRandom();

        [DllImport("example")]
        public static extern int pre_alloc_test(IntPtr pArrayOfDouble);

        [DllImport("example")]
        public static extern int alloc_in_test(IntPtr pArrayOfDouble);

        [DllImport("example")]
        public static extern void my_free(IntPtr pArrayOfDouble);

        [DllImport("example")]
        public static extern void write();

        [DllImport("example")]
        public static extern IntPtr create_linear_model(int inputs_count);

        [DllImport("example")]
       public static extern void delete_linear_model(IntPtr model);
    }
}