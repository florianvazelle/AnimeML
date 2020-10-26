using System;
using System.Runtime.InteropServices;

namespace TestCSharpLibrary
{

    public class LoadLibrary
    {

        [DllImport("helloworldShared")]
        public static extern int GetRandom();

        [DllImport("helloworldShared")]
        public static extern int pre_alloc_test(IntPtr pArrayOfDouble);

        [DllImport("helloworldShared")]
        public static extern int alloc_in_test(IntPtr pArrayOfDouble);

        [DllImport("helloworldShared")]
        public static extern void my_free(IntPtr pArrayOfDouble);

        [DllImport("helloworldShared")]
        public static extern void write();
    }
}