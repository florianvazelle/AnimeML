using System;
using System.Runtime.InteropServices;

namespace TestCSharpLibrary
{

    public static class LoadLibrary
    {
        [DllImport("example")]
        public static extern IntPtr create_linear_model(int inputs_count);

        [DllImport("example")]
        public static extern void delete_linear_model(IntPtr model);
    }
}