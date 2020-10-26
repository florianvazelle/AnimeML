using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System.IO;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class UseLibrary : MonoBehaviour
{
    void Start()
    {
        int random = TestCSharpLibrary.LoadLibrary.GetRandom();
        Debug.Log(random);

        // TestCSharpLibrary.LoadLibrary.write();
        // ReadFile("test.txt");

        PreAllocTest(); // work
        AllocInTest();  // don't work
    }

    static void PreAllocTest()
    {   
        int size = 3 * 5;
        IntPtr pManagedArray = Marshal.AllocHGlobal(3 * 5 * sizeof(double));

        // Call test()
        TestCSharpLibrary.LoadLibrary.pre_alloc_test(pManagedArray);
        
        // Read IntPtr
        double[] managedArray = new double[size];
        Marshal.Copy(pManagedArray, managedArray, 0, size);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 5; j++) {
                Debug.Log(managedArray[i * 5 + j]);
            }
        }

        // Free the managed memory
        Marshal.FreeHGlobal(pManagedArray);
        pManagedArray = IntPtr.Zero;
    }

    static void AllocInTest()
    {
        IntPtr pUnmanagedArray = IntPtr.Zero;

        // Call test()
        int size = TestCSharpLibrary.LoadLibrary.alloc_in_test(pUnmanagedArray);
        
         // Read IntPtr
        double[] managedArray = new double[size];
        Marshal.Copy(pUnmanagedArray, managedArray, 0, size);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 5; j++) {
                Debug.Log(managedArray[i * 5 + j]);
            }
        }

        // Free the unmanaged memory
        TestCSharpLibrary.LoadLibrary.my_free(pUnmanagedArray);
        pUnmanagedArray = IntPtr.Zero;
    }

    void ReadFile(String file) {
        if (File.Exists(file)){
            var sr = File.OpenText(file);
            var line = sr.ReadLine();
            while(line != null){
                Debug.Log(line); // prints each line of the file
                line = sr.ReadLine();
            }  
        } else {
            Debug.Log("Could not Open the file: " + file + " for reading.");
            return;
        }
    }
}