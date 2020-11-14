using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System.IO;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// Needs to match the c++ order
enum Flag { LinearModel, MLP };

public class UseLibrary : MonoBehaviour
{
    [DllImport("example")]
    internal static extern IntPtr CreateModel(int flag);

    [DllImport("example")]
    internal static extern void Print(IntPtr model, int flag);

    [DllImport("example")]
    internal static extern void DeleteModel(IntPtr model);

    void Start()
    {
        IntPtr model = CreateModel((int)Flag.LinearModel);
        Print(model, (int)Flag.LinearModel);
        ReadFile("Linear.txt");
        DeleteModel(model);
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