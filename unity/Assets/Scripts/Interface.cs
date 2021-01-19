using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine; // SaveFilePanel, OpenFilePanel
using UnityEditor;
using RapidGUI;


public class Interface : MonoBehaviour {
    private Rect windowRect = new Rect(0, 0, 500, 500);
    public string picturePath;
    public string modelPath;
    public double result;

    public enum ModelType { Linear, MLP, };
    public ModelType modelType;
    public bool isClassification;
    public int epochs;
    public float learningRate; // always positive between 0 and 1
    

    private IntPtr model;

    public void Start() {
        model = IntPtr.Zero;
        modelType = ModelType.MLP;
        isClassification = true;
    }

	private void OnGUI() {
        windowRect = GUI.ModalWindow(GetHashCode(), windowRect, DoGUI, "Actions", RGUIStyle.darkWindow);
    }

    public void DoGUI(int windowID) {
        // Draw the GUI
        // Analyse part
        GUILayout.Label("Analyse pictures :");

        picturePath = RGUI.Field(picturePath, "Picture path");
        modelPath = RGUI.Field(modelPath, "Model path");
        if (GUILayout.Button("Compute")) {

        }
        GUILayout.Label("Result of the analyse (0 is Manga and 1 is BD) => " + result);


        // Training part
        GUILayout.Label("\n\nTrain model :");

        modelType = RGUI.Field(modelType, "Model Type");
        isClassification = RGUI.Field(isClassification, "Is Classification");
        epochs = RGUI.Slider(epochs, 10000, "Epochs");
        learningRate = RGUI.Slider(learningRate, "Learning rate");

        if (GUILayout.Button("Create Model")) {
            if (model != IntPtr.Zero) {
                LoadLibrary.DeleteModel(model);
            }
            if (modelType == ModelType.Linear) // MLP doesn't exist for now
                model = LoadLibrary.CreateModel((int)modelType, 3, isClassification);
        }

        if (model != IntPtr.Zero) {
            if (GUILayout.Button("Save Model")) {
                string path = EditorUtility.SaveFilePanel("Save model as csv", "", "model.csv", "csv");
                LoadLibrary.SaveModel(model, path);
            }

            if (GUILayout.Button("Load Model")) {
                string path = EditorUtility.OpenFilePanel("Overwrite with csv", "", "csv");
                LoadLibrary.LoadModel(model, path);
            }
        }

    }

    void OnDestroy() {
        LoadLibrary.DeleteModel(model);
        model = IntPtr.Zero;
    }
}