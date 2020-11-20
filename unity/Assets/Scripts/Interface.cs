using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine; // SaveFilePanel, OpenFilePanel
using UnityEditor;
using RapidGUI;

public class Interface : MonoBehaviour {

    private Rect windowRect = new Rect(0, 0, 250, 250);

    public enum ModelType { Linear, MLP, };

    public ModelType modelType;
    public bool isClassification;

    private IntPtr model;

    public void Start() {
        model = IntPtr.Zero;
        modelType = ModelType.Linear;
        isClassification = true;
    }

	private void OnGUI() {
        windowRect = GUI.ModalWindow(GetHashCode(), windowRect, DoGUI, "Actions", RGUIStyle.darkWindow);
    }

    public void DoGUI(int windowID) {
        modelType = RGUI.Field(modelType, "Model Type");
        isClassification = RGUI.Field(isClassification, "Is Classification");

        if (GUILayout.Button("Create Model")) {
            if (model != IntPtr.Zero) {
                LoadLibrary.DeleteModel(model);
            }
            if (modelType == 0) // MLP doesn't exist for now
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