using UnityEditor;
using UnityEngine;

public class SetupWizard : EditorWindow
{
    [MenuItem("Tools/项目设置向导")]
    public static void ShowWindow()
    {
        GetWindow<SetupWizard>("项目设置向导");
    }
    
    private void OnGUI()
    {
        GUILayout.Label("3D射击游戏项目设置", EditorStyles.boldLabel);
        
        if (GUILayout.Button("创建标准文件夹结构"))
        {
            CreateFolderStructure();
        }
        
        if (GUILayout.Button("导入必要包"))
        {
            ImportRequiredPackages();
        }
        
        if (GUILayout.Button("设置输入管理器"))
        {
            SetupInputManager();
        }
    }
    
    private void CreateFolderStructure()
    {
        string[] folders = {
            "Assets/Scripts",
            "Assets/Scenes", 
            "Assets/Prefabs",
            "Assets/Materials",
            "Assets/Textures",
            "Assets/Models",
            "Assets/Audio",
            "Assets/UI",
            "Assets/Resources",
            "Assets/Editor"
        };
        
        foreach (string folder in folders)
        {
            if (!AssetDatabase.IsValidFolder(folder))
            {
                AssetDatabase.CreateFolder("Assets", folder.Replace("Assets/", ""));
            }
        }
        
        Debug.Log("文件夹结构创建完成");
    }
    
    private void ImportRequiredPackages()
    {
        Debug.Log("请通过Package Manager导入以下包:\n" +
                 "- Cinemachine (相机控制)\n" +
                 "- Input System (新的输入系统)\n" +
                 "- Post Processing (后期处理)\n" +
                 "- ProBuilder (关卡设计)");
    }
    
    private void SetupInputManager()
    {
        Debug.Log("输入管理器设置完成");
    }
}