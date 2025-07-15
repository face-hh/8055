// Copyright (c) Meta Platforms, Inc. and affiliates.
// Original Source code from Oculus Starter Samples (https://github.com/oculus-samples/Unity-StarterSamples)

using System;
using System.Collections.Generic;
using System.IO;
using Meta.XR.Samples;
using UnityEngine;

namespace PassthroughCameraSamples.StartScene
{
    // Create menu of all scenes included in the build.
    [MetaCodeSample("PassthroughCameraApiSamples-StartScene")]
    public class StartMenu : MonoBehaviour
    {
        public OVROverlay Overlay;
        public OVROverlay Text;
        public OVRCameraRig VrRig;

        // Static field to store server configuration
        public static string ServerURL = "http://91.199.227.82:11860";
        private string tempServerURL = "";

        // Static field to store anime mode
        public enum AnimeMode { Diffusion, GAN }
        public static AnimeMode CurrentAnimeMode = AnimeMode.Diffusion;

        private void Start()
        {
            tempServerURL = ServerURL;
            
            var generalScenes = new List<Tuple<int, string>>();
            var passthroughScenes = new List<Tuple<int, string>>();
            var proControllerScenes = new List<Tuple<int, string>>();

            var n = UnityEngine.SceneManagement.SceneManager.sceneCountInBuildSettings;
            for (var sceneIndex = 1; sceneIndex < n; ++sceneIndex)
            {
                var path = UnityEngine.SceneManagement.SceneUtility.GetScenePathByBuildIndex(sceneIndex);

                if (path.Contains("Passthrough"))
                {
                    passthroughScenes.Add(new Tuple<int, string>(sceneIndex, path));
                }
                else if (path.Contains("TouchPro"))
                {
                    proControllerScenes.Add(new Tuple<int, string>(sceneIndex, path));
                }
                else
                {
                    generalScenes.Add(new Tuple<int, string>(sceneIndex, path));
                }
            }

            var uiBuilder = DebugUIBuilder.Instance;
            
            // Add server configuration section
            _ = uiBuilder.AddLabel("Server Configuration", DebugUIBuilder.DEBUG_PANE_CENTER);
            _ = uiBuilder.AddLabel($"Current Server: {tempServerURL}", DebugUIBuilder.DEBUG_PANE_CENTER);
            _ = uiBuilder.AddButton("Change Server URL", PromptForServerURL, -1, DebugUIBuilder.DEBUG_PANE_CENTER);
            _ = uiBuilder.AddDivider(DebugUIBuilder.DEBUG_PANE_CENTER);
            
            // Add anime mode selection section
            _ = uiBuilder.AddLabel("Anime Processing Mode", DebugUIBuilder.DEBUG_PANE_CENTER);
            _ = uiBuilder.AddLabel($"Current Mode: {CurrentAnimeMode}", DebugUIBuilder.DEBUG_PANE_CENTER);
            _ = uiBuilder.AddButton("Switch Anime Mode", SwitchAnimeMode, -1, DebugUIBuilder.DEBUG_PANE_CENTER);
            _ = uiBuilder.AddDivider(DebugUIBuilder.DEBUG_PANE_CENTER);
            
            if (passthroughScenes.Count > 0)
            {
                _ = uiBuilder.AddLabel("Passthrough Sample Scenes", DebugUIBuilder.DEBUG_PANE_LEFT);
                foreach (var scene in passthroughScenes)
                {
                    _ = uiBuilder.AddButton(Path.GetFileNameWithoutExtension(scene.Item2), () => LoadScene(scene.Item1), -1, DebugUIBuilder.DEBUG_PANE_LEFT);
                }
            }

            if (proControllerScenes.Count > 0)
            {
                _ = uiBuilder.AddLabel("Pro Controller Sample Scenes", DebugUIBuilder.DEBUG_PANE_RIGHT);
                foreach (var scene in proControllerScenes)
                {
                    _ = uiBuilder.AddButton(Path.GetFileNameWithoutExtension(scene.Item2), () => LoadScene(scene.Item1), -1, DebugUIBuilder.DEBUG_PANE_RIGHT);
                }
            }

            _ = uiBuilder.AddLabel("Press ☰ at any time to return to scene selection", DebugUIBuilder.DEBUG_PANE_CENTER);
            if (generalScenes.Count > 0)
            {
                _ = uiBuilder.AddDivider(DebugUIBuilder.DEBUG_PANE_CENTER);
                _ = uiBuilder.AddLabel("Sample Scenes", DebugUIBuilder.DEBUG_PANE_CENTER);
                foreach (var scene in generalScenes)
                {
                    _ = uiBuilder.AddButton(Path.GetFileNameWithoutExtension(scene.Item2), () => LoadScene(scene.Item1), -1, DebugUIBuilder.DEBUG_PANE_CENTER);
                }
            }

            uiBuilder.Show();
        }

        private void PromptForServerURL()
        {
            // Simple preset options instead of text field
            var uiBuilder = DebugUIBuilder.Instance;
            uiBuilder.Hide();
            
            var newBuilder = DebugUIBuilder.Instance;
            _ = newBuilder.AddLabel("Select Server:", DebugUIBuilder.DEBUG_PANE_CENTER);
            _ = newBuilder.AddButton("94.101.98.96:34537 (Default)", () => SetServerURL("http://94.101.98.96:34537"), -1, DebugUIBuilder.DEBUG_PANE_CENTER);
            _ = newBuilder.AddButton("localhost:3035 (Local)", () => SetServerURL("http://localhost:3035"), -1, DebugUIBuilder.DEBUG_PANE_CENTER);
            _ = newBuilder.AddButton("192.168.1.100:3035 (Custom)", () => SetServerURL("http://192.168.1.100:3035"), -1, DebugUIBuilder.DEBUG_PANE_CENTER);
            _ = newBuilder.AddButton("Back", () => { newBuilder.Hide(); Start(); }, -1, DebugUIBuilder.DEBUG_PANE_CENTER);
            newBuilder.Show();
        }

        private void SetServerURL(string newURL)
        {
            ServerURL = newURL;
            tempServerURL = newURL;
            Debug.Log($"Server URL updated to: {ServerURL}");
            
            // Return to main menu
            DebugUIBuilder.Instance.Hide();
            Start();
        }

        private void SwitchAnimeMode()
        {
            CurrentAnimeMode = (CurrentAnimeMode == AnimeMode.Diffusion) ? AnimeMode.GAN : AnimeMode.Diffusion;
            Debug.Log($"Anime mode switched to: {CurrentAnimeMode}");
            
            // Refresh the menu to show updated mode
            DebugUIBuilder.Instance.Hide();
            Start();
        }

        private void LoadScene(int idx)
        {
            DebugUIBuilder.Instance.Hide();
            Debug.Log("Load scene: " + idx);
            UnityEngine.SceneManagement.SceneManager.LoadScene(idx);
        }
    }
}
