using System.Collections;
using System.Collections.Generic;
using Meta.XR.Samples;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using System;
using System.Net.WebSockets;
using System.Threading;
using UnityEngine.XR;
using PassthroughCameraSamples.StartScene;

namespace PassthroughCameraSamples.CameraViewer
{
    [MetaCodeSample("PassthroughCameraApiSamples-CameraViewer")]
    public class CameraViewerManager : MonoBehaviour
    {
        [SerializeField] private WebCamTextureManager m_webCamTextureManager;
        [SerializeField] private Text m_debugText;
        [SerializeField] private RawImage m_image;
        [SerializeField] private RawImage m_animeImage;
        
        private string SERVER_URL => $"{StartMenu.ServerURL}/upload";
        private string WEBSOCKET_URL => $"ws://{StartMenu.ServerURL.Replace("http://", "").Replace("https://", "")}/";
        
        [SerializeField] private int m_jpegQuality = 30;
        [SerializeField] private int m_parallelUploads = 1;
        [SerializeField] private int m_maxQueueSize = 1;

        private float m_frameCount = 0;
        private float m_dt = 0.0f;
        private float m_fps = 0.0f;
        private float m_updateRate = 2.0f;

        private Texture2D m_screenshotTexture;
        private Texture2D m_animeTexture;

        private Queue<byte[]> m_imageQueue = new Queue<byte[]>();
        private Queue<byte[]> m_receivedImages = new Queue<byte[]>();
        private Queue<Texture2D> m_texturePool = new Queue<Texture2D>();
        private const int TEXTURE_POOL_SIZE = 3;

        private ClientWebSocket m_webSocket;
        private CancellationTokenSource m_cancellationTokenSource;

        private bool m_showAnimeView = true;
        private bool m_wasButtonPressed = false;

        private int m_frameSkip = 0;
        private const int FRAME_SKIP_COUNT = 0;

        private string m_statusMessage = "";
        private int m_webSocketConnectAttempts = 0;

        private float m_receivedFrameCount = 0;
        private float m_receivedDt = 0.0f;
        private float m_receivedFps = 0.0f;

        private float m_averageUploadTime = 0f;
        private int m_uploadCount = 0;
        
        private int m_totalImagesReceived = 0;
        private int m_validImagesProcessed = 0;

        private IEnumerator Start()
        {
            while (m_webCamTextureManager.WebCamTexture == null)
            {
                yield return null;
            }
            
            m_image.texture = m_webCamTextureManager.WebCamTexture;

            m_screenshotTexture = new Texture2D(1024, 1024, TextureFormat.RGB24, false);
            m_animeTexture = new Texture2D(2, 2);
            m_animeImage.texture = m_animeTexture;
            
            for (int i = 0; i < TEXTURE_POOL_SIZE; i++)
            {
                Texture2D poolTexture = new Texture2D(2, 2, TextureFormat.RGB24, false);
                m_texturePool.Enqueue(poolTexture);
            }
            
            if (m_animeImage != null)
            {
                m_animeImage.raycastTarget = false;
                
                CanvasGroup canvasGroup = m_animeImage.GetComponentInParent<CanvasGroup>();
                if (canvasGroup != null)
                {
                    canvasGroup.alpha = 1f;
                }
            }

            StartCoroutine(ContinuousScreenshotCapture());
            
            for (int i = 0; i < m_parallelUploads; i++)
            {
                StartCoroutine(UploadWorker());
            }
            
            StartCoroutine(ConnectWebSocket());
            StartCoroutine(ProcessReceivedImages());
        }

        private void Update()
        {
            m_frameCount++;
            m_dt += Time.unscaledDeltaTime;
            if (m_dt > 1.0f / m_updateRate)
            {
                m_fps = m_frameCount / m_dt;
                m_frameCount = 0;
                m_dt -= 1.0f / m_updateRate;

                string uploadInfo = m_averageUploadTime > 0 ? $"Avg:{m_averageUploadTime:F0}ms" : "";
                string queueInfo = $" Q:{m_imageQueue.Count}→{m_receivedImages.Count}";
                string imgStats = m_totalImagesReceived > 0 ? $" Imgs:{m_validImagesProcessed}/{m_totalImagesReceived}" : "";
                m_debugText.text = $"Send:{m_fps:F0} Recv:{m_receivedFps:F1}{uploadInfo}\n{SERVER_URL}{imgStats}\nMode:{StartMenu.CurrentAnimeMode}/{m_statusMessage}";
            }
            
            m_receivedDt += Time.unscaledDeltaTime;
            if (m_receivedDt > 1.0f / m_updateRate)
            {
                m_receivedFps = m_receivedFrameCount / m_receivedDt;
                m_receivedFrameCount = 0;
                m_receivedDt -= 1.0f / m_updateRate;
            }

            bool buttonPressed = false;
            
            InputDevice rightController = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);
            if (rightController.isValid)
            {
                rightController.TryGetFeatureValue(CommonUsages.primaryButton, out bool aButton);
                rightController.TryGetFeatureValue(CommonUsages.triggerButton, out bool trigger);
                buttonPressed = aButton || trigger;
            }
            
            if (!buttonPressed)
            {
                InputDevice leftController = InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
                if (leftController.isValid)
                {
                    leftController.TryGetFeatureValue(CommonUsages.primaryButton, out bool xButton);
                    leftController.TryGetFeatureValue(CommonUsages.triggerButton, out bool trigger);
                    buttonPressed = xButton || trigger;
                }
            }
            
            if (buttonPressed && !m_wasButtonPressed)
            {
                m_showAnimeView = !m_showAnimeView;
                if (m_animeImage != null)
                {
                    m_animeImage.gameObject.SetActive(m_showAnimeView);
                }
                if (m_image != null)
                {
                    m_image.gameObject.SetActive(!m_showAnimeView);
                }
            }
            
            m_wasButtonPressed = buttonPressed;
        }

        private IEnumerator ContinuousScreenshotCapture()
        {
            RenderTexture persistentRT = new RenderTexture(1024, 1024, 0, RenderTextureFormat.ARGB32);
            persistentRT.Create();
            
            while (true)
            {
                if (m_webCamTextureManager.WebCamTexture.isPlaying)
                {
                    m_frameSkip++;
                    if (m_frameSkip < FRAME_SKIP_COUNT)
                    {
                        yield return null;
                        continue;
                    }
                    m_frameSkip = 0;
                    
                    if (m_imageQueue.Count > 0)
                    {
                        m_imageQueue.Clear();
                    }
                    
                    var webCamTexture = m_webCamTextureManager.WebCamTexture;
                    yield return new WaitForEndOfFrame();

                    Graphics.Blit(webCamTexture, persistentRT);
                    
                    RenderTexture.active = persistentRT;
                    m_screenshotTexture.ReadPixels(new Rect(0, 0, 1024, 1024), 0, 0);
                    m_screenshotTexture.Apply(false);
                    RenderTexture.active = null;

                    byte[] imageData = m_screenshotTexture.EncodeToJPG(m_jpegQuality);
                    m_imageQueue.Enqueue(imageData);
                }
                yield return null;
            }
        }

        private IEnumerator UploadWorker()
        {
            while (true)
            {
                if (m_imageQueue.Count > 0)
                {
                    byte[] imageData = m_imageQueue.Dequeue();

                    WWWForm form = new WWWForm();
                    form.AddBinaryData("image", imageData, "frame.jpg", "image/jpeg");

                    UnityWebRequest www = UnityWebRequest.Post(SERVER_URL, form);
                    
                    www.SetRequestHeader("Accept-Encoding", "gzip, deflate");
                    www.SetRequestHeader("Connection", "keep-alive");
                    
                    if (StartMenu.CurrentAnimeMode == StartMenu.AnimeMode.GAN)
                    {
                        www.SetRequestHeader("x-anime-mode", "gan");
                    }
                    
                    www.timeout = 1;
                    
                    float uploadStart = Time.realtimeSinceStartup;
                    yield return www.SendWebRequest();
                    float uploadTime = (Time.realtimeSinceStartup - uploadStart) * 1000f;

                    if (www.result != UnityWebRequest.Result.Success)
                    {
                        m_statusMessage = $"NET: {www.error}";
                    }
                    else
                    {
                        m_uploadCount++;
                        if (m_uploadCount == 1)
                        {
                            m_averageUploadTime = uploadTime;
                        }
                        else
                        {
                            m_averageUploadTime = (m_averageUploadTime * 0.8f) + (uploadTime * 0.2f);
                        }
                    }

                    www.Dispose();
                }
                yield return null;
            }
        }

        private IEnumerator ConnectWebSocket()
        {
            m_webSocketConnectAttempts++;
            m_statusMessage = $"Connecting WS (attempt {m_webSocketConnectAttempts})...";
            
            m_cancellationTokenSource = new CancellationTokenSource();
            m_webSocket = new ClientWebSocket();

            var uri = new Uri(WEBSOCKET_URL);
            var connectTask = m_webSocket.ConnectAsync(uri, m_cancellationTokenSource.Token);

            yield return new WaitUntil(() => connectTask.IsCompleted);

            if (connectTask.IsFaulted)
            {
                m_statusMessage = $"WS connect failed: {connectTask.Exception?.InnerException?.Message ?? "Unknown"}";
                yield return null;
            }

            if (m_webSocket.State == WebSocketState.Open)
            {
                string message = "{\"type\":\"quest_client\"}";
                byte[] messageBytes = System.Text.Encoding.UTF8.GetBytes(message);
                var sendTask = m_webSocket.SendAsync(new ArraySegment<byte>(messageBytes),
                    WebSocketMessageType.Text, true, m_cancellationTokenSource.Token);

                yield return new WaitUntil(() => sendTask.IsCompleted);

                if (sendTask.IsFaulted)
                {
                    m_statusMessage = $"WS send failed: {sendTask.Exception?.InnerException?.Message ?? "Unknown"}";
                }
                else
                {
                    m_statusMessage = "WS Connected";
                    StartCoroutine(ReceiveImages());
                }
            }
            else
            {
                m_statusMessage = $"WS state: {m_webSocket.State}";
            }
        }

        private IEnumerator ReceiveImages()
        {
            var buffer = new byte[1024 * 1024];
            var receivedData = new List<byte>();

            while (m_webSocket.State == WebSocketState.Open)
            {
                var receiveTask = m_webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), m_cancellationTokenSource.Token);

                yield return new WaitUntil(() => receiveTask.IsCompleted);

                if (receiveTask.IsFaulted)
                {
                    m_statusMessage = $"WS receive error: {receiveTask.Exception?.InnerException?.Message ?? "Unknown"}";
                    break;
                }

                var result = receiveTask.Result;

                if (result.MessageType == WebSocketMessageType.Binary)
                {
                    receivedData.AddRange(new ArraySegment<byte>(buffer, 0, result.Count));

                    if (result.EndOfMessage)
                    {
                        m_receivedImages.Enqueue(receivedData.ToArray());
                        m_receivedFrameCount++;
                        receivedData.Clear();
                    }
                }
            }
        }

        private IEnumerator ProcessReceivedImages()
        {
            while (true)
            {
                while (m_receivedImages.Count > 1)
                {
                    m_receivedImages.Dequeue();
                }
                
                if (m_receivedImages.Count > 0)
                {
                    byte[] imageData = m_receivedImages.Dequeue();
                    m_totalImagesReceived++;
                    
                    if (imageData.Length < 500)
                    {
                        yield return null;
                        continue;
                    }
                    
                    bool isJpeg = imageData.Length > 2 && 
                                 imageData[0] == 0xFF && 
                                 imageData[1] == 0xD8;
                    
                    if (!isJpeg)
                    {
                        yield return null;
                        continue;
                    }
                    
                    Texture2D newTexture;
                    if (m_texturePool.Count > 0)
                    {
                        newTexture = m_texturePool.Dequeue();
                    }
                    else
                    {
                        newTexture = new Texture2D(2, 2, TextureFormat.RGB24, false);
                    }
                    
                    bool loadSuccess = newTexture.LoadImage(imageData, false);
                    
                    if (loadSuccess)
                    {
                        m_validImagesProcessed++;
                        
                        Texture2D oldTexture = m_animeTexture;
                        m_animeTexture = newTexture;

                        if (m_animeImage != null)
                        {
                            m_animeImage.texture = m_animeTexture;
                            m_animeImage.SetAllDirty();
                        }
                        
                        yield return null;
                        
                        if (oldTexture != null && oldTexture != m_animeTexture)
                        {
                            if (m_texturePool.Count < TEXTURE_POOL_SIZE && oldTexture.width >= 32 && oldTexture.height >= 32)
                            {
                                m_texturePool.Enqueue(oldTexture);
                            }
                            else
                            {
                                Destroy(oldTexture);
                            }
                        }
                    }
                    else
                    {
                        if (m_texturePool.Count < TEXTURE_POOL_SIZE)
                        {
                            m_texturePool.Enqueue(newTexture);
                        }
                        else
                        {
                            Destroy(newTexture);
                        }
                    }
                }
                
                yield return null;
            }
        }

        private void OnDestroy()
        {
            if (m_screenshotTexture != null)
            {
                Destroy(m_screenshotTexture);
            }
            
            if (m_animeTexture != null)
            {
                Destroy(m_animeTexture);
            }
            
            while (m_texturePool.Count > 0)
            {
                Texture2D poolTexture = m_texturePool.Dequeue();
                if (poolTexture != null)
                {
                    Destroy(poolTexture);
                }
            }
            
            if (m_cancellationTokenSource != null)
            {
                m_cancellationTokenSource.Cancel();
            }
            
            if (m_webSocket != null && m_webSocket.State == WebSocketState.Open)
            {
                try
                {
                    var closeTask = m_webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None);
                    closeTask.Wait(TimeSpan.FromSeconds(1));
                }
                catch { }
                m_webSocket.Dispose();
            }
        }
    }
}