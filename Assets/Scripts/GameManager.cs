using UnityEngine;

public class GameManager : MonoBehaviour
{
    public static GameManager Instance { get; private set; }
    
    [Header("游戏设置")]
    public int playerHealth = 100;
    public int playerScore = 0;
    public bool isGamePaused = false;
    
    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }
    
    public void AddScore(int points)
    {
        playerScore += points;
        Debug.Log($"得分: {playerScore}");
    }
    
    public void TakeDamage(int damage)
    {
        playerHealth -= damage;
        Debug.Log($"生命值: {playerHealth}");
        
        if (playerHealth <= 0)
        {
            GameOver();
        }
    }
    
    private void GameOver()
    {
        Debug.Log("游戏结束!");
        // 这里可以添加游戏结束逻辑
    }
    
    public void TogglePause()
    {
        isGamePaused = !isGamePaused;
        Time.timeScale = isGamePaused ? 0f : 1f;
        Debug.Log(isGamePaused ? "游戏暂停" : "游戏继续");
    }
}