using UnityEngine;
using UnityEngine.UI;

public class UIManager : MonoBehaviour
{
    public Text scoreText;
    public Text healthText;
    public Text ammoText;
    public Text weaponText;
    
    public GameObject pauseMenu;
    public GameObject gameOverMenu;
    
    private PlayerController player;
    private WeaponSystem weapon;
    
    void Start()
    {
        player = FindObjectOfType<PlayerController>();
        weapon = FindObjectOfType<WeaponSystem>();
    }
    
    void Update()
    {
        UpdateHUD();
        
        // 显示/隐藏菜单
        pauseMenu.SetActive(GameManager.Instance.isGamePaused);
        gameOverMenu.SetActive(GameManager.Instance.isGameOver);
    }
    
    void UpdateHUD()
    {
        if (player != null)
        {
            healthText.text = "Health: " + player.health;
        }
        
        if (weapon != null)
        {
            weaponText.text = "Weapon: " + weapon.currentWeapon;
            
            switch (weapon.currentWeapon)
            {
                case WeaponSystem.WeaponType.Pistol:
                    ammoText.text = "Ammo: " + weapon.pistolAmmo + "/12";
                    break;
                case WeaponSystem.WeaponType.Rifle:
                    ammoText.text = "Ammo: " + weapon.rifleAmmo + "/30";
                    break;
                case WeaponSystem.WeaponType.Shotgun:
                    ammoText.text = "Ammo: " + weapon.shotgunAmmo + "/6";
                    break;
            }
        }
        
        scoreText.text = "Score: " + GameManager.Instance.score;
    }
    
    // UI按钮方法
    public void OnResumeButton()
    {
        GameManager.Instance.TogglePause();
    }
    
    public void OnRestartButton()
    {
        GameManager.Instance.RestartGame();
    }
    
    public void OnQuitButton()
    {
        GameManager.Instance.QuitGame();
    }
}