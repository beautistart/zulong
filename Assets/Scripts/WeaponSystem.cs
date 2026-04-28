using UnityEngine;

public class WeaponSystem : MonoBehaviour
{
    public enum WeaponType { Pistol, Rifle, Shotgun }
    
    public WeaponType currentWeapon = WeaponType.Pistol;
    public int pistolAmmo = 12;
    public int rifleAmmo = 30;
    public int shotgunAmmo = 6;
    
    public float pistolDamage = 25f;
    public float rifleDamage = 15f;
    public float shotgunDamage = 40f;
    
    public float pistolFireRate = 0.5f;
    public float rifleFireRate = 0.1f;
    public float shotgunFireRate = 1f;
    
    private float nextFireTime = 0f;
    
    void Update()
    {
        // 武器切换
        if (Input.GetKeyDown(KeyCode.1)) currentWeapon = WeaponType.Pistol;
        if (Input.GetKeyDown(KeyCode.2)) currentWeapon = WeaponType.Rifle;
        if (Input.GetKeyDown(KeyCode.3)) currentWeapon = WeaponType.Shotgun;
        
        // 射击
        if (Input.GetButton("Fire1") && Time.time >= nextFireTime)
        {
            Shoot();
        }
        
        // 换弹
        if (Input.GetKeyDown(KeyCode.R))
        {
            Reload();
        }
    }
    
    void Shoot()
    {
        switch (currentWeapon)
        {
            case WeaponType.Pistol:
                if (pistolAmmo > 0)
                {
                    pistolAmmo--;
                    nextFireTime = Time.time + pistolFireRate;
                    FireWeapon(pistolDamage);
                }
                break;
            case WeaponType.Rifle:
                if (rifleAmmo > 0)
                {
                    rifleAmmo--;
                    nextFireTime = Time.time + rifleFireRate;
                    FireWeapon(rifleDamage);
                }
                break;
            case WeaponType.Shotgun:
                if (shotgunAmmo > 0)
                {
                    shotgunAmmo--;
                    nextFireTime = Time.time + shotgunFireRate;
                    FireWeapon(shotgunDamage);
                }
                break;
        }
    }
    
    void FireWeapon(float damage)
    {
        RaycastHit hit;
        if (Physics.Raycast(Camera.main.transform.position, Camera.main.transform.forward, out hit, 100f))
        {
            if (hit.collider.CompareTag("Enemy"))
            {
                hit.collider.GetComponent<EnemyAI>()?.TakeDamage(damage);
            }
        }
        
        // 播放射击音效和特效
        Debug.Log("Fired weapon: " + currentWeapon + " with damage: " + damage);
    }
    
    void Reload()
    {
        switch (currentWeapon)
        {
            case WeaponType.Pistol:
                pistolAmmo = 12;
                break;
            case WeaponType.Rifle:
                rifleAmmo = 30;
                break;
            case WeaponType.Shotgun:
                shotgunAmmo = 6;
                break;
        }
        Debug.Log("Reloaded: " + currentWeapon);
    }
}