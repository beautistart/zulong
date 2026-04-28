using UnityEngine;

public class EnemyAI : MonoBehaviour
{
    public int health = 50;
    public float moveSpeed = 3f;
    public float attackRange = 15f;
    public float attackCooldown = 2f;
    public float damage = 10f;
    
    private Transform player;
    private float nextAttackTime;
    private bool isChasing = false;
    
    void Start()
    {
        player = GameObject.FindGameObjectWithTag("Player").transform;
    }
    
    void Update()
    {
        float distanceToPlayer = Vector3.Distance(transform.position, player.position);
        
        if (distanceToPlayer <= attackRange)
        {
            isChasing = true;
            ChasePlayer();
            
            if (distanceToPlayer <= attackRange / 2 && Time.time >= nextAttackTime)
            {
                Attack();
            }
        }
        else
        {
            isChasing = false;
            Patrol();
        }
    }
    
    void ChasePlayer()
    {
        // 朝向玩家
        transform.LookAt(player);
        
        // 向玩家移动
        if (Vector3.Distance(transform.position, player.position) > 2f)
        {
            transform.Translate(Vector3.forward * moveSpeed * Time.deltaTime);
        }
    }
    
    void Patrol()
    {
        // 简单的巡逻逻辑
        transform.Translate(Vector3.forward * moveSpeed * 0.5f * Time.deltaTime);
        
        // 随机转向
        if (Random.Range(0, 100) < 2)
        {
            transform.Rotate(0, Random.Range(-90, 90), 0);
        }
    }
    
    void Attack()
    {
        // 攻击玩家
        player.GetComponent<PlayerController>()?.TakeDamage((int)damage);
        nextAttackTime = Time.time + attackCooldown;
        Debug.Log("Enemy attacked player for " + damage + " damage");
    }
    
    public void TakeDamage(float damage)
    {
        health -= (int)damage;
        if (health <= 0)
        {
            Die();
        }
    }
    
    void Die()
    {
        // 敌人死亡逻辑
        Debug.Log("Enemy died!");
        Destroy(gameObject);
    }
}