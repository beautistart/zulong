"""
简单计算器主程序
提供命令行用户界面
"""

import calculator

def display_welcome():
    """显示欢迎信息和操作指南"""
    print("=" * 40)
    print("          简单计算器")
    print("=" * 40)
    print("支持运算: + (加), - (减), * (乘), / (除)")
    print("输入格式: 数字1 运算符 数字2")
    print("示例: 5 + 3")
    print("输入 'quit' 退出程序")
    print("=" * 40)

def get_user_input():
    """获取用户输入并解析"""
    while True:
        try:
            user_input = input("\n请输入计算表达式: ").strip()
            
            if user_input.lower() == 'quit':
                return None, None, None
            
            # 分割输入
            parts = user_input.split()
            if len(parts) != 3:
                print("错误：请输入正确的格式 (例如: 5 + 3)")
                continue
            
            a = float(parts[0])
            operator = parts[1]
            b = float(parts[2])
            
            return a, b, operator
            
        except ValueError:
            print("错误：请输入有效的数字")
        except KeyboardInterrupt:
            print("\n程序已退出")
            return None, None, None

def main():
    """主程序循环"""
    display_welcome()
    
    while True:
        a, b, operator = get_user_input()
        
        if a is None:  # 用户选择退出
            break
        
        try:
            result = calculator.calculate(a, b, operator)
            print(f"结果: {a} {operator} {b} = {result}")
            
        except ValueError as e:
            print(e)
        except Exception as e:
            print(f"发生未知错误: {e}")

if __name__ == "__main__":
    main()